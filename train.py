#!/usr/bin/env python3
"""Train one multilingual-macaroni condition into models/<condition>[-sNN]/.

Usage:
    python train.py --corpus switch [--seed 42] [--config configs/train.yaml]
    python train.py --corpus curriculum --seed 43

This is a thin wrapper around the external trainer cloned by setup.sh into
`multilingual-training/train.py`. It fills the per-condition fields of
`configs/train.yaml` (data path, output dir, seeds, tokenizer, and, for the
curriculum conditions, the per-stage checkpoint_dir / init_new_model) and shells
out to the trainer once per run. It does NOT re-implement the HF Trainer.

Conditions:
  * single-run  : switch, noswitch, word, sent, par, salad
                  -> one trainer run over data/<corpus>/data.parquet
  * curriculum  : curriculum, curriculum_noswitch
                  -> 3 carry-forward stages over the phase parquets
                     data/<corpus>/{1_intra,2_sentence,3_mono}.parquet
                     (stage 1 inits a fresh model; stages 2-3 continue from the
                     previous stage's final checkpoint).
"""
import argparse
import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# Repo root == this file's directory. final_checkpoint.py lives here.
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from final_checkpoint import final_checkpoint  # noqa: E402

SINGLE_RUN = ["switch", "noswitch", "word", "sent", "par", "salad"]
CURRICULUM = ["curriculum", "curriculum_noswitch"]
CONDITIONS = SINGLE_RUN + CURRICULUM

# Curriculum phase parquets, in curriculum order (matches download_data.py layout).
CURRICULUM_PHASES = ["1_intra", "2_sentence", "3_mono"]

# Curriculum runs save an extra early milestone on top of the single-run schedule,
# mirroring scripts/run_cscl_curriculum.py (MINIMAL_SAVE = [1.0, 5.0, 10.0]).
CURRICULUM_SAVE_INTERVALS = [1.0, 5.0, 10.0]


def condition_label(corpus: str, seed: int) -> str:
    """seed 42 -> bare corpus name; other seeds -> '<corpus>-sNN'."""
    return corpus if seed == 42 else f"{corpus}-s{seed}"


def resolve_tokenizer_dir(tokenizer: str | None, models_dir: Path) -> Path:
    """Resolve a local tokenizer directory containing tokenizer.json.

    The external trainer requires a LOCAL tokenizer dir (it checks for
    tokenizer.json). Resolution order:
      1. --tokenizer points at a local dir with tokenizer.json -> use it.
      2. --tokenizer looks like an HF id (or a path without tokenizer.json)
         -> materialize it locally under <models_dir>/_tokenizer/.
      3. --tokenizer omitted -> reuse the tokenizer shipped inside any already
         downloaded model under <models_dir>/*/ (first dir with tokenizer.json).
    """
    if tokenizer:
        p = Path(tokenizer)
        if (p / "tokenizer.json").is_file():
            return p.resolve()
        if p.is_dir():
            raise SystemExit(
                f"--tokenizer '{tokenizer}' is a directory but has no tokenizer.json."
            )
        # Treat as an HF id (or remote path) and materialize it locally so the
        # trainer's local-dir requirement is satisfied.
        # NOTE: requires network + `transformers`; only triggered for an HF id.
        staged = (models_dir / "_tokenizer").resolve()
        print(f"[tokenizer] downloading '{tokenizer}' -> {staged}")
        from transformers import AutoTokenizer  # lazy import
        tok = AutoTokenizer.from_pretrained(tokenizer)
        tok.save_pretrained(str(staged))
        return staged

    # No tokenizer given: reuse one from a downloaded model.
    if models_dir.is_dir():
        for cand in sorted(models_dir.iterdir()):
            if cand.is_dir() and (cand / "tokenizer.json").is_file():
                print(f"[tokenizer] reusing tokenizer from {cand}")
                return cand.resolve()
    raise SystemExit(
        "No tokenizer found. Pass --tokenizer <path-or-hfid>, or download a model "
        "first (e.g. `python download_models.py --only switch`) so its bundled "
        "tokenizer can be reused."
    )


def stage_dataset_dir(parquet_paths: list[Path], stage_dir: Path) -> Path:
    """Symlink the given parquet file(s) into a clean directory and return it.

    The trainer's data loader (multilingual-training/data.py) treats a single
    local FILE path as a plain-text file; only a DIRECTORY of parquets is loaded
    as parquet. We therefore always hand it an isolated directory containing
    exactly the parquet(s) we want for this run/stage.
    """
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    for src in parquet_paths:
        if not src.is_file():
            raise SystemExit(
                f"Missing data file: {src}\n"
                f"Download the corpus first, e.g. `python download_data.py`."
            )
        (stage_dir / src.name).symlink_to(src.resolve())
    return stage_dir.resolve()


def load_base_config(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    if not isinstance(cfg, dict):
        raise SystemExit(f"{config_path} must contain a YAML mapping.")
    return cfg


def run_trainer(trainer_dir: Path, cfg: dict, cfg_path: Path) -> None:
    """Write cfg to cfg_path and shell out to {trainer_dir}/train.py --config."""
    trainer_py = trainer_dir / "train.py"
    if not trainer_py.is_file():
        raise SystemExit(
            f"Trainer not found at {trainer_py}.\n"
            f"Run `bash setup.sh` to clone the external trainer into "
            f"'{trainer_dir}'."
        )
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    # The external trainer hard-requires WANDB_ENTITY / WANDB_PROJECT and reports
    # to wandb. For a hands-off reproducibility run we default to offline mode
    # (only filling anything the caller has NOT already set) so training never
    # blocks on a wandb login.
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("WANDB_PROJECT", "multilingual-macaroni")
    env.setdefault("WANDB_ENTITY", "multilingual-macaroni")

    cmd = [sys.executable, str(trainer_py), "--config", str(cfg_path)]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def train_single(args, base_cfg, tokenizer_dir, model_config, out_dir, label):
    data_dir = Path(args.data_dir) / args.corpus
    parquet = data_dir / "data.parquet"
    staged = stage_dataset_dir([parquet], out_dir / "_data")

    cfg = copy.deepcopy(base_cfg)
    cfg["datasets"] = str(staged)
    cfg["tokenizer_dir"] = str(tokenizer_dir)
    cfg["model_config"] = str(model_config)
    cfg["output_dir"] = str(out_dir.resolve())
    cfg["model_name"] = label
    cfg["model_seed"] = args.seed
    cfg["data_seed"] = args.seed
    cfg["init_new_model"] = "True"
    cfg["checkpoint_dir"] = ""

    run_trainer(Path(args.trainer_dir), cfg, out_dir / "run_config.yaml")

    final = final_checkpoint(out_dir)
    print(f"\n[done] {label}: final checkpoint -> {final}")


def train_curriculum(args, base_cfg, tokenizer_dir, model_config, out_dir, label):
    data_dir = Path(args.data_dir) / args.corpus
    cfg_dir = out_dir / "stage_configs"

    prev_ckpt = ""
    for i, phase in enumerate(CURRICULUM_PHASES, start=1):
        stage_out = out_dir / f"stage{i}"
        parquet = data_dir / f"{phase}.parquet"
        staged = stage_dataset_dir([parquet], stage_out / "_data")

        # Clean any stale checkpoint-* first so final_checkpoint() can never carry
        # a previous run's lineage forward across stages (the cscl lineage bug).
        if stage_out.is_dir():
            for stale in sorted(stage_out.glob("checkpoint-*")):
                print(f"    [clean] removing stale {stale}")
                shutil.rmtree(stale, ignore_errors=True)

        cfg = copy.deepcopy(base_cfg)
        cfg["datasets"] = str(staged)
        cfg["tokenizer_dir"] = str(tokenizer_dir)
        cfg["model_config"] = str(model_config)
        cfg["output_dir"] = str(stage_out.resolve())
        cfg["model_name"] = f"{label}-stage{i}"
        cfg["model_seed"] = args.seed
        cfg["data_seed"] = args.seed
        cfg["save_intervals"] = CURRICULUM_SAVE_INTERVALS
        cfg["init_new_model"] = "True" if i == 1 else "False"
        cfg["checkpoint_dir"] = prev_ckpt  # ignored when init_new_model == "True"

        print(f"\n=== Curriculum {label} stage {i}/3: {phase} "
              f"(init_new={cfg['init_new_model']}, from={prev_ckpt or 'scratch'}) ===")
        run_trainer(Path(args.trainer_dir), cfg, cfg_dir / f"stage{i}.yaml")

        ckpt = final_checkpoint(stage_out)
        if ckpt is None:
            raise SystemExit(f"stage {i} produced no checkpoint under {stage_out}")
        prev_ckpt = str(ckpt)

    print(f"\n[done] {label}: final checkpoint -> {prev_ckpt}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, choices=CONDITIONS,
                    help="training condition / corpus subset name")
    ap.add_argument("--seed", type=int, default=42,
                    help="model & data seed (42 -> bare model dir name)")
    ap.add_argument("--config", default="configs/train.yaml",
                    help="base training recipe YAML")
    ap.add_argument("--data-dir", default="data",
                    help="root of downloaded corpora (data/<corpus>/...)")
    ap.add_argument("--models-dir", default="models",
                    help="root to write trained models into")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer dir or HF id (default: reuse the tokenizer "
                         "inside any downloaded model under --models-dir)")
    ap.add_argument("--trainer-dir", default="multilingual-training",
                    help="cloned external trainer repo (see setup.sh)")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")
    base_cfg = load_base_config(config_path)

    models_dir = Path(args.models_dir)
    label = condition_label(args.corpus, args.seed)
    out_dir = models_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = resolve_tokenizer_dir(args.tokenizer, models_dir)

    # Resolve the arch config to an absolute path so the trainer finds it
    # regardless of its working directory.
    model_config = Path(base_cfg.get("model_config", "configs/gpt2-baseline.json"))
    if not model_config.is_absolute():
        model_config = (REPO_ROOT / model_config).resolve()
    if not model_config.is_file():
        raise SystemExit(f"Model config not found: {model_config}")

    print(f"corpus     : {args.corpus}")
    print(f"seed       : {args.seed}")
    print(f"model dir  : {out_dir.resolve()}")
    print(f"tokenizer  : {tokenizer_dir}")

    if args.corpus in CURRICULUM:
        train_curriculum(args, base_cfg, tokenizer_dir, model_config, out_dir, label)
    else:
        train_single(args, base_cfg, tokenizer_dir, model_config, out_dir, label)


if __name__ == "__main__":
    main()
