#!/usr/bin/env python3
"""Evaluate a model dir or HF id; write results under out/eval/<label>/.

Usage:
    python evaluate.py --model models/curriculum
    python evaluate.py --model BabyLM-community/BabyLM-2026-Baseline-GPT2-en_nld_zho_equal \
        --tasks zeroshot --label baseline

Runs the two halves of the BabyLM multilingual eval (cloned by setup.sh into
`babylm-eval/`) against `babylm-eval/multilingual`:

  * zero-shot minimal-pair suites (incl. Global PIQA) via lm-eval-harness ->
      out/eval/<label>/zeroshot/<lang>/
  * fine-tuning GLUE-style tasks via finetune_classification.py ->
      out/eval/<label>/finetune/<lang>/<task>/
  * cross-lingual POS tagging via finetune_token_classification.py ->
      out/eval/<label>/finetune/pos/<lang>/eval_results.json

This is a thin orchestrator: it shells out to the eval repo's own entrypoints,
mirroring multilingual/scripts/{zeroshot_model.sh,finetune_model.sh}. It does not
re-implement any scoring.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from final_checkpoint import final_checkpoint  # noqa: E402

# lm-eval zero-shot task groups, one per language.
ZEROSHOT_LANGS = ["eng", "nld", "zho"]

# Fine-tune GLUE-style tasks per language (mirrors slurm/evaluate_finetune.sbatch:
# arc belebele bmlama mnli sib200 truthfulqa, +include for nl/zh, +xnli for en/zh).
FINETUNE_TASKS = {
    "en": ["arc", "belebele", "bmlama", "mnli", "sib200", "truthfulqa", "xnli"],
    "nl": ["arc", "belebele", "bmlama", "include", "mnli", "sib200", "truthfulqa"],
    "zh": ["arc", "belebele", "bmlama", "include", "mnli", "sib200", "truthfulqa", "xnli"],
}

# POS is trained once cross-lingually over all requested languages.
POS_LANGS = ["en", "nl", "zh"]

# Shared fine-tuning hyperparameters (mirrors the source runner scripts).
FT_MAX_SEQ_LEN = "128"
FT_BATCH_SIZE = "64"
FT_LR = "5e-5"
FT_EPOCHS = "10"
FT_PATIENCE = "3"
FT_SEED = "12"


def resolve_model(model: str) -> str:
    """Resolve --model to a usable weights location (dir with config.json / HF id).

    * Non-existent local path -> assume an HF id, return as-is.
    * Local dir with config.json -> use directly (downloaded model or checkpoint).
    * Local dir with checkpoint-* -> pick the lineage-safe final checkpoint.
    * Local curriculum output (stage*/) -> final checkpoint of the last stage.
    """
    p = Path(model)
    if not p.exists():
        return model  # HF id
    if (p / "config.json").is_file():
        return str(p.resolve())
    ck = final_checkpoint(p)
    if ck is not None:
        return str(ck.resolve())
    stages = sorted(p.glob("stage*"))
    for stage in reversed(stages):
        ck = final_checkpoint(stage)
        if ck is not None:
            return str(ck.resolve())
    raise SystemExit(
        f"Could not find usable weights under {p}. Expected a config.json, a "
        f"checkpoint-* dir, or curriculum stage*/ subdirs."
    )


def eval_multilingual_dir(eval_dir: Path) -> Path:
    ml = eval_dir / "multilingual"
    if not ml.is_dir():
        raise SystemExit(
            f"Eval repo not found at {ml}.\n"
            f"Run `bash setup.sh` to clone babylm-eval into '{eval_dir}'."
        )
    return ml


def run(cmd, cwd, env=None):
    print(f"[run] (cwd={cwd}) {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(cwd), env=env)


def run_zeroshot(ml_dir, model, out_root, device):
    out_dir = out_root / "zeroshot"
    out_dir.mkdir(parents=True, exist_ok=True)
    # The Chinese custom scorer marks minimal pairs containing an UNK token
    # (per the evaluated model's tokenizer) as incorrect; point it at our model.
    env = os.environ.copy()
    env.setdefault("HANZI_TOKENIZER", model)
    env.setdefault("HANZI_TOKENIZER_REVISION", "main")
    for lang in ZEROSHOT_LANGS:
        # BOS fix (as used for the paper): dispatch through the eval repo's
        # scripts/run_lm_eval.py, which registers the in-repo `hf-bos` model, and
        # prepend BOS via add_bos_token=True. Mirrors multilingual/scripts/zeroshot_model.sh
        # (runner=run_lm_eval.py, model_flag=hf-bos, extra=,add_bos_token=True).
        run(
            [sys.executable, "scripts/run_lm_eval.py",
             "--model", "hf-bos",
             "--model_args", f"pretrained={model},trust_remote_code=True,add_bos_token=True",
             "--tasks", f"zeroshot_{lang}",
             "--device", device,
             "--output_path", out_dir / lang,
             "--batch_size", "auto:10",
             "--num_fewshot", "0",
             "--log_samples",
             "--include_path", "tasks/"],
            cwd=ml_dir, env=env,
        )


def run_finetune(ml_dir, model, out_root):
    for lang, tasks in FINETUNE_TASKS.items():
        for task in tasks:
            task_out = (out_root / "finetune" / lang / task).resolve()
            task_out.mkdir(parents=True, exist_ok=True)
            train_file = f"finetune/data/multilingual/{lang}/{task}/{task}_{lang}.train.jsonl"
            valid_file = f"finetune/data/multilingual/{lang}/{task}/{task}_{lang}.valid.jsonl"
            # NOTE: finetune_classification.py takes `--do_train True` (a value),
            # matching multilingual/scripts/finetune_model.sh.
            run(
                [sys.executable, "finetune/finetune_classification.py",
                 "--model_name_or_path", model,
                 "--language", lang,
                 "--output_dir", task_out,
                 "--train_file", train_file,
                 "--validation_file", valid_file,
                 "--do_train", "True", "--do_eval", "--do_predict",
                 "--max_seq_length", FT_MAX_SEQ_LEN,
                 "--per_device_train_batch_size", FT_BATCH_SIZE,
                 "--learning_rate", FT_LR,
                 "--num_train_epochs", FT_EPOCHS,
                 "--patience", FT_PATIENCE,
                 "--eval_strategy", "epoch",
                 "--save_strategy", "epoch",
                 "--overwrite_output_dir",
                 "--seed", FT_SEED],
                cwd=ml_dir,
            )


def run_pos(ml_dir, model, out_root):
    pos_out = (out_root / "finetune" / "pos").resolve()
    pos_out.mkdir(parents=True, exist_ok=True)
    # Trained once on a uniform cross-lingual mixture; writes per-language
    # subdirs pos/<lang>/eval_results.json (metric: eval_accuracy).
    # NOTE: finetune_token_classification.py uses a bare `--do_train` flag (no
    # value), matching multilingual/scripts/finetune_model.sh's POS block.
    run(
        [sys.executable, "finetune/finetune_token_classification.py",
         "--model_name_or_path", model,
         "--language", " ".join(POS_LANGS),
         "--output_dir", pos_out,
         "--do_train", "--do_eval", "--do_predict",
         "--max_seq_length", FT_MAX_SEQ_LEN,
         "--per_device_train_batch_size", FT_BATCH_SIZE,
         "--learning_rate", FT_LR,
         "--num_train_epochs", FT_EPOCHS,
         "--patience", FT_PATIENCE,
         "--eval_strategy", "epoch",
         "--save_strategy", "epoch",
         "--overwrite_output_dir",
         "--seed", FT_SEED],
        cwd=ml_dir,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="local model dir (e.g. models/curriculum) or HF id")
    ap.add_argument("--out", default="out",
                    help="output root; results go to <out>/eval/<label>/")
    ap.add_argument("--eval-dir", default="babylm-eval",
                    help="cloned eval repo (see setup.sh)")
    ap.add_argument("--tasks", default="all",
                    choices=["all", "zeroshot", "finetune", "pos"],
                    help="which eval half(s) to run")
    ap.add_argument("--device", default="cuda", help="device for zero-shot eval")
    ap.add_argument("--label", default=None,
                    help="output label (default: basename of --model)")
    args = ap.parse_args()

    ml_dir = eval_multilingual_dir(Path(args.eval_dir))
    model = resolve_model(args.model)
    label = args.label or Path(args.model).name
    out_root = (Path(args.out) / "eval" / label).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"model  : {model}")
    print(f"label  : {label}")
    print(f"out    : {out_root}")
    print(f"tasks  : {args.tasks}")

    if args.tasks in ("all", "zeroshot"):
        run_zeroshot(ml_dir, model, out_root, args.device)
    if args.tasks in ("all", "finetune"):
        run_finetune(ml_dir, model, out_root)
    if args.tasks in ("all", "pos"):
        run_pos(ml_dir, model, out_root)

    print(f"\n[done] results under {out_root}")


if __name__ == "__main__":
    main()
