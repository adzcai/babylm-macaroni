"""Config + checkpoint discovery for the word-level pipeline (Figs 4 and 8).

The paper selected a per-seed ``config_v2_s<seed>.yaml`` whose checkpoint paths
pointed under a private tree. Here a config is a plain dict built from the repo's
portable roots, and checkpoints are discovered under ``models/<condition>[-sNN]/``.

Two arms are compared, keyed by the short names the paper's code uses:

    cs        -> curriculum            (the code-switched curriculum)
    noswitch  -> curriculum_noswitch   (its non-CS twin)

Figures 4 and 8 read only the FINAL checkpoint of each arm (published on the
Hub). ``discover_checkpoints`` additionally locates the ten trajectory
checkpoints per arm (init + epochs 1/5/10 of each stage) when they exist on disk
(``models/<condition>/stage{1,2,3}/checkpoint-*``, as written by ``train.py``).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ARM_CONDITION = {"cs": "curriculum", "noswitch": "curriculum_noswitch"}
STAGE_PHASE = {"stage1": "1_intra", "stage2": "2_sentence", "stage3": "3_mono"}
CKPT_ORDER = [("stage1", 0), ("stage1", 1), ("stage1", 5), ("stage1", 10),
              ("stage2", 1), ("stage2", 5), ("stage2", 10),
              ("stage3", 1), ("stage3", 5), ("stage3", 10)]
FINAL = ("stage3", 10)


class CheckpointsMissing(FileNotFoundError):
    """Raised when required checkpoints are not on disk (message lists the gaps)."""


def build_config(models_dir, data_dir, out_dir, seed: int, babylm_dir=None,
                 layer_aggregate=None) -> dict:
    """Pipeline knobs (verbatim from the paper's config_v2_s*.yaml) + portable roots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)
    return {
        "models_dir": Path(models_dir), "data_dir": data_dir, "out_dir": out_dir,
        "babylm_dir": Path(babylm_dir) if babylm_dir else data_dir / "babylm",
        "seed": int(seed),
        "k_contexts": 20, "caliper_log10": 0.1, "pos_purity_min": 0.8,
        "max_sent_chars": 300, "min_sent_chars": 10,
        "layer_aggregate": list(layer_aggregate) if layer_aggregate else [6, 7, 8, 9, 10, 11],
        "batch_size": 64, "bootstrap_n": 2000, "bootstrap_seed": 0,
        "arms": dict(ARM_CONDITION),
    }


def phase_parquet(cfg, condition: str, stage: str) -> Path:
    return cfg["data_dir"] / condition / f"{STAGE_PHASE[stage]}.parquet"


def tokenizer_json(cfg) -> Path:
    """The shared tokenizer.json, taken from the curriculum model's weights dir."""
    import common
    ck = common.final_ckpt(common.model_dir("curriculum", cfg["seed"], cfg["models_dir"]))
    tj = Path(ck) / "tokenizer.json"
    if not tj.is_file():
        raise CheckpointsMissing(
            f"tokenizer.json not found under {ck}. Download the curriculum model first: "
            f"python download_models.py --only curriculum --seeds {cfg['seed']}")
    return tj


def final_checkpoints(cfg):
    """``[(arm, 'stage3', 10, weights_dir)]`` for both arms' final checkpoints."""
    import common
    out, missing = [], []
    for arm, condition in cfg["arms"].items():
        md = common.model_dir(condition, cfg["seed"], cfg["models_dir"])
        try:
            out.append((arm, *FINAL, Path(common.final_ckpt(md))))
        except FileNotFoundError as e:
            missing.append(f"  [{arm}={condition}] {e}")
    if missing:
        raise CheckpointsMissing("final checkpoints missing:\n" + "\n".join(missing))
    return out


def _read_epoch(ckpt: Path):
    st = ckpt / "trainer_state.json"
    if not st.is_file():
        return None
    try:
        return int(round(float(json.loads(st.read_text()).get("epoch", -1))))
    except Exception:
        return None


def _find_checkpoint(model_root: Path, stage: str, epoch: int):
    d = model_root / stage
    if not d.is_dir():
        return None
    if stage == "stage1" and epoch == 0:
        c0 = d / "checkpoint-0"
        return c0 if (c0 / "config.json").is_file() else None
    for ck in sorted(d.glob("checkpoint-*")):
        if (ck / "config.json").is_file() and _read_epoch(ck) == epoch:
            return ck
    return None


def discover_checkpoints(cfg):
    """``[(arm, stage, epoch, ckpt_dir)]`` for all ten trajectory points of both arms.

    Raises ``CheckpointsMissing`` (listing every gap) if any point is absent: a
    trajectory is never assembled from a partial set.
    """
    import common
    found, missing = [], []
    for arm, condition in cfg["arms"].items():
        root = common.model_dir(condition, cfg["seed"], cfg["models_dir"])
        for stage, epoch in CKPT_ORDER:
            ck = _find_checkpoint(root, stage, epoch)
            if ck is None:
                missing.append(f"  [{arm}={condition}] {stage} ep{epoch}: not under {root}/{stage}/")
            else:
                found.append((arm, stage, epoch, Path(ck)))
    if missing:
        raise CheckpointsMissing(
            "Per-checkpoint TRAJECTORY models are missing:\n" + "\n".join(missing) +
            "\n\nThe Hub ships only final checkpoints. Train both curriculum arms "
            "locally (python train.py --corpus curriculum / curriculum_noswitch), "
            "which saves checkpoint-0 and the epoch 1/5/10 checkpoints of every stage "
            "under models/<condition>[-sNN]/stage{1,2,3}/.")
    return found


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_strings(items) -> str:
    h = hashlib.sha256()
    for s in items:
        h.update(s.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()
