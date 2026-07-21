"""Config + per-checkpoint discovery for the alignment-trajectory pipeline.

This replaces the paper's ``traj_common.py`` mechanism (a per-seed
``config_v2_s<seed>.yaml`` selected through the ``CS_TRAJ_CONFIG`` env var, whose
paths pointed under a private ``lmkiddo`` tree). Here a config is a plain dict
built from ``(models_dir, data_dir, out_dir, seed)`` — every path is derived from
the three portable roots resolved in ``figures/common.py``.

Two arms are compared, keyed internally by the SAME short names the paper's
statistics code uses (``cs`` / ``noswitch``) so the ported ``measure``/``stats``
modules are byte-faithful. Each arm maps to a public training condition:

    cs        -> curriculum            (the code-switched curriculum)
    noswitch  -> curriculum_noswitch   (the monolingual twin)

CHECKPOINT DISCOVERY. The paper hard-listed ``checkpoint-<step>`` dirs under
``lmkiddo``. This figure needs the model at TEN points along training for BOTH
arms (init, then epochs 1/5/10 within each of the 3 curriculum stages). Those
intermediate checkpoints are discovered on disk under
``model_dir(condition, seed, models_dir)`` — NOT by hard-coded step number
(steps-per-epoch differ per seed/arm), but by reading each candidate
``checkpoint-*/trainer_state.json`` and matching its rounded ``epoch`` within the
stage. See ``discover_checkpoints`` and the ``CheckpointsMissing`` error.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# The internal arm key -> public training condition (download_models.py names).
ARM_CONDITION = {"cs": "curriculum", "noswitch": "curriculum_noswitch"}

# The three curriculum stages -> the phase parquet basename under the corpus
# (data/<condition>/{1_intra,2_sentence,3_mono}.parquet).
STAGE_PHASE = {"stage1": "1_intra", "stage2": "2_sentence", "stage3": "3_mono"}

# The ten measured points, in trajectory order. Epoch 0 of stage1 is the shared
# initialisation (both arms start identical); stages 2/3 have no epoch-0 point
# because their epoch 0 == the previous stage's final checkpoint.
CKPT_ORDER = [("stage1", 0), ("stage1", 1), ("stage1", 5), ("stage1", 10),
              ("stage2", 1), ("stage2", 5), ("stage2", 10),
              ("stage3", 1), ("stage3", 5), ("stage3", 10)]


class CheckpointsMissing(FileNotFoundError):
    """Raised when the per-checkpoint trajectory models are not on disk.

    The published Hub repo ships only each condition's FINAL checkpoint, so the
    intermediate trajectory checkpoints — especially for the no-CS arm — must be
    supplied separately (re-train saving milestone checkpoints, or drop the
    intermediates into the expected layout). The message lists exactly what was
    looked for and what is absent.
    """


def build_config(models_dir, data_dir, out_dir, seed: int,
                 n_layers: int = 12, layer_aggregate=None) -> dict:
    """Build a trajectory-pipeline config from the portable roots.

    Parameters
    ----------
    models_dir, data_dir : path-like
        The repo's ``models/`` and ``data/`` roots (see ``common.resolve_paths``).
    out_dir : path-like
        Per-seed working/output directory; all intermediates land here.
    seed : int
        Model seed (42/43/44). Selects the model branch via ``model_dir``.
    n_layers : int, default 12
        Number of TRANSFORMER layers. Extraction stores every hidden state
        (``n_layers + 1`` including the embedding layer) but sizes its buffers
        from the model at runtime, so this is only a sanity reference.
    layer_aggregate : list[int] | None
        Hidden-state indices pooled into the "upper-layer" estimand. Default is
        the paper's ``[6, 7, 8, 9, 10, 11]`` (0 = embedding … n = top).
    """
    models_dir = Path(models_dir)
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "models_dir": models_dir,
        "data_dir": data_dir,
        "out_dir": out_dir,
        "seed": int(seed),
        # --- pipeline knobs (verbatim from config_v2_s*.yaml) ---
        "k_contexts": 20,
        "caliper_log10": 0.1,
        "pos_purity_min": 0.8,
        "max_sent_chars": 300,
        "min_sent_chars": 10,
        "layer_aggregate": list(layer_aggregate) if layer_aggregate is not None
        else [6, 7, 8, 9, 10, 11],
        "n_layers": int(n_layers),
        "batch_size": 64,
        "bootstrap_n": 2000,
        # bootstrap RNG seed — CONSTANT across model seeds (was config `seed: 0`).
        "bootstrap_seed": 0,
        "arms": dict(ARM_CONDITION),
    }


# --------------------------------------------------------------------------- #
# Corpus / lexicon input locations (all under data_dir), and tokenizer.
# --------------------------------------------------------------------------- #
def phase_parquet(cfg, condition: str, stage: str) -> Path:
    """Corpus parquet for a (condition, stage): data/<condition>/<phase>.parquet."""
    return cfg["data_dir"] / condition / f"{STAGE_PHASE[stage]}.parquet"


def tokenizer_json(cfg) -> Path:
    """The tokenizer.json shared by every model.

    Resolved from the curriculum model's final checkpoint (every downloaded model
    dir carries its tokenizer). Replaces the old absolute ``tokenizer_json`` path.
    """
    import common
    md = common.model_dir("curriculum", cfg["seed"], cfg["models_dir"])
    ck = common.final_ckpt(md)
    tj = Path(ck) / "tokenizer.json"
    if not tj.is_file():
        raise CheckpointsMissing(
            f"tokenizer.json not found under {ck}. Download the curriculum model "
            f"first: python download_models.py --only curriculum --seeds {cfg['seed']}")
    return tj


# --------------------------------------------------------------------------- #
# Per-checkpoint discovery (replaces iter_checkpoints' lmkiddo lookups).
# --------------------------------------------------------------------------- #
def _read_epoch(ckpt: Path):
    """Rounded integer epoch from a checkpoint's trainer_state.json, or None."""
    import json
    st = ckpt / "trainer_state.json"
    if not st.is_file():
        return None
    try:
        return int(round(float(json.loads(st.read_text()).get("epoch", -1))))
    except Exception:
        return None


def _stage_dirs(model_root: Path, stage: str):
    """Candidate directories that may hold a stage's ``checkpoint-*`` dirs.

    Supports both a per-stage subdir layout (``<model>/<stage>/checkpoint-*`` or
    ``<model>/<phase>/checkpoint-*``) and a flat layout (``<model>/checkpoint-*``).
    """
    cands = [model_root / stage, model_root / STAGE_PHASE[stage], model_root]
    return [d for d in cands if d.is_dir()]


def _find_checkpoint(model_root: Path, stage: str, epoch: int):
    """Locate the checkpoint dir for (stage, epoch) under a model root, or None.

    epoch 0 of stage1 is the shared init: prefer an explicit ``checkpoint-0``,
    else the model root itself if it already holds ``config.json``.
    """
    if stage == "stage1" and epoch == 0:
        for d in _stage_dirs(model_root, stage):
            c0 = d / "checkpoint-0"
            if (c0 / "config.json").is_file():
                return c0
        if (model_root / "config.json").is_file():
            return model_root
        return None
    for d in _stage_dirs(model_root, stage):
        for ck in sorted(d.glob("checkpoint-*")):
            if (ck / "config.json").is_file() and _read_epoch(ck) == epoch:
                return ck
    return None


def discover_checkpoints(cfg):
    """Yield ``(arm, stage, epoch, ckpt_dir)`` for every trajectory point.

    Raises ``CheckpointsMissing`` (listing all gaps) if any point is absent —
    fabricating a trajectory from whatever happens to be on disk would silently
    corrupt the DiD, so the pipeline refuses to proceed with a partial set.
    """
    import common
    found, missing = [], []
    for arm, condition in cfg["arms"].items():
        model_root = common.model_dir(condition, cfg["seed"], cfg["models_dir"])
        for stage, epoch in CKPT_ORDER:
            ck = _find_checkpoint(model_root, stage, epoch)
            if ck is None:
                missing.append(f"  [{arm}={condition}] {stage} ep{epoch}: "
                               f"no checkpoint under {model_root}")
            else:
                found.append((arm, stage, epoch, Path(ck)))
    if missing:
        raise CheckpointsMissing(
            "Per-checkpoint TRAJECTORY models are missing (Fig 7 needs the model "
            "at 10 points per arm):\n" + "\n".join(missing) +
            "\n\nThe published Hub repo ships only each condition's FINAL "
            "checkpoint. The intermediate checkpoints — especially for the no-CS "
            "arm (curriculum_noswitch), which are NOT on the Hub — must be "
            "supplied. Re-train each arm saving milestone checkpoints at epochs "
            "0/1/5/10 within each stage, laid out as "
            "models/<condition>[-sNN]/<stage>/checkpoint-<step>/ with a "
            "trainer_state.json recording the epoch, or drop equivalent "
            "intermediates there.")
    return found


# --------------------------------------------------------------------------- #
# Hashing (frozen measurement-set integrity; verbatim from traj_common.py).
# --------------------------------------------------------------------------- #
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_strings(items) -> str:
    """Order-sensitive hash of a list of strings (for the frozen word list)."""
    h = hashlib.sha256()
    for s in items:
        h.update(s.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()
