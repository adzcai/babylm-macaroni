"""Shared plotting palette, path resolution, and model loading for every figure.

This is the single import every ``figNN_*.py`` script depends on. It provides:

  * the CVD-validated colour system (one colour channel per variable) — kept
    identical to the paper so all figures read as one visual system;
  * ``style_axes`` / ``save`` house helpers and the ``plt`` re-export;
  * ``add_paths_args`` / ``resolve_paths`` — the portable replacement for the
    old hardcoded ``/n/.../drooryck/lmkiddo`` roots; every script resolves
    ``models_dir`` / ``data_dir`` / ``out`` from CLI flags or environment, with
    sensible ``./models``, ``./data``, ``./out`` defaults;
  * ``model_dir`` / ``final_ckpt`` / ``load_model`` — the "right place"
    convention: a trained-or-downloaded model lives at
    ``models/<condition>[-sNN]/`` and its usable weights are its final HF
    ``checkpoint-*`` (via the repo-root ``final_checkpoint.py``), or the dir
    itself if it already holds ``config.json``.

Visual grammar (each figure uses ONE colour family):
  LANG     - language identity (eng/nld/zho) = purple / orange / brown.
  PAIR     - language-PAIR identity          = blue / green / magenta.
  TIER     - word-pair tier, ORDERED (switched > novel) = dark / light teal.
  DIFF     - a derived difference / DiD quantity = violet.
  MODEL_LS - code-switched vs unilingual is LINE STYLE: CS solid, unilingual dashed.
Seeds are collapsed to a mean line + shaded +-1 SD band; chance/reference lines
are dotted grey (MUT, ls=":").
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # re-exported

# --------------------------------------------------------------------------- #
# Palette (verbatim from the paper's plot_common.py; all families CVD-safe).
# --------------------------------------------------------------------------- #
INK, MUT, GRID = "#161b21", "#556069", "#d8dee5"
DIFF = "#4a3aa7"  # violet: difference / DiD quantities

LANG = {"eng": "#7B5EA7", "nld": "#D95F02", "zho": "#8C512C"}              # purple / orange / brown
PAIR = {"eng-nld": "#2a78d6", "eng-zho": "#1baf7a", "nld-zho": "#c34a9d"}  # blue / green / magenta
TIER = {"attested": "#0f6d64", "pair_unseen": "#8fd0c8"}                   # dark / light teal (switched / novel)
# Model contrast -> line style (CS solid, unilingual dashed). Bare + run-prefixed keys.
MODEL_LS = {"cs": "-", "nsw": "--",
            "cur_cs": "-", "cur_nsw": "--", "rnd_cs": "-", "rnd_nsw": "--"}

# Legacy aliases retained for un-migrated plot code.
CS, NSW = "#2a78d6", "#e0662f"
BLUE, GREEN, RED = "#2a78d6", "#1baf7a", "#e34948"

LAYER_X = "layer  (0 = embedding … 12 = top)"
SAVE_KW = dict(dpi=150, bbox_inches="tight")

# The three language pairs, in canonical order (matches PAIR keys).
PAIRS = [("eng", "nld"), ("eng", "zho"), ("nld", "zho")]


def style_axes(ax):
    """Drop the top/right spines and add a light horizontal grid."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)


def save(fig, path):
    """Save with the house defaults and close the figure."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Portable path resolution (replaces every hardcoded /n/... root).
# --------------------------------------------------------------------------- #
def add_paths_args(parser, default_out="out/figures"):
    """Add the standard --models-dir / --data-dir / --out flags to a parser.

    Defaults come from env (MACARONI_MODELS_DIR / _DATA_DIR / _OUT) then fall
    back to ./models, ./data, and ``default_out``. Figure scripts write their
    ``figNN_*.{csv,png}`` directly into ``--out`` (default ``out/figures``);
    non-figure consumers (e.g. evaluate.py) pass ``default_out="out"`` and add
    their own subdir.
    """
    parser.add_argument("--models-dir", default=os.environ.get("MACARONI_MODELS_DIR", "models"),
                        help="root holding models/<condition>[-sNN]/ (default: ./models)")
    parser.add_argument("--data-dir", default=os.environ.get("MACARONI_DATA_DIR", "data"),
                        help="root holding downloaded corpora (default: ./data)")
    parser.add_argument("--out", default=os.environ.get("MACARONI_OUT", default_out),
                        help=f"output directory (default: ./{default_out})")
    return parser


def resolve_paths(args):
    """Return (models_dir, data_dir, out_dir) as absolute Paths; make out_dir."""
    models_dir = Path(args.models_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, data_dir, out_dir


# --------------------------------------------------------------------------- #
# Model-location convention: models/<condition>[-sNN]/  (seed 42 => no suffix).
# --------------------------------------------------------------------------- #
def model_dir(condition: str, seed: int, models_dir) -> Path:
    """Path to a model's directory by condition + seed (42 => bare name)."""
    suffix = "" if int(seed) == 42 else f"-s{int(seed)}"
    return Path(models_dir) / f"{condition}{suffix}"


# Load final_checkpoint() from the repo-root module without requiring a package.
def _load_final_checkpoint():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("final_checkpoint", root / "final_checkpoint.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.final_checkpoint


final_checkpoint = _load_final_checkpoint()


def final_ckpt(path) -> Path:
    """Resolve a model path to its usable weights dir.

    Handles the three on-disk shapes a model can take here:
      1. a plain HF model dir (has ``config.json``) — e.g. a downloaded branch;
      2. a single-run training output (holds ``checkpoint-*``);
      3. a locally-trained curriculum model (``stage1/ stage2/ stage3/``) — the
         weights are the final checkpoint of the highest stage.
    """
    path = Path(path)
    if (path / "config.json").is_file():
        return path
    ck = final_checkpoint(path)
    if ck is not None:
        return ck
    # Curriculum layout: descend into the highest-numbered stage dir.
    stages = sorted(path.glob("stage*"), key=lambda p: p.name)
    for stage in reversed(stages):
        if (stage / "config.json").is_file():
            return stage
        ck = final_checkpoint(stage)
        if ck is not None:
            return ck
    raise FileNotFoundError(f"no config.json, checkpoint-*, or stage*/ weights under {path}")


def _model_src(path) -> str:
    """Resolve a model path/id to a ``from_pretrained`` source, with clear errors.

    A local model (existing dir, or an absolute / ``./``-style path that simply
    hasn't been downloaded yet) goes through ``final_ckpt`` — which raises a
    friendly ``FileNotFoundError`` pointing at ``download_models.py`` if it is
    missing, instead of letting HuggingFace misread the path as a repo id. A bare
    ``namespace/name`` string is treated as an HF id and passed through.
    """
    p, s = Path(path), str(path)
    if p.exists() or p.is_absolute() or s.startswith((".", os.sep)):
        try:
            return str(final_ckpt(p))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{e}. If this is one of our models, download it first: "
                f"python download_models.py --only <condition>") from None
    return s  # bare HF id


def load_model(path, device="cuda", dtype=None):
    """Load an ``AutoModelForCausalLM`` from a model dir / HF id (lazy import)."""
    from transformers import AutoModelForCausalLM
    kwargs = {}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(_model_src(path), output_hidden_states=True, **kwargs)
    model.to(device).eval()
    return model


def load_tokenizer(path):
    """Load an ``AutoTokenizer`` from a model dir / HF id (lazy import)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(_model_src(path))
