"""Shared palette, fonts, path resolution and model loading for every figure.

Every ``figN_*.py`` and ``measure_*.py`` script imports this. It carries:

  * the paper's visual grammar (verbatim from the paper's ``plot_common.py``), so
    the regenerated figures are pixel-for-pixel the published ones;
  * ``add_paths_args`` / ``resolve_paths`` — the ``--models-dir`` / ``--data-dir``
    / ``--results-dir`` / ``--out`` flags (env: ``MACARONI_MODELS_DIR``,
    ``MACARONI_DATA_DIR``, ``MACARONI_RESULTS_DIR``, ``MACARONI_OUT``);
  * ``model_dir`` / ``final_ckpt`` / ``load_model`` — the model-location
    convention ``models/<condition>[-sNN]/`` (seed 42 = bare name), whose usable
    weights are the final HF ``checkpoint-*`` (single-run) or the final
    checkpoint of ``stage3/`` (curriculum), or the dir itself if it already holds
    ``config.json`` (a downloaded Hub branch).

Visual grammar (one colour channel per variable):
  LANG      language identity            = blue / orange / red   (English / Dutch / Chinese)
  CORPUS_C  corpus type                  = teal / violet / grey  (word-level CS / sentence-level CS / non-CS)
  CONTROL_C the two other corpus controls = brown (document translation) / pink (word salad)
  EXPOSURE_C/EXPOSURE_MK  seen-embedded vs never-embedded words = green diamond / magenta triangle
  The code-switched vs non-CS MODEL contrast is value + marker, never hue or
  dash: CS = INK filled square, non-CS = SECONDARY grey open circle; every line
  is solid. Seeds collapse to a mean line + shaded +-1 SD band; chance lines are
  dotted grey.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # re-exported

# Fonts: STIX (a Times clone shipped as .ttf) so figures read as part of the
# Times body text, and TrueType (Type 42) embedding so the PDFs pass
# aclpubcheck / text extraction. Importing this module applies them; do NOT
# restate the block in a figure script.
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

INK, MUT, GRID = "#161b21", "#556069", "#d8dee5"
SECONDARY = "#7d8a97"  # the non-CS model's grey (INK is the CS model)
DIFF = "#4a3aa7"

LANG = {"eng": "#2a6fd6", "nld": "#e8850c", "zho": "#d1342f"}
LANG_NAME = {"eng": "English", "nld": "Dutch", "zho": "Chinese"}
LANG_ORDER = ["eng", "nld", "zho"]
PAIRS = ["eng-nld", "eng-zho", "nld-zho"]

CORPUS_C = {"word": "#0f6d64", "sent": "#5b3fa0", "uni": "#9aa5ad"}
CORPUS_NAME = {"word": "word-level CS", "sent": "sentence-level CS", "uni": "non-CS"}
CONTROL_C = {"par": "#8c5a2b", "salad": "#c34a9d"}
EXPOSURE_C = {"seen": "#1a9e5f", "never": "#c34a9d"}
EXPOSURE_MK = {"seen": "D", "never": "^"}

# Within a language-faceted panel: primary = CS model, secondary = non-CS model.
FACET = {"primary":   dict(c=INK, ls="-", mk="s", lw=1.8, ms=3.8, fill=True),
         "secondary": dict(c=SECONDARY, ls="-", mk="o", lw=1.6, ms=3.6, fill=False)}

SAVE_KW = dict(dpi=360, bbox_inches="tight")

# The eight training conditions (== corpus subsets == model names), and which
# are code-switched. ``switch``/``curriculum`` are the same data in two orders.
CONDITIONS = ["switch", "noswitch", "word", "sent", "par", "salad",
              "curriculum", "curriculum_noswitch"]
PAPER_SEEDS = list(range(42, 50))       # the paper's eight seeds
PUBLISHED_SEEDS = [42, 43, 44]          # seeds with final checkpoints on the Hub


def condition_label(condition: str, seed: int) -> str:
    """seed 42 -> bare condition name; other seeds -> '<condition>-sNN'."""
    return condition if int(seed) == 42 else f"{condition}-s{int(seed)}"


def is_noswitch(label: str) -> bool:
    """True for the non-code-switched conditions (``noswitch`` is not a substring of ``switch``)."""
    return "noswitch" in label


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def pair_title(ax, pair, y=1.03, fontsize=10, gap=0.013):
    """Panel title for a PAIR-faceted figure: each language in its own LANG hue."""
    a, b = sorted(pair.split("-"), key=LANG_ORDER.index)
    kw = dict(transform=ax.transAxes, va="bottom", fontsize=fontsize, clip_on=False)
    ax.text(0.5, y, "–", ha="center", color=INK, **kw)
    ax.text(0.5 - gap, y, LANG_NAME[a], ha="right", color=LANG[a], **kw)
    ax.text(0.5 + gap, y, LANG_NAME[b], ha="left", color=LANG[b], **kw)


def lang_title(ax, lang, y=1.03, fontsize=10):
    """Panel title for a language-faceted figure: the language name in its LANG hue."""
    ax.text(0.5, y, LANG_NAME[lang], transform=ax.transAxes, ha="center",
            va="bottom", fontsize=fontsize, color=LANG[lang], clip_on=False)


def stage_boundaries(ax, spans):
    """Neutral hairlines at the curriculum stage boundaries."""
    for st in list(spans)[1:]:
        ax.axvline(spans[st][0] - 0.5, color=GRID, lw=0.8, zorder=0)


def style_axes(ax):
    """Drop the top/right spines and add a light horizontal grid."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)


def save(fig, path):
    """Save ``path`` (a .pdf, the vector file the paper includes) plus a .png sibling."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), **SAVE_KW)
    plt.close(fig)
    print(f"wrote {p.with_suffix('.pdf')}")


# --------------------------------------------------------------------------- #
# Portable path resolution
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]


def add_paths_args(parser, default_out="out/figures"):
    """Add --models-dir / --data-dir / --results-dir / --out (env-overridable)."""
    parser.add_argument("--models-dir", default=os.environ.get("MACARONI_MODELS_DIR", "models"),
                        help="root holding models/<condition>[-sNN]/ (default: ./models)")
    parser.add_argument("--data-dir", default=os.environ.get("MACARONI_DATA_DIR", "data"),
                        help="root holding the downloaded corpora (default: ./data)")
    parser.add_argument("--results-dir",
                        default=os.environ.get("MACARONI_RESULTS_DIR", str(REPO_ROOT / "results")),
                        help="directory of measurement CSVs the figures read "
                             "(default: the paper's CSVs shipped in results/)")
    parser.add_argument("--out", default=os.environ.get("MACARONI_OUT", default_out),
                        help=f"output directory (default: ./{default_out})")
    return parser


def resolve_paths(args):
    """Return (models_dir, data_dir, results_dir, out_dir) as absolute Paths; mkdir out."""
    models_dir = Path(args.models_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, data_dir, results_dir, out_dir


# --------------------------------------------------------------------------- #
# Model location: models/<condition>[-sNN]/
# --------------------------------------------------------------------------- #
def model_dir(condition: str, seed: int, models_dir) -> Path:
    return Path(models_dir) / condition_label(condition, seed)


def _load_final_checkpoint():
    spec = importlib.util.spec_from_file_location(
        "final_checkpoint", REPO_ROOT / "final_checkpoint.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.final_checkpoint


final_checkpoint = _load_final_checkpoint()


def final_ckpt(path) -> Path:
    """Resolve a model dir to its usable weights (see module docstring)."""
    path = Path(path)
    if (path / "config.json").is_file():
        return path
    ck = final_checkpoint(path)
    if ck is not None:
        return ck
    stages = sorted(path.glob("stage*"), key=lambda p: p.name)
    for stage in reversed(stages):
        if (stage / "config.json").is_file():
            return stage
        ck = final_checkpoint(stage)
        if ck is not None:
            return ck
    raise FileNotFoundError(
        f"no config.json, checkpoint-*, or stage*/ weights under {path}. "
        f"Download it (python download_models.py --only <condition>) or train it "
        f"(python train.py --corpus <condition>).")


def load_model(path, device="cuda", dtype=None):
    from transformers import AutoModelForCausalLM
    kwargs = {"torch_dtype": dtype} if dtype is not None else {}
    model = AutoModelForCausalLM.from_pretrained(str(final_ckpt(path)),
                                                 output_hidden_states=True, **kwargs)
    return model.to(device).eval()


def load_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(final_ckpt(path)))
