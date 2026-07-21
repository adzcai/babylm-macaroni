#!/usr/bin/env python3
"""Figs 2 & 5 -- per-layer cross-lingual bitext retrieval P@1.

Code-switched vs. matched unilingual baseline, one panel per language pair, mean
+/- 1 SD over seeds, with a dotted 1/N chance line. Two ordering arms:

  * ``curriculum`` arm (Fig 2): curriculum  vs  curriculum_noswitch
  * ``random`` arm     (Fig 5): switch      vs  noswitch

One invocation: COMPUTE (per-(layer,pair) retrieval P@1 over all requested
conditions x seeds via ``_lib.represent_align.retrieval_per_layer``) ->
``out/figures/fig02_05_retrieval_per_layer.csv`` -> PLOT. ``--arm both`` (default)
writes both PDFs; ``--arm curriculum``/``random`` writes only one.

  out/figures/fig02_05_retrieval_per_layer.pdf         (curriculum arm; paper Fig 2)
  out/figures/fig02_05_retrieval_per_layer_random.pdf  (random arm;     paper Fig 5)

FLORES+ probe needs HF auth (gated dataset) -- see figures/_lib/represent_align.py.
Use ``--skip-compute`` to replot from an existing CSV without loading any model.
"""
import argparse
import statistics
from collections import defaultdict

import pandas as pd

import common
from common import PAIR, MUT, plt
from _lib import represent_align as ra

PAIRS = ["eng-nld", "eng-zho", "nld-zho"]
PAIR_LABEL = {"eng-nld": "English-Dutch", "eng-zho": "English-Chinese",
              "nld-zho": "Dutch-Chinese"}
# Model contrast -> line style: +CS solid, unilingual dashed (common.MODEL_LS).
GROUPS = [("unilingual", "--"), ("cs", "-")]
N_PROBE = 997  # chance = 1/N

# Which conditions belong to each arm, and whether each is the +CS side.
ARM_CONDITIONS = {
    "curriculum": ["curriculum", "curriculum_noswitch"],
    "random": ["switch", "noswitch"],
}


def label_for(condition, seed):
    """Model label encoding condition + seed (seed 42 => bare, matches model_dir)."""
    return condition if int(seed) == 42 else f"{condition}-s{int(seed)}"


def cs_side(model):
    """'cs' (code-switched) vs 'unilingual' for a model label -> line style key."""
    return "unilingual" if ra.is_unilingual(model) else "cs"


def compute(conditions, seeds, models_dir, device):
    """Resolve model dirs, run the retrieval measurement, return a DataFrame."""
    model_labels = {}
    for cond in conditions:
        for seed in seeds:
            label = label_for(cond, seed)
            model_labels[label] = str(common.model_dir(cond, seed, models_dir))
    return ra.retrieval_per_layer(model_labels, device=device)


def plot_arm(df, arm, out_path):
    """Port of plot_retrieval_per_layer for ONE arm; colour = pair, style = model."""
    sub = df[df["model"].map(ra.arm_of) == arm]
    layers = sorted(sub["layer"].unique())
    # (side, layer, pair) -> [p1%, ...]
    agg = defaultdict(list)
    for _, r in sub.iterrows():
        agg[(cs_side(r["model"]), int(r["layer"]), r["pair"])].append(float(r["retrieval_p1"]) * 100.0)

    chance = 100.0 / N_PROBE
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0), sharey=True)
    for ax, pair in zip(axes, PAIRS):
        col = PAIR[pair]
        for side, ls in GROUPS:
            vals = [agg.get((side, L, pair), []) for L in layers]
            mean = [statistics.mean(v) if v else float("nan") for v in vals]
            sd = [statistics.pstdev(v) if len(v) > 1 else 0.0 for v in vals]
            lo = [m - s for m, s in zip(mean, sd)]
            hi = [m + s for m, s in zip(mean, sd)]
            ax.fill_between(layers, lo, hi, color=col, alpha=0.14, linewidth=0)
            ax.plot(layers, mean, ls, marker="o", ms=3, lw=1.8, color=col)
        ax.axhline(chance, ls=":", lw=1.3, color=MUT)
        ax.set_ylim(-4, 100)
        ax.set_title(PAIR_LABEL[pair], fontsize=10, color=col)
        ax.set_xlabel("layer", fontsize=9)
        ax.set_xticks([0, 3, 6, 9, 12])
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("bitext retrieval P@1 (%)", fontsize=9)
    from matplotlib.lines import Line2D
    proxies = [Line2D([0], [0], color="#555", ls="--", lw=1.8, label="unilingual baseline"),
               Line2D([0], [0], color="#555", ls="-", lw=1.8, label="+CS"),
               Line2D([0], [0], color=MUT, ls=":", lw=1.3, label="chance (1/N)")]
    axes[0].legend(handles=proxies, fontsize=7.5, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path} (arm={arm})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_paths_args(ap)
    ap.add_argument("--arm", choices=["curriculum", "random", "both"], default="both")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="override the conditions to measure (default: those "
                         "implied by --arm)")
    ap.add_argument("--device", default=None, help="torch device (default: auto)")
    ap.add_argument("--skip-compute", action="store_true",
                    help="replot from the existing CSV without loading models")
    args = ap.parse_args()

    models_dir, _data_dir, out_dir = common.resolve_paths(args)
    csv_path = out_dir / "fig02_05_retrieval_per_layer.csv"

    arms = ["curriculum", "random"] if args.arm == "both" else [args.arm]

    if args.skip_compute:
        if not csv_path.is_file():
            raise SystemExit(f"--skip-compute but {csv_path} does not exist; run once "
                             "without it to compute the CSV.")
        df = pd.read_csv(csv_path)
    else:
        if args.conditions is not None:
            conditions = args.conditions
        else:
            conditions = []
            for a in arms:
                conditions += ARM_CONDITIONS[a]
        if args.device:
            device = args.device
        else:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        df = compute(conditions, args.seeds, models_dir, device)
        df.to_csv(csv_path, index=False)
        print(f"wrote {len(df)} rows to {csv_path}")

    suffix = {"curriculum": "", "random": "_random"}
    for a in arms:
        plot_arm(df, a, out_dir / f"fig02_05_retrieval_per_layer{suffix[a]}.pdf")


if __name__ == "__main__":
    main()
