#!/usr/bin/env python3
"""Figure 4: does the alignment gain reach words never seen embedded?

Final-checkpoint median percentile rank of a word's correct translation among
the target-language words of the matched set, per direction, for seen-embedded
(green diamond) and never-embedded (magenta triangle) source words; CS model
filled, non-CS model open; the connecting segment is the CS gain. 8-seed mean
+-1 SD.

    python figures/fig4_exposure_generalization.py   # -> out/figures/fig4_exposure_generalization.pdf

Input: <results-dir>/exposure_rank.csv (ships in results/; regenerate with
figures/measure_wordlevel.py).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from common import plt, LANG, MUT, INK, EXPOSURE_C, EXPOSURE_MK, style_axes, save

GROUPS = [("seen_embedded", EXPOSURE_C["seen"], EXPOSURE_MK["seen"], "seen embedded"),
          ("never_embedded", EXPOSURE_C["never"], EXPOSURE_MK["never"], "never embedded")]


def direction_labels(ax, dirs, n_by, y=-0.04, gap=0.11, fontsize=7.5):
    ax.set_xticklabels([])
    kw = dict(transform=ax.get_xaxis_transform(), va="top", fontsize=fontsize, clip_on=False)
    for i, dd in enumerate(dirs):
        src, tgt = dd.split("-")
        ax.text(i, y, "$\\to$", ha="center", color=INK, **kw)
        ax.text(i - gap, y, src, ha="right", color=LANG[src], **kw)
        ax.text(i + gap, y, tgt, ha="left", color=LANG[tgt], **kw)
        ax.text(i, y - 0.10, f"(n={n_by[dd]:,})", ha="center", va="top", color=MUT,
                fontsize=6.5, transform=ax.get_xaxis_transform(), clip_on=False)


def exposure_panel(ax, d):
    f = d[(d.stage == "stage3") & (d.epoch == 10) & (d.direction != "pooled")]
    agg = f.groupby(["direction", "group"])[["median_cs", "median_noswitch"]].agg(["mean", "std"])
    dlt = f.groupby(["direction", "group"])["delta"].mean()
    n_by = f.groupby("direction")["n"].first()
    dirs = dlt.xs("seen_embedded", level="group").sort_values().index.tolist()
    ax.axhline(0.5, color=MUT, lw=1, ls=":", zorder=1)
    ax.text(-0.45, 0.5, "chance", ha="left", va="bottom", fontsize=6.5, color=MUT)
    off = 0.13
    for j, (g, col, mk, _lab) in enumerate(GROUPS):
        for i, dd in enumerate(dirs):
            x = i + (j - 0.5) * 2 * off
            cs, ns = agg.loc[(dd, g), "median_cs"], agg.loc[(dd, g), "median_noswitch"]
            ax.plot([x, x], [ns["mean"], cs["mean"]], color=col, lw=1.4, zorder=2)
            ax.errorbar(x, ns["mean"], yerr=ns["std"], fmt=mk, ms=5, mfc="white", mec=col,
                        mew=1.4, ecolor=col, elinewidth=1.2, capsize=2.5, zorder=3)
            ax.errorbar(x, cs["mean"], yerr=cs["std"], fmt=mk, ms=5, color=col,
                        elinewidth=1.2, capsize=2.5, zorder=4)
    ax.set_ylim(0.475, 1.03)
    ax.set_xticks(list(range(len(dirs))))
    direction_labels(ax, dirs, n_by)
    ax.set_ylabel("median percentile rank\nof translation", color=INK, fontsize=8.5)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_xlim(-0.5, len(dirs) - 0.5)
    style_axes(ax)


def build(d, out):
    d = d[d.subset == "all"]
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    exposure_panel(ax, d)
    handles = [Line2D([], [], color=col, marker=mk, ms=5, lw=0, label=lab)
               for _, col, mk, lab in GROUPS]
    handles += [Line2D([], [], color=INK, marker="o", ms=5, lw=0, label="CS model"),
                Line2D([], [], color=INK, marker="o", ms=5, lw=0, mfc="white", mew=1.4,
                       label="non-CS model")]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.32), columnspacing=1.0, handletextpad=0.4,
              labelspacing=0.3, borderaxespad=0)
    fig.tight_layout()
    save(fig, out)


def report(d):
    """The retention numbers quoted in the paper's text."""
    for subset in ("all", "balanced"):
        s = d[(d.subset == subset) & (d.direction == "pooled") & (d.stage == "stage3") & (d.epoch == 10)]
        if s.empty:
            continue
        g = s.groupby("group")["delta"].agg(["mean", "std"])
        e, c = g.loc["seen_embedded", "mean"], g.loc["never_embedded", "mean"]
        print(f"  {subset:9s} delta: seen {e:+.4f}+-{g.loc['seen_embedded', 'std']:.4f}  "
              f"never {c:+.4f}+-{g.loc['never_embedded', 'std']:.4f}  retention {c / e:.1%}  "
              f"n={s.n.iloc[0]:,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_paths_args(ap)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    _m, _d, results_dir, out_dir = common.resolve_paths(args)
    d = pd.read_csv(Path(args.csv) if args.csv else results_dir / "exposure_rank.csv")
    build(d, out_dir / "fig4_exposure_generalization.pdf")
    report(d)


if __name__ == "__main__":
    main()
