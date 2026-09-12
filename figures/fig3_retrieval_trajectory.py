#!/usr/bin/env python3
"""Figure 3: sentence-level alignment across the training curriculum.

FLORES+ bitext retrieval P@1 (layers 6-11 mean) at ten checkpoints (init, then
epochs 1/5/10 of each of the three stages) for the CS curriculum model and its
non-CS twin, one panel per language pair, 8-seed mean +-1 SD. The CS line is
coloured by the corpus of its current stage (teal word-level CS, violet
sentence-level CS, grey non-CS); the non-CS line is grey throughout.

    python figures/fig3_retrieval_trajectory.py   # -> out/figures/fig3_retrieval_trajectory.pdf

Input: <results-dir>/retrieval_trajectory.csv (ships in results/; regenerate
with figures/measure_retrieval.py --set trajectory, which needs the per-stage
checkpoints written by train.py for both curriculum arms).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from common import (plt, INK, MUT, SECONDARY, CORPUS_C, CORPUS_NAME, PAIRS,
                    style_axes, save, pair_title, stage_boundaries)

UPPER = list(range(6, 12))
CKPT = [("s1", 0), ("s1", 1), ("s1", 5), ("s1", 10),
        ("s2", 1), ("s2", 5), ("s2", 10),
        ("s3", 1), ("s3", 5), ("s3", 10)]
XLAB = ["init", "1", "5", "10", "1", "5", "10", "1", "5", "10"]
STAGES = ["s1", "s2", "s3"]
STAGE_SPAN = {"s1": (0, 3), "s2": (4, 6), "s3": (7, 9)}
STAGE_NUM = {"s1": "stage 1", "s2": "stage 2", "s3": "stage 3"}
STAGE_C = {"s1": CORPUS_C["word"], "s2": CORPUS_C["sent"], "s3": SECONDARY}
STAGE_LABEL = {"s1": CORPUS_NAME["word"], "s2": CORPUS_NAME["sent"], "s3": CORPUS_NAME["uni"]}
ARM = {"cs": dict(c=INK, mk="s", lw=2.0, ms=5.2, fill=True, name="code-switched"),
       "noswitch": dict(c=SECONDARY, mk="o", lw=1.8, ms=4.8, fill=False, name="non-CS")}
XMIN, XMAX = -0.5, len(CKPT) - 0.5
METRICS = {
    "retrieval_p1": dict(scale=100.0, chance=100.0 / 997, lim=(-4, 100),
                         label="bitext retrieval P@1 (%)"),
    "median_pct_rank": dict(scale=1.0, chance=0.5, lim=(0.45, 1.02),
                            label="median percentile rank of the\ncorrect translation"),
}


def parse_label(lab):
    m = re.match(r"cur_(cs|noswitch)-s(\d+)-(s\d)e(\d+)$", lab)
    return m.group(1), int(m.group(2)), m.group(3), int(m.group(4))


def load(csv, metric):
    r = pd.read_csv(csv)
    r = r[r.layer.isin(UPPER)]
    r[["group", "seed", "stage", "epoch"]] = r["model"].apply(lambda l: pd.Series(parse_label(l)))
    up = r.groupby(["group", "seed", "pair", "stage", "epoch"])[metric].mean().reset_index()
    agg = up.groupby(["group", "pair", "stage", "epoch"])[metric].agg(["mean", "std"]).reset_index()
    agg[["mean", "std"]] *= METRICS[metric]["scale"]
    return agg


def stage_runs():
    runs = []
    for i, st in enumerate(STAGES):
        a, b = STAGE_SPAN[st]
        idx = list(range(a, b + 1))
        if i + 1 < len(STAGES):
            idx.append(STAGE_SPAN[STAGES[i + 1]][0])
        runs.append((st, idx))
    return runs


def draw(ax, x, agg, pair):
    for arm in ("noswitch", "cs"):
        t = (agg[(agg.group == arm) & (agg.pair == pair)]
             .set_index(["stage", "epoch"]).reindex(CKPT))
        m, sd = t["mean"].values, t["std"].values
        a = ARM[arm]
        if arm == "noswitch":
            ax.fill_between(x, m - sd, m + sd, color=a["c"], alpha=0.16, lw=0, zorder=2)
            ax.plot(x, m, ls="-", color=a["c"], lw=a["lw"], zorder=3, solid_capstyle="round")
            ax.plot(x, m, ls="none", marker=a["mk"], ms=a["ms"], zorder=4,
                    mfc="white", mec=a["c"], mew=1.2)
        else:
            for st, idx in stage_runs():
                c = STAGE_C[st]
                ax.fill_between(x[idx], m[idx] - sd[idx], m[idx] + sd[idx],
                                color=c, alpha=0.16, lw=0, zorder=2)
                ax.plot(x[idx], m[idx], ls="-", color=c, lw=a["lw"], zorder=3,
                        solid_capstyle="round")
            for xi, mi, (st, _) in zip(x, m, CKPT):
                ax.plot([xi], [mi], ls="none", marker=a["mk"], ms=a["ms"], zorder=4,
                        mfc=STAGE_C[st], mec=STAGE_C[st], mew=1.2)


def build(agg, metric, out):
    x = np.arange(len(CKPT))
    M = METRICS[metric]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6), sharey=True)
    fig.subplots_adjust(top=0.80, wspace=0.09, bottom=0.24)
    for k, (ax, pair) in enumerate(zip(axes, PAIRS)):
        stage_boundaries(ax, STAGE_SPAN)
        draw(ax, x, agg, pair)
        if M["chance"] is not None:
            ax.axhline(M["chance"], color=MUT, lw=1.0, ls=":", zorder=1)
        pair_title(ax, pair, y=1.10, fontsize=10.5)
        for st in STAGES:
            a, b = STAGE_SPAN[st]
            ax.text((a + b) / 2, 1.012, STAGE_NUM[st], transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6.6, color=MUT)
        ax.set_xticks(x)
        ax.set_xticklabels(XLAB, fontsize=8)
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(*M["lim"])
        ax.set_xlabel("epoch within stage", color=MUT, fontsize=9)
        style_axes(ax)
        if k == 0:
            ax.set_ylabel(f"{M['label']}\n(FLORES+, layers 6-11 mean)", color=INK, fontsize=9)
    arm_h = [Line2D([], [], color=ARM[g]["c"], lw=ARM[g]["lw"], ls="none",
                    marker=ARM[g]["mk"], ms=ARM[g]["ms"],
                    mfc=ARM[g]["c"] if ARM[g]["fill"] else "white",
                    mec=ARM[g]["c"], label=ARM[g]["name"]) for g in ("cs", "noswitch")]
    col_h = [Line2D([], [], color=STAGE_C[st], lw=1.8, ls="-", label=STAGE_LABEL[st]) for st in STAGES]
    l1 = fig.legend(handles=arm_h, frameon=False, fontsize=8.5, ncol=2, loc="lower center",
                    bbox_to_anchor=(0.28, -0.055), handlelength=1.4, columnspacing=1.8,
                    title="model", title_fontsize=8)
    l2 = fig.legend(handles=col_h, frameon=False, fontsize=8.5, ncol=3, loc="lower center",
                    bbox_to_anchor=(0.70, -0.055), handlelength=3.0, columnspacing=1.6,
                    title="color = corpus of the CS model's current stage", title_fontsize=8)
    for lg in (l1, l2):
        lg.get_title().set_color(MUT)
    save(fig, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_paths_args(ap)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--metric", choices=list(METRICS), default="retrieval_p1")
    args = ap.parse_args()
    _m, _d, results_dir, out_dir = common.resolve_paths(args)
    csv_path = Path(args.csv) if args.csv else results_dir / "retrieval_trajectory.csv"
    build(load(csv_path, args.metric), args.metric, out_dir / "fig3_retrieval_trajectory.pdf")


if __name__ == "__main__":
    main()
