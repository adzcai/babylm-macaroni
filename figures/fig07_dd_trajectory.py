#!/usr/bin/env python3
"""Fig 7 — embedded-word cross-lingual alignment trajectory (difference-in-differences).

Code-switching's effect on embedded words' cross-lingual alignment over training,
on the probability-of-superiority (``pct``) metric, upper layers. Three series:
  * Delta_E   — same word, cs minus unilingual arm (primary; needs no controls)
  * DD        — matched-control difference-in-differences (embedded-specific)
  * Delta_noswitch — unilingual arm's embedded-minus-control (selection artifact)
Shown as the 3-seed mean +- 1 SD, in percentage points.

ONE invocation runs COMPUTE then PLOT:
    python figures/fig07_dd_trajectory.py --seeds 42 43 44 \
        --models-dir models --data-dir data --out out/figures
  -> out/figures/fig07_dd_trajectory.csv   (per-seed stats_trajectory rows)
  -> out/figures/fig07_dd_trajectory.png
Use --skip-compute to replot from an existing CSV.

============================ REQUIRES TRAJECTORY CHECKPOINTS ==================
COMPUTE needs the model at 10 points along training for BOTH arms (curriculum and
curriculum_noswitch). The Hub ships only FINAL checkpoints, and the no-CS arm's
intermediate checkpoints are NOT published, so COMPUTE raises a friendly
``CheckpointsMissing`` error unless those intermediates are supplied on disk (see
figures/_lib/traj/__init__.py). --skip-compute needs no models or GPU.
=============================================================================
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import common
from common import plt, DIFF, GREEN, NSW, MUT, INK, style_axes, save
from _lib import traj

CKPT = traj.CKPT_ORDER
XLAB = ["init", "1", "5", "10", "1", "5", "10", "1", "5", "10"]
STAGE_SPAN = {"stage1": (0, 3), "stage2": (4, 6), "stage3": (7, 9)}
STAGE_NAME = {"stage1": "stage 1 · word-level CS",
              "stage2": "stage 2 · sentence-level CS",
              "stage3": "stage 3 · unilingual"}


def _stage_bands(ax):
    for st, (a, b) in STAGE_SPAN.items():
        if st != "stage3":
            ax.axvspan(a - 0.5, b + 0.5,
                       color="#dce9f7" if st == "stage1" else "#e8f3e3",
                       alpha=0.6, lw=0)
        ax.text((a + b) / 2, 1.015, STAGE_NAME[st], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color=MUT)


def plot(csv_path, png_path):
    """Reproduce the dd_plot_v2 figure from the pooled per-seed stats CSV."""
    d = pd.read_csv(csv_path)
    d = d[d.metric == "pct"]
    x = list(range(len(CKPT)))

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.axhline(0, color=MUT, lw=1, ls="--", zorder=1)
    series = (("Delta_E", DIFF, "-", 2.8,
               r"$\Delta_E$: same word, cs $-$ unilingual (primary)"),
              ("DD", GREEN, "-", 2.0,
               r"DiD: matched-control (embedded-specific)"),
              ("Delta_noswitch", "#8895a3", "--", 1.6,
               r"unilingual arm: embedded $-$ control (selection artifact)"))
    for est, col, ls, lw, lab in series:
        piv = (d[d.estimand == est]
               .pivot_table(index=["stage", "epoch"], columns="seed", values="mean")
               .reindex(CKPT)) * 100
        mean = piv.mean(axis=1).values
        sd = piv.std(axis=1, ddof=0).values
        ax.plot(x, mean, ls, marker="o", color=col, lw=lw, ms=5, label=lab)
        ax.fill_between(x, mean - sd, mean + sd, color=col, alpha=0.13, lw=0)
    _stage_bands(ax)
    ax.set_xticks(x); ax.set_xticklabels(XLAB, fontsize=9)
    ax.set_xlabel("epoch within stage", color=MUT)
    ax.set_ylabel("cross-arm alignment difference\n"
                  "(probability-of-superiority, percentage points)", color=INK)
    ax.set_title("Embedded-word alignment builds during CS, persists through stage 3 "
                 "(3 seeds, mean $\\pm$ 1 SD)", fontsize=10.5, pad=24)
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    ax.set_xlim(-0.5, len(CKPT) - 0.5)
    style_axes(ax)
    save(fig, png_path)
    print(f"wrote {png_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_paths_args(ap)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                    help="model seeds to run and pool (default: 42 43 44)")
    ap.add_argument("--skip-compute", action="store_true",
                    help="replot from an existing fig07_dd_trajectory.csv (no models/GPU)")
    args = ap.parse_args()
    models_dir, data_dir, out_dir = common.resolve_paths(args)

    csv_path = out_dir / "fig07_dd_trajectory.csv"
    png_path = out_dir / "fig07_dd_trajectory.png"

    if not args.skip_compute:
        # Per-seed intermediates live under out/figures/traj_s<seed>/.
        df = traj.dd_trajectory(models_dir, data_dir, out_dir,
                                seeds=args.seeds)
        df.to_csv(csv_path, index=False)
        print(f"wrote {csv_path}  ({len(df)} rows, seeds {args.seeds})")
    elif not csv_path.is_file():
        raise SystemExit(f"--skip-compute set but {csv_path} does not exist; "
                         f"run once without --skip-compute first.")

    plot(csv_path, png_path)


if __name__ == "__main__":
    main()
