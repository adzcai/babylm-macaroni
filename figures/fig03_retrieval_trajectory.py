#!/usr/bin/env python3
"""Fig 3 -- cross-lingual retrieval trajectory across the curriculum.

Same FLORES+ bitext retrieval P@1 measurement as fig02_05, but evaluated at a
sequence of PER-CHECKPOINT milestones along training, for the two curriculum arms:

  * cur_cs        = ``curriculum``            (code-switched)
  * cur_noswitch  = ``curriculum_noswitch``   (unilingual baseline)

x seeds 42/43/44. Milestones (init + epochs {1,5,10} within each of the 3
curriculum stages) are averaged over the upper layers (6-11) and shown as a mean
line +/- 1 SD band per language pair.

One invocation: COMPUTE -> ``out/figures/fig03_retrieval_trajectory.csv`` -> PLOT
-> ``out/figures/fig03_retrieval_trajectory.png``. ``--skip-compute`` replots from
the CSV.

=============================== CHECKPOINT GAP ===============================
The per-checkpoint (mid-training) MILESTONE weights are NOT published on the
HuggingFace Hub -- ``download_models.py`` only fetches each run's FINAL checkpoint.
Reproducing this figure therefore requires the intermediate curriculum
checkpoints, which you must obtain by RE-TRAINING both curriculum arms while
saving milestone checkpoints (init + epochs 1/5/10 of stages 1-3).

Expected on-disk layout (populate it yourself; each milestone is a plain HF model
dir containing ``config.json``):

  models/<condition>[-sNN]/trajectory/<stage>e<epoch>/

e.g. ``models/curriculum/trajectory/s1e0`` (init) ... ``models/curriculum/trajectory/s3e10``
and the seed variants ``models/curriculum-s43/...``, ``models/curriculum_noswitch-s44/...``.

If any expected milestone is missing this script raises a friendly, explicit error
naming what is absent -- it never fabricates or interpolates trajectory data.

FLORES+ probe needs HF auth (gated dataset) -- see figures/_lib/represent_align.py.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

import common
from common import PAIR, INK, MUT, style_axes, save, plt
from _lib import represent_align as ra

UPPER = list(range(6, 12))  # upper-layer window averaged for the trajectory
# (stage, epoch) milestones, in order; epoch 0 of stage 1 is init.
CKPT = [("s1", 0), ("s1", 1), ("s1", 5), ("s1", 10),
        ("s2", 1), ("s2", 5), ("s2", 10),
        ("s3", 1), ("s3", 5), ("s3", 10)]
XLAB = ["init", "1", "5", "10", "1", "5", "10", "1", "5", "10"]
STAGE_SPAN = {"s1": (0, 3), "s2": (4, 6), "s3": (7, 9)}  # x-index ranges
STAGE_NAME = {"s1": "stage 1 - word-level CS", "s2": "stage 2 - sentence-level CS",
              "s3": "stage 3 - unilingual"}
# arm 'cs'/'noswitch' -> local condition directory name.
CONDITION = {"cs": "curriculum", "noswitch": "curriculum_noswitch"}
DEFAULT_SEEDS = [42, 43, 44]


class MissingTrajectoryCheckpoints(SystemExit):
    """Raised (as a clean exit) when milestone checkpoints are not on disk."""


def milestone_dir(arm, seed, stage, epoch, models_dir):
    """models/<condition>[-sNN]/trajectory/<stage>e<epoch>/ for one milestone."""
    suffix = "" if int(seed) == 42 else f"-s{int(seed)}"
    return Path(models_dir) / f"{CONDITION[arm]}{suffix}" / "trajectory" / f"{stage}e{epoch}"


def label_for(arm, seed, stage, epoch):
    """Label parsed by the trajectory plot: cur_<arm>-s<seed>-<stage>e<epoch>."""
    return f"cur_{arm}-s{int(seed)}-{stage}e{epoch}"


def build_model_labels(seeds, models_dir):
    """Map every requested milestone label -> its on-disk path, or error if absent."""
    labels, missing = {}, []
    for arm in ("cs", "noswitch"):
        for seed in seeds:
            for stage, epoch in CKPT:
                d = milestone_dir(arm, seed, stage, epoch, models_dir)
                if not (d / "config.json").is_file():
                    missing.append(str(d))
                else:
                    labels[label_for(arm, seed, stage, epoch)] = str(d)
    if missing:
        n = len(missing)
        shown = "\n  ".join(missing[:12])
        more = f"\n  ... and {n - 12} more" if n > 12 else ""
        raise MissingTrajectoryCheckpoints(
            f"\nfig03 needs {n} mid-training milestone checkpoint(s) that are not on "
            f"disk, e.g.:\n  {shown}{more}\n\n"
            "These intermediate curriculum checkpoints are NOT published on the "
            "HuggingFace Hub (download_models.py fetches only each run's FINAL "
            "checkpoint). To reproduce Fig 3 you must RE-TRAIN the curriculum and "
            "curriculum_noswitch arms while saving milestone checkpoints (init + "
            "epochs 1/5/10 of stages 1-3), then place each as a plain HF model dir "
            "at models/<condition>[-sNN]/trajectory/<stage>e<epoch>/ (see this "
            "script's docstring). No trajectory data is fabricated.")
    return labels


def parse_label(lab):
    m = re.match(r"cur_(cs|noswitch)-s(\d+)-(s\d)e(\d+)$", lab)
    return m.group(1), int(m.group(2)), m.group(3), int(m.group(4))


def stage_bands(ax):
    for st, (a, b) in STAGE_SPAN.items():
        if st != "s3":
            ax.axvspan(a - 0.5, b + 0.5, color="#dce9f7" if st == "s1" else "#e8f3e3",
                       alpha=0.6, lw=0)
        ax.text((a + b) / 2, 0.975, STAGE_NAME[st], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color=MUT)


def plot(df, out_path):
    """Port of persist_plot_v2: upper-layer-mean retrieval trajectory, one panel."""
    r = df[df["layer"].isin(UPPER)].copy()
    r[["group", "seed", "stage", "epoch"]] = r["model"].apply(
        lambda l: pd.Series(parse_label(l)))
    # upper-layer mean of per-layer P@1, then mean +/- SD over the seeds
    up = (r.groupby(["group", "seed", "pair", "stage", "epoch"])["retrieval_p1"]
            .mean().reset_index())
    agg = (up.groupby(["group", "pair", "stage", "epoch"])["retrieval_p1"]
             .agg(["mean", "std"]).reset_index())

    x = list(range(len(CKPT)))
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    for pair, col in PAIR.items():
        for grp, ls, a in (("noswitch", "--", 0.9), ("cs", "-", 1.0)):  # CS solid / uni dashed
            t = (agg[(agg.group == grp) & (agg.pair == pair)]
                 .set_index(["stage", "epoch"]).reindex(CKPT))
            lab = f"{pair} - {'code-switched' if grp == 'cs' else 'unilingual baseline'}"
            ax.plot(x, t["mean"], ls, marker="o", color=col, lw=2.0, ms=4,
                    alpha=a, label=lab)
            ax.fill_between(x, t["mean"] - t["std"], t["mean"] + t["std"],
                            color=col, alpha=0.12, lw=0)
    stage_bands(ax)
    ax.axhline(1 / 997, color=MUT, lw=1, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(XLAB, fontsize=9)
    ax.set_xlabel("epoch within stage", color=MUT)
    ax.set_ylabel("bitext retrieval P@1  (FLORES+, layers 6-11 mean)", color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    ax.set_xlim(-0.5, len(CKPT) - 0.5)
    style_axes(ax)
    save(fig, Path(out_path))
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_paths_args(ap)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--device", default=None, help="torch device (default: auto)")
    ap.add_argument("--skip-compute", action="store_true",
                    help="replot from the existing CSV without loading models")
    args = ap.parse_args()

    models_dir, _data_dir, out_dir = common.resolve_paths(args)
    csv_path = out_dir / "fig03_retrieval_trajectory.csv"

    if args.skip_compute:
        if not csv_path.is_file():
            raise SystemExit(f"--skip-compute but {csv_path} does not exist; run once "
                             "without it to compute the CSV.")
        df = pd.read_csv(csv_path)
    else:
        model_labels = build_model_labels(args.seeds, models_dir)  # errors if absent
        if args.device:
            device = args.device
        else:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        df = ra.retrieval_per_layer(model_labels, device=device)
        df.to_csv(csv_path, index=False)
        print(f"wrote {len(df)} rows to {csv_path}")

    plot(df, out_dir / "fig03_retrieval_trajectory.png")


if __name__ == "__main__":
    main()
