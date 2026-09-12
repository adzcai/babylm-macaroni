#!/usr/bin/env python3
"""Measure FLORES+ bitext retrieval per layer: models -> results CSVs (Figs 2, 3, 6, 7).

    python figures/measure_retrieval.py --set finals      # curriculum, curriculum_noswitch, switch, noswitch
    python figures/measure_retrieval.py --set controls    # word, sent, par, salad
    python figures/measure_retrieval.py --set trajectory  # 10 checkpoints x 2 curriculum arms
    python figures/measure_retrieval.py --set all --seeds 42 43 44

Writes (into --results-dir, default the repo's results/, OVERWRITING the paper's
CSV of the same name; pass --results-dir out/results to keep them apart):

    retrieval_finals.csv       read by fig2, fig6, fig7
    retrieval_controls.csv     read by fig7
    retrieval_trajectory.csv   read by fig3

Models are read from --models-dir as models/<condition>[-sNN]/ (downloaded or
trained). The trajectory set needs the per-stage checkpoints written by
train.py (checkpoint-0 + epochs 1/5/10 of stage1-3 for both curriculum arms),
which are NOT on the Hub. Conditions or seeds that are missing on disk are
skipped with a warning, so partial suites still produce a CSV.

Needs a GPU and access to the gated FLORES+ dataset (HF_TOKEN). --emb-cache DIR
caches pooled hidden states per model so re-scoring is CPU-only.
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from _lib import represent_align as ra
from _lib.traj import config as tcfg

SETS = {
    "finals": ["curriculum", "curriculum_noswitch", "switch", "noswitch"],
    "controls": ["word", "sent", "par", "salad"],
}
OUT = {"finals": "retrieval_finals.csv", "controls": "retrieval_controls.csv",
       "trajectory": "retrieval_trajectory.csv"}


def final_labels(conditions, seeds, models_dir):
    labels = {}
    for cond in conditions:
        for seed in seeds:
            md = common.model_dir(cond, seed, models_dir)
            try:
                labels[common.condition_label(cond, seed)] = common.final_ckpt(md)
            except FileNotFoundError:
                print(f"  ! skipping {cond} seed {seed}: no weights under {md}")
    return labels


def trajectory_labels(seeds, models_dir):
    """cur_{cs|noswitch}-s<seed>-s<stage>e<epoch> -> checkpoint dir (paper label format)."""
    labels = {}
    for seed in seeds:
        cfg = tcfg.build_config(models_dir, "data", tempfile.mkdtemp(prefix="traj_"), seed)
        try:
            found = tcfg.discover_checkpoints(cfg)
        except tcfg.CheckpointsMissing as e:
            print(f"  ! skipping trajectory seed {seed}:\n{e}")
            continue
        for arm, stage, epoch, ck in found:
            labels[f"cur_{arm}-s{seed}-s{stage[-1]}e{epoch}"] = ck
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_paths_args(ap)
    ap.add_argument("--set", nargs="+", choices=["finals", "controls", "trajectory", "all"],
                    default=["finals"], help="which checkpoint sets to score")
    ap.add_argument("--seeds", nargs="+", type=int, default=common.PUBLISHED_SEEDS,
                    help="seeds to score (paper: 42-49; Hub: 42 43 44)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--csls-k", type=int, default=ra.CSLS_K)
    ap.add_argument("--emb-cache", default=None, help="cache dir for pooled hidden states")
    args = ap.parse_args()
    models_dir, _data_dir, results_dir, _out = common.resolve_paths(args)
    results_dir.mkdir(parents=True, exist_ok=True)

    sets = ["finals", "controls", "trajectory"] if "all" in args.set else args.set
    flores = ra.load_flores_plus()
    for name in sets:
        if name == "trajectory":
            labels = trajectory_labels(args.seeds, models_dir)
        else:
            labels = final_labels(SETS[name], args.seeds, models_dir)
        if not labels:
            print(f"[{name}] nothing to score; skipped")
            continue
        print(f"[{name}] scoring {len(labels)} checkpoints")
        df = ra.retrieval_per_layer(labels, device=args.device, flores_cache=flores,
                                    csls_k=args.csls_k, emb_cache=args.emb_cache)
        out = results_dir / OUT[name]
        df.to_csv(out, index=False)
        print(f"[{name}] wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
