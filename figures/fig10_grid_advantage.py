#!/usr/bin/env python3
"""Figure 10 -- grid experiment (grid_heatmaps.png + grid_advantage.png).

Full factorial word-language x sentence-language x seen-embedded, measuring the
upper-layer context-language shift per cell, for the curriculum cs vs.
unilingual arms. Single seed (42): the grid is a per-cell factorial contrast,
not a seed-variance figure.

Pipeline (one invocation):
  sample own/synthetic/natural contexts per SENTENCE language  ->  extract
  per-(word,model,layer) reps (curriculum + curriculum_noswitch FINAL
  checkpoints)  ->  per-cell cross/reliab/shift  ->
  out/figures/fig10_grid_advantage.csv  ->  grid_advantage.png + grid_heatmaps.png

Run:  python figures/fig10_grid_advantage.py [--seed 42] [--skip-compute]

CORPUS FLAGS: needs the monolingual BabyLM corpora (--babylm-dir), the Exp-3
matched pairs (--pairs: measurement_pairs_mono_freq.parquet), and phase_intra
= DOWNLOADED data/curriculum/1_intra.parquet (--phase-intra). BabyLM + the
pairs parquet are NOT fetched by download_data.py -- see figures/_lib/grid.py.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import common`
import common
from _lib import grid, contextual

LANGS = ("eng", "nld", "zho")
CONDITIONS = {"cur_cs": "curriculum", "cur_nsw": "curriculum_noswitch"}


def compute(args, models_dir, data_dir, out_dir):
    babylm_dir = Path(args.babylm_dir) if args.babylm_dir else data_dir / "babylm"
    pairs = Path(args.pairs) if args.pairs else data_dir / "traj" / "measurement_pairs_mono_freq.parquet"
    phase_intra = Path(args.phase_intra) if args.phase_intra else data_dir / "curriculum" / "1_intra.parquet"

    seed = args.seed
    model_paths = {lbl: common.final_ckpt(common.model_dir(cond, seed, models_dir))
                   for lbl, cond in CONDITIONS.items()}

    # one pass per SENTENCE language, concatenated (keys carry word-lang + cell)
    ctx_all = out_dir / f"fig10_contexts_grid_s{seed}.jsonl"
    with open(ctx_all, "w") as f:
        for lang in LANGS:
            recs = grid.sample_contexts(lang, pairs, phase_intra, babylm_dir)
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    npz = contextual.extract_reps(
        model_paths, ctx_all, tokenizer_dir=model_paths["cur_cs"],
        device=args.device, batch_size=args.batch_size)
    df = grid.analyze(npz, pairs)
    csv = out_dir / "fig10_grid_advantage.csv"
    df.to_csv(csv, index=False)
    print(f"wrote {csv}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_paths_args(ap)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--babylm-dir", default=None,
                    help="monolingual BabyLM corpora root (default: <data-dir>/babylm)")
    ap.add_argument("--pairs", default=None,
                    help="measurement_pairs_mono_freq.parquet (default: <data-dir>/traj/...)")
    ap.add_argument("--phase-intra", default=None,
                    help="phase_intra parquet (default: <data-dir>/curriculum/1_intra.parquet)")
    ap.add_argument("--skip-compute", action="store_true",
                    help="replot from the existing fig10_grid_advantage.csv")
    args = ap.parse_args()
    models_dir, data_dir, out_dir = common.resolve_paths(args)

    csv = out_dir / "fig10_grid_advantage.csv"
    df = pd.read_csv(csv) if args.skip_compute else compute(args, models_dir, data_dir, out_dir)

    grid.plot_advantage(df, out_dir / "fig10_grid_advantage.png")
    grid.plot_heatmaps(df, out_dir / "fig10_grid_heatmaps.png")
    print(f"wrote {out_dir / 'fig10_grid_advantage.png'} + fig10_grid_heatmaps.png")


if __name__ == "__main__":
    main()
