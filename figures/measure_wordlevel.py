#!/usr/bin/env python3
"""Measure the word-level analyses: models -> results CSVs (Figs 4 and 8).

    python figures/measure_wordlevel.py --seeds 42 43 44

Per seed, on the FINAL checkpoints of `curriculum` (CS) and `curriculum_noswitch`
(non-CS):

  1. mine the lexicon + exposure sets from the stage-1 corpora      -> data/align/
  2. monolingual frequency tables from the BabyLM corpora           -> data/freq_{lang}.tsv
  3. type -> part-of-speech map (spaCy / jieba) over the non-CS stage-1 corpus
  4. seen-embedded vs never-embedded word pairs, matched 1:1 (Hungarian)
  5. K=20 BabyLM sentences per word, frozen and hashed
  6. contextual representations of every word at every layer (GPU)
  7. Fig 4 measure (exposure_rank) and Fig 8 measure (divergence)

Writes results/exposure_rank.csv and results/word_divergence.csv (into
--results-dir; OVERWRITES the paper's CSVs unless you pass another dir).
Intermediates live under <out>/wordlevel/s<seed>/ and are reused on re-runs.

Inputs beyond download_data.py: `python download_aux.py` (monolingual BabyLM
corpora into data/babylm/ + MUSE dictionaries into data/muse/) and the spaCy
models `en_core_web_sm`, `nl_core_news_sm`. Steps 1-5 are corpus-level and
shared across seeds (run once, then symlinked); step 6 needs a GPU.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from _lib import traj
from _lib.align import prepare_lexicon
from _lib.babylm import ensure_freq_tables

SHARED = ["pos_map.parquet", "measurement_pairs_mono_freq.parquet",
          "contexts_eng.jsonl", "contexts_nld.jsonl", "contexts_zho.jsonl"]


def link_shared(src_dir, dst_dir):
    """Reuse the corpus-level intermediates of the first seed for the others."""
    for f in SHARED:
        s, d = src_dir / f, dst_dir / f
        if s.exists() and not d.exists():
            os.symlink(os.path.relpath(s, dst_dir), d)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_paths_args(ap)
    ap.add_argument("--seeds", nargs="+", type=int, default=common.PUBLISHED_SEEDS)
    ap.add_argument("--babylm-dir", default=None,
                    help="monolingual BabyLM corpora (default: <data-dir>/babylm)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trajectory", action="store_true",
                    help="also extract the 10 trajectory checkpoints per arm (if trained); "
                         "exposure_rank.csv then carries every checkpoint")
    args = ap.parse_args()
    models_dir, data_dir, results_dir, out_dir = common.resolve_paths(args)
    results_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "wordlevel"
    babylm_dir = Path(args.babylm_dir) if args.babylm_dir else data_dir / "babylm"

    # Corpus-level inputs (seed-independent).
    tokenizer = common.load_tokenizer(common.model_dir("curriculum", args.seeds[0], models_dir))
    prepare_lexicon(data_dir, tokenizer)
    ensure_freq_tables(babylm_dir, data_dir)

    first = None
    exp_frames, div_frames = [], []
    for seed in args.seeds:
        cfg = traj.build_config(models_dir, data_dir, work / f"s{seed}", seed, babylm_dir=babylm_dir)
        if first is not None:
            link_shared(first, cfg["out_dir"])
        traj.build_pos_map(cfg)
        pairs = traj.build_measurement_pairs(cfg, freq_col="mono_freq")
        traj.sample_contexts(cfg, pairs)
        ckpts = traj.final_checkpoints(cfg)
        if args.trajectory:
            ckpts = traj.discover_checkpoints(cfg)
        traj.extract_reps(cfg, ckpt_list=ckpts, device=args.device)
        exp_frames.append(traj.exposure_rank(cfg, pairs))
        div_frames.append(traj.divergence_layerwise(cfg, pairs).assign(seed=seed))
        first = first or cfg["out_dir"]

    for frames, name in ((exp_frames, "exposure_rank.csv"), (div_frames, "word_divergence.csv")):
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(results_dir / name, index=False)
        print(f"wrote {len(df)} rows -> {results_dir / name}")


if __name__ == "__main__":
    main()
