#!/usr/bin/env python3
"""Figure 9 -- embedded-vs-own-language representation (embedded_own.png).

Panel A: own-vs-embedded cosine with the own-vs-own reliability ceiling, per
arm. Panel B: the cs invariance advantage (shift_nsw - shift_cs) per layer.
3-seed mean +- 1 SD.

Pipeline (one invocation):
  sample own1/own2/emb contexts per lang  ->  extract per-(word,model,layer)
  reps (curriculum + curriculum_noswitch FINAL checkpoints)  ->  per-word
  cross/reliab/shift  ->  out/figures/fig09_embedded_own.csv  ->  .png

Run:  python figures/fig09_embedded_own.py [--seeds 42 43 44] [--skip-compute]

CORPUS FLAGS: needs the monolingual BabyLM corpora (--babylm-dir), the mining
output attested_lexicon.parquet (--attested), and phase_intra, which maps to
the DOWNLOADED data/curriculum/1_intra.parquet (--phase-intra). Only the
BabyLM corpora + attested_lexicon are NOT fetched by download_data.py --
see figures/_lib/embown.py.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import common`
import common
from _lib import embown, contextual

LANGS = ("eng", "nld", "zho")
CONDITIONS = {"cur_cs": "curriculum", "cur_nsw": "curriculum_noswitch"}


def compute(args, models_dir, data_dir, out_dir):
    babylm_dir = Path(args.babylm_dir) if args.babylm_dir else data_dir / "babylm"
    attested = Path(args.attested) if args.attested else data_dir / "align" / "attested_lexicon.parquet"
    phase_intra = Path(args.phase_intra) if args.phase_intra else data_dir / "curriculum" / "1_intra.parquet"

    frames = []
    for seed in args.seeds:
        model_paths = {lbl: common.final_ckpt(common.model_dir(cond, seed, models_dir))
                       for lbl, cond in CONDITIONS.items()}
        tokenizer_json = model_paths["cur_cs"] / "tokenizer.json"

        # sample per-lang, concatenate into one contexts_all jsonl (keys carry lang)
        ctx_all = out_dir / f"fig09_contexts_all_s{seed}.jsonl"
        with open(ctx_all, "w") as f:
            for lang in LANGS:
                recs = embown.sample_contexts(
                    lang, attested, phase_intra, babylm_dir, tokenizer_json, k=args.k)
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        npz = contextual.extract_reps(
            model_paths, ctx_all, tokenizer_dir=model_paths["cur_cs"],
            device=args.device, batch_size=args.batch_size)
        df = embown.analyze(npz)
        df["seed"] = seed
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    csv = out_dir / "fig09_embedded_own.csv"
    out.to_csv(csv, index=False)
    print(f"wrote {csv}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_paths_args(ap)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--k", type=int, default=20, help="contexts per (word, ctype)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--babylm-dir", default=None,
                    help="monolingual BabyLM corpora root (default: <data-dir>/babylm)")
    ap.add_argument("--attested", default=None,
                    help="attested_lexicon.parquet (default: <data-dir>/align/attested_lexicon.parquet)")
    ap.add_argument("--phase-intra", default=None,
                    help="phase_intra parquet (default: <data-dir>/curriculum/1_intra.parquet)")
    ap.add_argument("--skip-compute", action="store_true",
                    help="replot from the existing fig09_embedded_own.csv")
    args = ap.parse_args()
    models_dir, data_dir, out_dir = common.resolve_paths(args)

    csv = out_dir / "fig09_embedded_own.csv"
    df = pd.read_csv(csv) if args.skip_compute else compute(args, models_dir, data_dir, out_dir)

    embown.plot_embedded_own(df, out_dir / "fig09_embedded_own.png")
    print(f"wrote {out_dir / 'fig09_embedded_own.png'}")


if __name__ == "__main__":
    main()
