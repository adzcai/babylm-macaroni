#!/usr/bin/env python3
"""Figure 8 -- representation divergence (repr_divergence_raw.png).

Cross-arm cosine of the same word's CONTEXTUAL representation (curriculum cs
vs. curriculum unilingual), per layer, for embedded words vs. frequency-matched
never-embedded controls, per language; 3-seed mean +- 1 SD.

Pipeline (one invocation):
  build embedded/control word sets  ->  sample monolingual contexts per lang
  ->  extract per-(word,model,layer) reps (curriculum + curriculum_noswitch
      FINAL checkpoints)  ->  per-layer embedded-vs-control contrast
  ->  out/figures/fig08_repr_divergence.csv  ->  fig08_repr_divergence.png

Run:  python figures/fig08_repr_divergence.py [--seeds 42 43 44] [--skip-compute]

CORPUS FLAGS: needs the monolingual BabyLM corpora (--babylm-dir), the mining
outputs (--align-dir: attested_lexicon + exposure_sets) and monolingual freq
tables (--freq-dir: freq_{lang}.tsv). None are fetched by download_data.py --
see figures/_lib/contextual.py. phase_intra is NOT used by this figure.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import common`
import common
from _lib import contextual

LANGS = ("eng", "nld", "zho")
CONDITIONS = {"cur_cs": "curriculum", "cur_nsw": "curriculum_noswitch"}
PAIRS = {"curriculum": ("cur_cs", "cur_nsw")}


def compute(args, models_dir, data_dir, out_dir):
    align_dir = Path(args.align_dir) if args.align_dir else data_dir / "align"
    freq_dir = Path(args.freq_dir) if args.freq_dir else data_dir
    babylm_dir = Path(args.babylm_dir) if args.babylm_dir else data_dir / "babylm"

    vocab, ctrl = contextual.build_wordsets(align_dir, freq_dir)

    frames = []
    for seed in args.seeds:
        model_paths = {lbl: common.final_ckpt(common.model_dir(cond, seed, models_dir))
                       for lbl, cond in CONDITIONS.items()}
        tokenizer_json = common.final_ckpt(
            common.model_dir("curriculum", seed, models_dir)) / "tokenizer.json"

        npz_by_lang = {}
        for lang in LANGS:
            ctx = out_dir / f"fig08_contexts_{lang}_s{seed}.jsonl"
            contextual.sample_monolingual_contexts(
                vocab, ctrl, lang, babylm_dir, tokenizer_json, k=args.k, out=ctx)
            npz_by_lang[lang] = contextual.extract_reps(
                model_paths, ctx, tokenizer_dir=model_paths["cur_cs"],
                device=args.device, batch_size=args.batch_size)
        df = contextual.contextual_layerwise(npz_by_lang, pairs=PAIRS)
        df["seed"] = seed
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    csv = out_dir / "fig08_repr_divergence.csv"
    out.to_csv(csv, index=False)
    print(f"wrote {csv}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_paths_args(ap)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--k", type=int, default=15, help="monolingual contexts per word")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--babylm-dir", default=None,
                    help="monolingual BabyLM corpora root (default: <data-dir>/babylm)")
    ap.add_argument("--align-dir", default=None,
                    help="dir with attested_lexicon/exposure_sets parquet (default: <data-dir>/align)")
    ap.add_argument("--freq-dir", default=None,
                    help="dir with freq_{lang}.tsv (default: <data-dir>)")
    ap.add_argument("--skip-compute", action="store_true",
                    help="replot from the existing fig08_repr_divergence.csv")
    args = ap.parse_args()
    models_dir, data_dir, out_dir = common.resolve_paths(args)

    csv = out_dir / "fig08_repr_divergence.csv"
    if args.skip_compute:
        df = pd.read_csv(csv)
    else:
        df = compute(args, models_dir, data_dir, out_dir)

    df = df[df.condition == "curriculum"]
    contextual.plot_repr_divergence(df, out_dir / "fig08_repr_divergence.png")
    print(f"wrote {out_dir / 'fig08_repr_divergence.png'}")


if __name__ == "__main__":
    main()
