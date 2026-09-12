#!/usr/bin/env python3
"""Regenerate every data figure of the paper from the measurement CSVs.

    python figures/make_all.py                          # paper CSVs in results/ -> out/figures/
    python figures/make_all.py --results-dir out/results  # your own measurements

Runs fig2, fig3, fig4, fig6, fig7, fig8 in turn (each reads a CSV, writes a PDF
+ PNG) and prints a pass/fail summary. The two hand-drawn figures (1 and 5) are
TikZ/LaTeX: `bash figures/tex/build.sh`.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURES = ["fig2_retrieval_per_layer", "fig3_retrieval_trajectory",
           "fig4_exposure_generalization", "fig6_retrieval_median_rank",
           "fig7_retrieval_all_conditions", "fig8_word_divergence"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--out", default="out/figures")
    ap.add_argument("--only", nargs="+", metavar="FIG", choices=FIGURES)
    args = ap.parse_args()

    results = {}
    for fig in args.only or FIGURES:
        cmd = [sys.executable, str(HERE / f"{fig}.py"), "--out", args.out]
        if args.results_dir:
            cmd += ["--results-dir", args.results_dir]
        print(f"\n=== {fig} ===\n$ {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd).returncode
        results[fig] = "ok" if rc == 0 else f"FAILED (exit {rc})"
    print("\n" + "=" * 50 + "\nSUMMARY")
    for fig, r in results.items():
        print(f"  {r:20s} {fig}")
    if any(r != "ok" for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
