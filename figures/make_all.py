#!/usr/bin/env python3
"""Reproduce every paper figure: run each figNN_*.py over models/ -> out/figures/.

Each figure script is self-contained (load models -> compute CSV -> plot). This
just invokes them in turn and reports a pass/fail summary, so one broken figure
(e.g. one needing trajectory checkpoints not published on HF) does not stop the
rest.

    python figures/make_all.py --models-dir models --data-dir data --out out/figures
    python figures/make_all.py --only fig02_05_retrieval_per_layer fig08_repr_divergence

Figures 3, 6, 7 require PER-CHECKPOINT models for the no-CS arm, which are not on
HF (see README) — they will fail with a friendly error unless you supply those
checkpoints. Figures 2, 5, 8, 9, 10, 11 need only the published final checkpoints.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The figure entrypoints, in paper order. Names match the figNN_*.py files.
FIGURES = [
    "fig02_05_retrieval_per_layer",
    "fig03_retrieval_trajectory",
    "fig04_06_wordlevel",
    "fig07_dd_trajectory",
    "fig08_repr_divergence",
    "fig09_embedded_own",
    "fig10_grid_advantage",
    "fig11_dose_response",
]

# Figures that need trajectory checkpoints absent from HF (expected to fail bare).
NEEDS_TRAJECTORY = {"fig03_retrieval_trajectory", "fig04_06_wordlevel", "fig07_dd_trajectory"}

# Fig 11 reproduces only a partial dose-response (its maximal 'allcontent' point
# is not reproducible from public artifacts), so it is opt-in, not run by default.
# Request it explicitly with --only fig11_dose_response (or --all).
OPT_IN = {"fig11_dose_response"}
DEFAULT_FIGURES = [f for f in FIGURES if f not in OPT_IN]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="out/figures")
    p.add_argument("--only", nargs="+", metavar="FIG",
                   help="run only these figure scripts (default: all except opt-in ones)")
    p.add_argument("--all", action="store_true",
                   help=f"also run opt-in figures ({', '.join(sorted(OPT_IN))})")
    p.add_argument("--python", default=sys.executable, help="python interpreter to use")
    args = p.parse_args()

    targets = args.only or (FIGURES if args.all else DEFAULT_FIGURES)
    unknown = [t for t in targets if t not in FIGURES]
    if unknown:
        p.error(f"unknown figure(s): {unknown}. choose from {FIGURES}")

    results: dict[str, str] = {}
    for fig in targets:
        script = HERE / f"{fig}.py"
        if not script.is_file():
            results[fig] = "MISSING (script not found)"
            continue
        cmd = [args.python, str(script),
               "--models-dir", args.models_dir,
               "--data-dir", args.data_dir,
               "--out", args.out]
        print(f"\n=== {fig} ===\n$ {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            results[fig] = "ok"
        else:
            note = " (needs trajectory checkpoints not on HF)" if fig in NEEDS_TRAJECTORY else ""
            results[fig] = f"FAILED (exit {rc}){note}"

    print("\n" + "=" * 60 + "\nSUMMARY")
    for fig in targets:
        print(f"  {results[fig]:45s}  {fig}")
    if any(v.startswith("FAILED") or v.startswith("MISSING") for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
