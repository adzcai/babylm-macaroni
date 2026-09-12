#!/usr/bin/env python3
"""Figure 6: per-layer median percentile rank of the correct translation.

The same plot as Figure 2 on the rank metric, which separates a near-chance
baseline from a broadly-but-imprecisely aligned one (P@1 cannot).

    python figures/fig6_retrieval_median_rank.py   # -> out/figures/fig6_retrieval_median_rank.pdf

Input: <results-dir>/retrieval_finals.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig2_retrieval_per_layer import main

if __name__ == "__main__":
    main(default_metric="median_pct_rank", default_name="fig6_retrieval_median_rank")
