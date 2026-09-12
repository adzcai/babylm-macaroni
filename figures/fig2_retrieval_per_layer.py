#!/usr/bin/env python3
"""Figure 2 (and, via --metric median_pct_rank, Figure 6): per-layer FLORES+ bitext retrieval.

Code-switched vs non-CS models, one panel per language pair, mean +-1 SD over runs,
dotted chance line, layers 6-11 shaded. ``--arm both`` (default, the paper's
Figure 2) pools the curriculum and shuffled orderings (8 seeds each -> 16 runs
per group); ``--arm curriculum`` / ``random`` plot one ordering.

    python figures/fig2_retrieval_per_layer.py                 # -> out/figures/fig2_retrieval_per_layer.pdf
    python figures/fig6_retrieval_median_rank.py               # the same code, median percentile rank

Input: <results-dir>/retrieval_finals.csv (the paper's CSV ships in results/;
regenerate it with figures/measure_retrieval.py --set finals).
"""
import argparse
import collections
import csv
import statistics
import sys
from pathlib import Path

from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from common import FACET, MUT, PAIRS, pair_title, plt, save

ARM = {"switched": dict(FACET["primary"], name="code-switched"),
       "noswitch": dict(FACET["secondary"], name="non-CS")}
GROUPS = ["noswitch", "switched"]  # draw order: baseline first, CS on top
N_PROBE = 997
UPPER = (6, 11)
METRICS = {
    "retrieval_p1": dict(scale=100.0, chance=100.0 / N_PROBE, lim=(-4, 100),
                         label="bitext retrieval P@1 (%)"),
    "median_pct_rank": dict(scale=1.0, chance=0.5, lim=(0.42, 1.03),
                            label="median percentile rank\nof the correct translation"),
    "mean_pct_rank": dict(scale=1.0, chance=0.5, lim=(0.42, 1.03),
                          label="mean percentile rank\nof the correct translation"),
    "mrr": dict(scale=1.0, chance=None, lim=(-0.04, 1.03), label="mean reciprocal rank"),
    "retrieval_p5": dict(scale=100.0, chance=500.0 / N_PROBE, lim=(-4, 100),
                         label="bitext retrieval P@5 (%)"),
}


def arm_of(model):
    return "curriculum" if "curriculum" in model else "random"


def load(path, arm, metric):
    agg = collections.defaultdict(list)
    layers = set()
    rows = list(csv.DictReader(open(path)))
    if rows and metric not in rows[0]:
        raise SystemExit(f"column '{metric}' is not in {path}")
    scale = METRICS[metric]["scale"]
    for r in rows:
        if arm != "both" and arm_of(r["model"]) != arm:
            continue
        L = int(r["layer"])
        layers.add(L)
        agg[(r["group"], L, r["pair"])].append(float(r[metric]) * scale)
    return agg, sorted(layers)


def build(agg, layers, metric, out):
    M = METRICS[metric]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0), sharey=True)
    for ax, pair in zip(axes, PAIRS):
        ax.axvspan(UPPER[0], UPPER[1], color="#000000", alpha=0.045, linewidth=0, zorder=0)
        for group in GROUPS:
            a = ARM[group]
            vals = [agg.get((group, L, pair), []) for L in layers]
            mean = [statistics.mean(v) if v else float("nan") for v in vals]
            sd = [statistics.pstdev(v) if len(v) > 1 else 0.0 for v in vals]
            ax.fill_between(layers, [m - s for m, s in zip(mean, sd)],
                            [m + s for m, s in zip(mean, sd)], color=a["c"], alpha=0.16, linewidth=0)
            ax.plot(layers, mean, ls=a["ls"], lw=a["lw"], color=a["c"], solid_capstyle="round")
            ax.plot(layers, mean, ls="none", marker=a["mk"], ms=a["ms"],
                    mfc=a["c"] if a["fill"] else "white", mec=a["c"], mew=1.0)
        if M["chance"] is not None:
            ax.axhline(M["chance"], ls=":", lw=1.3, color=MUT)
        ax.set_ylim(*M["lim"])
        pair_title(ax, pair, fontsize=14, gap=0.02)
        ax.set_xlabel("layer", fontsize=9)
        ax.set_xticks([0, 3, 6, 9, 12])
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel(M["label"], fontsize=9)
    proxies = [Line2D([0], [0], color=ARM[g]["c"], ls=ARM[g]["ls"], lw=ARM[g]["lw"],
                      marker=ARM[g]["mk"], ms=ARM[g]["ms"],
                      mfc=ARM[g]["c"] if ARM[g]["fill"] else "white",
                      mec=ARM[g]["c"], label=ARM[g]["name"])
               for g in ("switched", "noswitch")]
    if M["chance"] is not None:
        proxies.append(Line2D([0], [0], color=MUT, ls=":", lw=1.3,
                              label="chance" if metric != "retrieval_p1" else "chance (1/N)"))
    fig.legend(handles=proxies, fontsize=9, ncol=len(proxies), frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.06),
               handlelength=2.6, columnspacing=2.0)
    fig.tight_layout()
    save(fig, out)


def main(default_metric="retrieval_p1", default_name="fig2_retrieval_per_layer"):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_paths_args(ap)
    ap.add_argument("--csv", default=None, help="override: retrieval CSV to plot")
    ap.add_argument("--arm", choices=["both", "curriculum", "random"], default="both")
    ap.add_argument("--metric", choices=list(METRICS), default=default_metric)
    ap.add_argument("--name", default=default_name, help="output basename")
    args = ap.parse_args()
    _m, _d, results_dir, out_dir = common.resolve_paths(args)
    csv_path = Path(args.csv) if args.csv else results_dir / "retrieval_finals.csv"
    agg, layers = load(csv_path, args.arm, args.metric)
    build(agg, layers, args.metric, out_dir / f"{args.name}.pdf")


if __name__ == "__main__":
    main()
