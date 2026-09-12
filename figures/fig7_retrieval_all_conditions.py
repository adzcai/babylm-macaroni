#!/usr/bin/env python3
"""Figure 7: per-layer bitext retrieval P@1 for all eight conditions.

One row of three language-pair panels with all eight conditions overlaid: the
four main conditions (CS / non-CS x curriculum / shuffled) in neutrals, the four
shuffled-ordering controls in colour. 8-seed means, bands +-1 SD, layers 6-11
shaded, dotted chance line.

    python figures/fig7_retrieval_all_conditions.py   # -> out/figures/fig7_retrieval_per_layer_all8.pdf

Inputs: <results-dir>/retrieval_finals.csv + retrieval_controls.csv.
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
from common import INK, MUT, CORPUS_C, CONTROL_C, PAIRS, pair_title, plt, save

N_PROBE = 997
CHANCE = 100.0 / N_PROBE
UPPER = (6, 11)
NEUTRAL = "#5c6b7a"
NEUTRAL_LIGHT = "#93a1af"

# Paper abbreviation -> (public condition name, group column).
COND = {
    "CS/Curr": ("curriculum", "switched"),
    "Curr": ("curriculum_noswitch", "noswitch"),
    "CS/Shuf": ("switch", "switched"),
    "Shuf": ("noswitch", "noswitch"),
    "Word": ("word", "switched"),
    "Sent": ("sent", "switched"),
    "Par": ("par", "switched"),
    "Salad": ("salad", "switched"),
}
STYLE = {
    "CS/Curr": dict(c=INK, ls="-", mk="s", lw=1.7, ms=3.4, fill=True),
    "CS/Shuf": dict(c=INK, ls="-", mk="o", lw=1.3, ms=3.4, fill=False),
    "Curr": dict(c=NEUTRAL, ls="-", mk="s", lw=1.5, ms=3.4, fill=True),
    "Shuf": dict(c=NEUTRAL_LIGHT, ls="-", mk="o", lw=1.5, ms=3.4, fill=False),
    "Word": dict(c=CORPUS_C["word"], ls="-", mk="^", lw=1.5, ms=3.8, fill=True),
    "Sent": dict(c=CORPUS_C["sent"], ls="-", mk="P", lw=1.5, ms=4.0, fill=True),
    "Par": dict(c=CONTROL_C["par"], ls="-", mk="D", lw=1.5, ms=3.0, fill=True),
    "Salad": dict(c=CONTROL_C["salad"], ls="-", mk="x", lw=1.5, ms=3.8, fill=True),
}
ORDER = ["Curr", "Shuf", "Salad", "Word", "Sent", "Par", "CS/Shuf", "CS/Curr"]
LEGEND_ORDER = ["CS/Curr", "CS/Shuf", "Curr", "Shuf", "Word", "Sent", "Par", "Salad"]


def load(paths):
    agg = collections.defaultdict(list)
    layers = set()
    by_prefix = {(prefix, group): name for name, (prefix, group) in COND.items()}
    for path in paths:
        for r in csv.DictReader(open(path)):
            prefix = r["model"]
            head, sep, tail = prefix.rpartition("-s")
            if sep and tail.isdigit():
                prefix = head
            name = by_prefix.get((prefix, r["group"]))
            if name is None:
                continue
            L = int(r["layer"])
            layers.add(L)
            agg[(name, L, r["pair"])].append(100.0 * float(r["retrieval_p1"]))
    return agg, sorted(layers)


def draw(ax, agg, layers, name, pair):
    s = STYLE[name]
    vals = [agg.get((name, L, pair), []) for L in layers]
    mean = [statistics.mean(v) if v else float("nan") for v in vals]
    sd = [statistics.pstdev(v) if len(v) > 1 else 0.0 for v in vals]
    ax.fill_between(layers, [m - d for m, d in zip(mean, sd)], [m + d for m, d in zip(mean, sd)],
                    color=s["c"], alpha=0.14, linewidth=0)
    ax.plot(layers, mean, ls=s["ls"], lw=s["lw"], color=s["c"], solid_capstyle="round")
    ax.plot(layers, mean, ls="none", marker=s["mk"], ms=s["ms"],
            mfc=s["c"] if s["fill"] else "white", mec=s["c"], mew=0.9)


def proxy(name):
    s = STYLE[name]
    return Line2D([0], [0], color=s["c"], ls=s["ls"], lw=s["lw"], marker=s["mk"], ms=s["ms"],
                  mfc=s["c"] if s["fill"] else "white", mec=s["c"], label=name)


def build(agg, layers, out):
    missing = [n for n in COND if not any(k[0] == n for k in agg)]
    if missing:
        raise SystemExit(f"no rows for conditions {missing} in the input CSVs")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.95), sharex=True, sharey=True)
    for ax, pair in zip(axes, PAIRS):
        ax.axvspan(UPPER[0], UPPER[1], color="#000000", alpha=0.045, linewidth=0, zorder=0)
        for name in ORDER:
            draw(ax, agg, layers, name, pair)
        ax.axhline(CHANCE, ls=":", lw=1.2, color=MUT)
        ax.set_ylim(-4, 100)
        ax.set_xticks([0, 3, 6, 9, 12])
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.set_xlabel("layer", fontsize=9)
        pair_title(ax, pair, fontsize=9.5)
    axes[0].set_ylabel("bitext retrieval P@1 (%)", fontsize=9)
    handles = [proxy(n) for n in LEGEND_ORDER]
    handles.append(Line2D([0], [0], color=MUT, ls=":", lw=1.2, label="chance (1/N)"))
    fig.tight_layout(w_pad=0.8, rect=(0, 0.09, 1, 1))
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=9, frameon=False,
               columnspacing=1.1, handlelength=2.2, handletextpad=0.5, bbox_to_anchor=(0.5, 0.0))
    save(fig, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_paths_args(ap)
    ap.add_argument("--main-csv", default=None)
    ap.add_argument("--controls-csv", default=None)
    args = ap.parse_args()
    _m, _d, results_dir, out_dir = common.resolve_paths(args)
    main_csv = Path(args.main_csv) if args.main_csv else results_dir / "retrieval_finals.csv"
    ctrl_csv = Path(args.controls_csv) if args.controls_csv else results_dir / "retrieval_controls.csv"
    agg, layers = load([main_csv, ctrl_csv])
    build(agg, layers, out_dir / "fig7_retrieval_per_layer_all8.pdf")


if __name__ == "__main__":
    main()
