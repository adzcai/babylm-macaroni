#!/usr/bin/env python3
"""Figure 8: layerwise cosine between the CS and non-CS models' word representations.

For the matched embedded / never-embedded word pairs, the cosine between the two
curriculum models' contextual representation of the same word, per layer and
language, as two lines (embedded = ink square, never embedded = grey circle),
8-seed mean +-1 SD.

    python figures/fig8_word_divergence.py   # -> out/figures/fig8_word_divergence_by_layer.pdf

Input: <results-dir>/word_divergence.csv (ships in results/; regenerate with
figures/measure_wordlevel.py).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from common import plt, LANG_ORDER, FACET, MUT, INK, lang_title, style_axes, save


def build(d, out):
    d = d[d.condition == "curriculum"]
    layers = sorted(d.layer.unique())
    groups = (("emb_cos", FACET["primary"], "embedded"),
              ("ctl_cos", FACET["secondary"], "never embedded"))
    fig, axes = plt.subplots(1, 3, figsize=(9.9, 3.5), sharey=True)
    handles = []
    for ax, lang in zip(axes, LANG_ORDER):
        dl = d[d.lang == lang]
        for col, st, lab in groups:
            piv = dl.pivot_table(index="layer", columns="seed", values=col).reindex(layers)
            m, sd = piv.mean(axis=1).values, piv.std(axis=1, ddof=0).values
            (ln,) = ax.plot(layers, m, ls=st["ls"], color=st["c"], lw=st["lw"],
                            marker=st["mk"], ms=st["ms"],
                            mfc=st["c"] if st["fill"] else "white", mew=1.1, label=lab)
            ax.fill_between(layers, m - sd, m + sd, color=st["c"], alpha=0.13, lw=0)
            if ax is axes[0]:
                handles.append(ln)
        lang_title(ax, lang)
        ax.set_xticks(layers[::2])
        ax.set_xlabel("layer", color=MUT, fontsize=9)
        style_axes(ax)
    axes[0].set_ylabel("cosine similarity\n(1 $=$ unchanged)", color=INK, fontsize=9)
    fig.tight_layout()
    fig.legend(handles=handles, labels=[g[2] for g in groups], loc="lower center",
               bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Layerwise cross-model cosine similarity of word representations",
                 fontsize=10, y=1.04)
    save(fig, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_paths_args(ap)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    _m, _d, results_dir, out_dir = common.resolve_paths(args)
    d = pd.read_csv(Path(args.csv) if args.csv else results_dir / "word_divergence.csv")
    build(d, out_dir / "fig8_word_divergence_by_layer.pdf")


if __name__ == "__main__":
    main()
