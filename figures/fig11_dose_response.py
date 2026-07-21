#!/usr/bin/env python3
"""Fig 11 — switch-density dose-response of cross-lingual alignment.

Applies the static (``wte``) alignment measure to each dose model's FINAL
checkpoint under a SHARED frequency-matched null, plus CSLS BLI. Ported from
``align_dose.py`` (measurement -> ``_lib/align_dose.py``) + the ``fig_doseresponse``
plotter in ``methods_extra_plots.py``.

Outputs (into ``--out``, default ``out/figures``):
  * ``fig11_dose_response.csv``       per-pair table (gap_<cond>, cos_<cond>, ...)
  * ``fig11_dose_response_bli.csv``   BLI CSLS P@1 by (cond, direction)
  * ``fig11_dose_response.png``       2-panel figure (cosine gap | BLI P@1)

Run: ``python figures/fig11_dose_response.py [--seed 42] [--holdout PATH] [--skip-compute]``

============================================================================
DOSE SERIES — WHAT THIS REPO CAN REPRODUCE
============================================================================
The PAPER's Fig 11 is an inverted-U over THREE density levels — no switching
(``noswitch``), sparse word-level switching (``intra``), and MAXIMAL switching
(``allcontent``) — and its headline is that alignment peaks at *sparse* switching
and falls off at maximal density.

The ``allcontent`` (maximal) point is **not reproducible from the public
artifacts**: it was a bespoke maximal-relexification corpus/model produced by the
corpus-synthesis pipeline, which is intentionally out of scope for this repo (no
dataset-creation code is shipped), and neither its corpus nor its model is among
the published subsets/branches. There is no public condition that faithfully
stands in for it (``salad`` is a shuffled-token control, a different kind of
corruption, not a maximal switch-density model).

So this repo reproduces only the doses backed by public models — the RISING part
of the curve (none -> sparse) — using an EXACT mapping (no guesswork):

    plotter cond    public model      rationale
    ------------    ------------      ------------------------------------------
    noswitch    <-  noswitch          0% switching (exact match)
    intra       <-  word              word-level (sparse intra-sentential) switch

This shows that sparse switching improves cross-lingual alignment over none; it
does NOT show the maximal-density turnover (that needs the unreproducible
``allcontent`` model). If you train such a model, add it with a third
``--dose <key>=<condition>`` entry to recover the full inverted-U.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import common` / `_lib`
import common
from common import plt, MUT, INK, TIER, PAIR, style_axes, save, model_dir, final_ckpt
from _lib import align, align_dose

SEED_DEFAULT = 42

# plotter cond key -> public model condition (see module docstring; override via --dose).
# Only the doses backed by PUBLIC models are included; the paper's maximal
# 'allcontent' point is not reproducible here (add via --dose if you train one).
DEFAULT_DOSE = [("noswitch", "noswitch"), ("intra", "word")]
XLAB = ["no-switch\n(0%)", "word-level\n(sparse)", "all-content\n(maximal)"]


def compute(models_dir, data_dir, out_dir, seed, dose, holdout):
    tok = common.load_tokenizer(model_dir(dose[0][1], seed, models_dir))
    lex, cands, _cfreq = align.prepare_lexicon(
        data_dir, tok, cache_dir=out_dir / "align_cache")
    dose_models = [(key, final_ckpt(model_dir(cond, seed, models_dir)))
                   for key, cond in dose]
    pairs, bli = align_dose.dose_response(dose_models, lex, cands,
                                          holdout_path=holdout, seed=seed)
    pcsv = out_dir / "fig11_dose_response.csv"
    bcsv = out_dir / "fig11_dose_response_bli.csv"
    pairs.to_csv(pcsv, index=False)
    bli.to_csv(bcsv, index=False)
    print(f"wrote {pcsv}\nwrote {bcsv}")
    return pairs, bli


def plot(pairs, bli, dose, out_dir):
    keys = [k for k, _ in dose]                 # cond keys, low -> high density
    dose_cols = [f"gap_{k}" for k in keys]
    order = {k: i for i, k in enumerate(keys)}
    x = range(len(keys))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.0))

    # Panel A: cosine gap by dose, per tier
    def row(mask):
        return [pairs.loc[mask, c].mean() for c in dose_cols]
    a_in = row((pairs.tier == "attested") & (~pairs.holdout))
    a_ho = row((pairs.tier == "attested") & (pairs.holdout))
    p_un = row(pairs.tier == "pair_unseen")
    axA.plot(x, a_in, "-o", color=TIER["attested"], lw=2.6, ms=6,
             label="switched pairs (in table)", zorder=4)
    if pairs.holdout.any():
        axA.plot(x, a_ho, "--s", color=TIER["attested"], lw=1.8, ms=5, alpha=0.85,
                 label="switched pairs (held out)", zorder=3)
    axA.plot(x, p_un, "-o", color=TIER["pair_unseen"], lw=2.4, ms=5.5,
             label="novel pairings", zorder=4)
    axA.axhline(0, color=MUT, lw=1, ls="--")
    axA.set_xticks(list(x))
    axA.set_xticklabels(XLAB[:len(keys)], fontsize=9)
    axA.set_ylabel("mean cross-lingual cosine gap", color=INK)
    axA.set_title("A · Alignment (cosine gap)", fontsize=11, loc="left")
    axA.legend(frameon=False, fontsize=8.5, loc="upper right")
    axA.set_xlim(-0.2, len(keys) - 0.8)
    style_axes(axA)

    # Panel B: BLI P@1 by dose, 6 directions (color = unordered pair, style = direction)
    pair_col = dict(PAIR)
    for direction, s in bli.groupby("direction"):
        a, b = direction.split("->")
        key = "-".join(sorted([a, b]))
        col = pair_col[key]
        ls = "-" if a < b else "--"
        s = s.assign(o=s.cond.map(order)).sort_values("o")
        axB.plot([order[c] for c in s.cond], s.p1, ls, color=col, lw=2, marker="o",
                 ms=5, alpha=0.9)
    axB.set_xticks(list(x))
    axB.set_xticklabels(XLAB[:len(keys)], fontsize=9)
    axB.set_ylabel("BLI accuracy  (CSLS precision@1)", color=INK)
    axB.set_title("B · Nearest-neighbor translation (BLI)", fontsize=11, loc="left")
    axB.set_xlim(-0.2, len(keys) - 0.8)
    from matplotlib.lines import Line2D
    leg1 = [Line2D([0], [0], color=c, lw=2.4, label=k) for k, c in pair_col.items()]
    leg2 = [Line2D([0], [0], color=MUT, lw=2, ls="-", label="a→b (alphabetical)"),
            Line2D([0], [0], color=MUT, lw=2, ls="--", label="b→a")]
    l1 = axB.legend(handles=leg1, frameon=False, fontsize=8.5, loc="upper right",
                    title="language pair")
    axB.add_artist(l1)
    axB.legend(handles=leg2, frameon=False, fontsize=8, loc="lower right")
    style_axes(axB)

    _title = ("Switch-density dose-response — sparse switching improves alignment over none"
              if len(keys) < 3 else
              "Switch-density dose-response — alignment peaks at sparse switching, not maximal (inverted-U)")
    fig.suptitle(_title, fontsize=12, y=1.0)
    fig.tight_layout()
    out = out_dir / "fig11_dose_response.png"
    save(fig, out)
    print(f"wrote {out.name}")


def _parse_dose(items):
    if not items:
        return DEFAULT_DOSE
    return [(kv.split("=", 1)[0], kv.split("=", 1)[1]) for kv in items]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_paths_args(ap)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--dose", nargs="+", metavar="KEY=CONDITION", default=None,
                    help="override the density series, e.g. "
                         "'noswitch=noswitch intra=word allcontent=salad' "
                         "(default; see module docstring for the flagged mapping)")
    ap.add_argument("--holdout", default=None,
                    help="parquet of attested pairs held out of the maximal-density "
                         "translation table (embedded_lang/word, matrix_lang/replaced_word). "
                         "Absent -> the 'held out' series is omitted.")
    ap.add_argument("--skip-compute", action="store_true",
                    help="replot from existing fig11_dose_response{,_bli}.csv")
    args = ap.parse_args()
    models_dir, data_dir, out_dir = common.resolve_paths(args)
    dose = _parse_dose(args.dose)
    print("[fig11] dose mapping (plotter cond <- public model): "
          + ", ".join(f"{k}<-{c}" for k, c in dose))

    pcsv = out_dir / "fig11_dose_response.csv"
    bcsv = out_dir / "fig11_dose_response_bli.csv"
    if args.skip_compute:
        if not (pcsv.is_file() and bcsv.is_file()):
            raise SystemExit(f"--skip-compute but {pcsv} / {bcsv} missing; run compute first.")
        pairs, bli = pd.read_csv(pcsv), pd.read_csv(bcsv)
    else:
        pairs, bli = compute(models_dir, data_dir, out_dir, args.seed, dose, args.holdout)
    plot(pairs, bli, dose, out_dir)


if __name__ == "__main__":
    main()
