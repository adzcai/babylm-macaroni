"""Switch-density dose-response measurement (static ``wte`` alignment).

Ported, path-parameterized version of the paper's ``align_dose.py``, feeding the
``fig_doseresponse`` plot (paper ``dose_response.png``). It applies the static
alignment measure to each dose model's FINAL checkpoint using a SHARED
frequency-matched null (identical null indices across all conditions, so only
the weights differ), plus CSLS BLI, and marks attested pairs that were held out
of the maximal-density translation table (a generalization test).

CONDITION SET — reconciliation + FLAGGED UNCERTAINTY
----------------------------------------------------
The original ``align_dose.py`` measures THREE models from a shared checkpoint-0,
differing only in switch density of the training data, and the plotter
(``methods_extra_plots.fig_doseresponse``) is HARDCODED to exactly those three
``cond`` keys / gap columns::

    density:   noswitch (0%)  <  intra (sparse)  <  allcontent (maximal)
    gap cols:  gap_noswitch      gap_intra           gap_allcontent
    x-labels:  "no-switch (0%)"  "word-level (sparse)"  "all-content (maximal)"

The public model repo, however, ships the switch-TYPE ablation set
``{switch, noswitch, word, sent, par, salad}`` (+ the two curriculum runs).
``plan.md`` describes Fig 11 as spanning ``{word, sent, par, salad, switch}``
(seed 42) — FIVE conditions — which does not line up 1:1 with the plotter's
three density levels. Rather than silently invent a mapping, this module keeps
the measurement fully GENERIC over an ordered list of ``(cond_key, ckpt_dir)``
pairs and lets the caller choose both the keys and the models. ``fig11`` supplies
the default reconciliation (best-guess density interpretation), documents it,
and flags the residual uncertainty there. See ``figures/fig11_dose_response.py``.

The measurement itself is a faithful port; only the model set is caller-supplied.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from _lib.align.measure import load_wte, unit, remove_top_pcs, CSLS_K, LANGS

K_PERM = 100  # permutations for the per-pair null spread (z / percentile units)


def _mark_holdout(lex, holdout_path):
    """Add a boolean ``holdout`` column: attested pairs held out of the maximal
    translation table (both orientations). If ``holdout_path`` is absent, mark
    all False and warn (the 'held out' series will be empty)."""
    if holdout_path is None or not Path(holdout_path).is_file():
        print(f"  [align_dose] holdout table {holdout_path} not found; "
              f"marking holdout=False for all pairs (no held-out series).")
        lex["holdout"] = False
        return lex
    hold = pd.read_parquet(holdout_path)
    hold_set = set(zip(hold["embedded_lang"], hold["embedded_word"],
                       hold["matrix_lang"], hold["replaced_word"]))

    def is_holdout(r):
        return ((r.src_lang, r.src_word, r.tgt_lang, r.tgt_word) in hold_set or
                (r.tgt_lang, r.tgt_word, r.src_lang, r.src_word) in hold_set)

    lex["holdout"] = [is_holdout(r) for r in lex.itertuples(index=False)]
    return lex


def dose_response(dose_models, lex, cands, holdout_path=None, seed=0):
    """Compute the dose-response pair table + BLI table.

    Parameters
    ----------
    dose_models : list of (cond_key, ckpt_dir)
        ordered low->high switch density; ``cond_key`` becomes the ``gap_<key>``
        column / BLI ``cond`` value the plotter reads (e.g. ``"noswitch"``,
        ``"intra"``, ``"allcontent"``). ``ckpt_dir`` holds ``model.safetensors``.
    lex : DataFrame          evaluation lexicon (single-token pairs are used).
    cands : {lang: DataFrame}  frozen BLI candidate vocabularies.
    holdout_path : path or None   maximal-density held-out attested pairs.

    Returns ``(pairs_df, bli_df)``:
      * ``pairs_df``: ``lex`` rows (single-token) + ``holdout`` + per condition
        ``gap_<key>, cos_<key>, null_mu_<key>, null_sd_<key>, z_<key>, pct_<key>``.
      * ``bli_df``: columns ``cond, direction, p1, n, med_rank, n_candidates``.
    """
    lex = lex[lex["src_single"] & lex["tgt_single"]].reset_index(drop=True)
    lex = _mark_holdout(lex, holdout_path)
    rng = np.random.default_rng(seed)

    cand_ids_all = np.unique(np.concatenate([c["token_id"].values for c in cands.values()]))

    # shared null: same random cross-lingual target per pair, identical across
    # all conditions (isolates the weight difference). Draw the K extra
    # permutations AFTER `perm` so gap_<key> reproduces bit-identically.
    perm = rng.permutation(len(lex))
    perms_k = np.stack([rng.permutation(len(lex)) for _ in range(K_PERM)])
    sids = lex["src_token_id"].values
    tids = lex["tgt_token_id"].values

    bli_rows = []
    for cond, ckpt in dose_models:
        wte = load_wte(ckpt)
        basis = wte[cand_ids_all]
        W = remove_top_pcs(wte, basis, 2)  # centered + top-2 PC removed
        U = unit(W)
        cos = np.einsum("ij,ij->i", U[sids], U[tids])
        null = np.einsum("ij,ij->i", U[sids], U[tids[perm]])
        lex[f"gap_{cond}"] = cos - null
        nc = np.stack([np.einsum("ij,ij->i", U[sids], U[tids[p]]) for p in perms_k], axis=1)
        lex[f"cos_{cond}"] = cos
        lex[f"null_mu_{cond}"] = nc.mean(1)
        lex[f"null_sd_{cond}"] = nc.std(1)
        lex[f"z_{cond}"] = (cos - nc.mean(1)) / (nc.std(1) + 1e-12)
        lex[f"pct_{cond}"] = (nc < cos[:, None]).mean(1)

        # BLI CSLS per direction
        for (sl, tl), sub in lex.groupby(["src_lang", "tgt_lang"]):
            cand_t = cands[tl]
            cid = cand_t["token_id"].values
            widx = {w: i for i, w in enumerate(cand_t["word"])}
            gold = {}
            for s, t in zip(sub["src_word"], sub["tgt_word"]):
                if t in widx:
                    gold.setdefault(s, set()).add(widx[t])
            if not gold:
                continue
            qwords = sorted(gold)
            qid = {w: i for w, i in zip(sub["src_word"], sub["src_token_id"])}
            Q = U[[qid[w] for w in qwords]]
            Cm = U[cid]
            sim = Q @ Cm.T
            src_cloud = U[cands[sl]["token_id"].values]
            r_t = np.sort(Cm @ src_cloud.T, axis=1)[:, -CSLS_K:].mean(1)
            r_s = np.sort(sim, axis=1)[:, -CSLS_K:].mean(1)
            csls = 2 * sim - r_t[None, :] - r_s[:, None]
            order = np.argsort(-csls, axis=1)
            p1 = np.mean([order[i, 0] in gold[w] for i, w in enumerate(qwords)])
            best_ranks = [1 + min(r for r, c in enumerate(order[i]) if c in gold[w])
                          for i, w in enumerate(qwords)]
            bli_rows.append({"cond": cond, "direction": f"{sl}->{tl}",
                             "p1": round(float(p1), 4), "n": len(qwords),
                             "med_rank": float(np.median(best_ranks)),
                             "n_candidates": len(cid)})

    return lex, pd.DataFrame(bli_rows)
