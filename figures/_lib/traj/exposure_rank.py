"""Word-level exposure generalisation on the retrieval-rank metric (Fig 4).

Port of the paper's ``exposure_rank.py``. For each source word w (seen-embedded E
or its matched never-embedded control C), rank its true translation among every
target-language word of the matched set in that direction, on the layer-6-11
mean of the contextual representation:

    prank = 1 - (rank - 1) / (pool - 1)      1.0 = nearest neighbour, 0.5 = chance
    Delta = median_pairs(prank_cs) - median_pairs(prank_non-CS)

per checkpoint, group and direction (plus ``pooled``), with a pair bootstrap CI
on the difference of medians. Subsets: ``all`` and ``balanced`` (pairs whose BOTH
translation equivalents were ever embedded).

Output schema (== results/exposure_rank.csv): seed, subset, group, direction,
stage, epoch, delta, ci_lo, ci_hi, median_cs, median_noswitch, mean_delta, n.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CKPT_ORDER

GROUP = {"E": "seen_embedded", "C": "never_embedded"}


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def pranks_for_file(f, pairs, layers):
    """Per-pair percentile rank of the translation for one checkpoint file."""
    z = np.load(f, allow_pickle=True)
    widx = {w: i for i, w in enumerate(z["words"])}
    cnt = z["counts"]
    U = _unit(z["vecs"][:, layers, :].mean(1))
    arm, stage, epoch = str(z["arm"]), str(z["stage"]), int(z["epoch"])
    out = []
    for (sl, tl), g in pairs.groupby(["src_lang", "tgt_lang"]):
        ok = g[[all(k in widx and cnt[widx[k]] > 0 for k in
                    (f"{sl}:{r.E_word}", f"{sl}:{r.C_word}", f"{tl}:{r.equiv_E}", f"{tl}:{r.equiv_C}"))
                for r in g.itertuples(index=False)]]
        pool = sorted({f"{tl}:{w}" for w in set(ok.equiv_E) | set(ok.equiv_C)})
        tidx = {k: i for i, k in enumerate(pool)}
        T = U[[widx[k] for k in pool]]
        for cond, wc, ec in (("E", "E_word", "equiv_E"), ("C", "C_word", "equiv_C")):
            for pid, w, e in zip(ok.pair_id, ok[wc], ok[ec]):
                kw, ke = f"{sl}:{w}", f"{tl}:{e}"
                sc = T @ U[widx[kw]]
                rank = int((sc > sc[tidx[ke]]).sum()) + 1
                out.append((arm, stage, epoch, pid, cond, f"{sl}-{tl}",
                            rank, 1 - (rank - 1) / (len(pool) - 1)))
    return pd.DataFrame(out, columns=["arm", "stage", "epoch", "pair_id", "condition",
                                      "direction", "rank", "prank"])


def boot_median_diff(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(a.shape[0])
    diffs = np.empty((n, a.shape[1]))
    for k in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        diffs[k] = np.median(a[s], axis=0) - np.median(b[s], axis=0)
    lo, hi = np.percentile(diffs, [2.5, 97.5], axis=0)
    return np.median(a, axis=0) - np.median(b, axis=0), lo, hi


def _rows_for(pr, ck, cfg, seed, subset, keep, directions):
    out = []
    for cond, gname in GROUP.items():
        for dname, ids in directions.items():
            sel = pr[pr.condition == cond]
            if ids is not None:
                sel = sel[sel.pair_id.isin(ids)]
            if keep is not None:
                sel = sel[sel.pair_id.isin(keep)]
            piv = (sel.pivot_table(index="pair_id", columns=["arm", "stage", "epoch"], values="prank")
                      .reindex(columns=[(a, s, e) for a in ("cs", "noswitch") for s, e in ck])
                      .dropna())
            if len(piv) < 20:
                continue
            a = piv[[("cs", s, e) for s, e in ck]].values
            b = piv[[("noswitch", s, e) for s, e in ck]].values
            m, lo, hi = boot_median_diff(a, b, cfg["bootstrap_n"], cfg["bootstrap_seed"])
            for i, (st, ep) in enumerate(ck):
                out.append({"seed": f"s{seed}", "subset": subset, "group": gname,
                            "direction": dname, "stage": st, "epoch": ep,
                            "delta": m[i], "ci_lo": lo[i], "ci_hi": hi[i],
                            "median_cs": np.median(a[:, i]), "median_noswitch": np.median(b[:, i]),
                            "mean_delta": a[:, i].mean() - b[:, i].mean(), "n": len(piv)})
    return out


def balanced_pair_ids(pairs, exposure):
    """Pairs whose BOTH translation equivalents were ever embedded."""
    emb = set(zip(exposure.loc[exposure.ever_embedded, "word"], exposure.loc[exposure.ever_embedded, "lang"]))
    tE = np.array([t in emb for t in zip(pairs.equiv_E, pairs.tgt_lang)])
    tC = np.array([t in emb for t in zip(pairs.equiv_C, pairs.tgt_lang)])
    return set(pairs.loc[tE & tC, "pair_id"])


def exposure_rank(cfg, pairs_path=None, exposure_path=None) -> pd.DataFrame:
    """Score every ``reps_*.npz`` under ``out_dir`` for one seed -> tidy DataFrame."""
    d = Path(cfg["out_dir"])
    pairs = pd.read_parquet(pairs_path or d / "measurement_pairs_mono_freq.parquet")
    exposure = pd.read_parquet(exposure_path or cfg["data_dir"] / "align" / "exposure_sets.parquet")
    files = sorted(d.glob("reps_*.npz"))
    if not files:
        raise FileNotFoundError(f"no reps_*.npz under {d}; run extraction first")
    pr = pd.concat([pranks_for_file(f, pairs, list(cfg["layer_aggregate"])) for f in files],
                   ignore_index=True)
    ck = [c for c in CKPT_ORDER if ((pr.stage == c[0]) & (pr.epoch == c[1])).any()]
    dirs = {"pooled": None}
    for (sl, tl), g in pairs.groupby(["src_lang", "tgt_lang"]):
        dirs[f"{sl}-{tl}"] = set(g.pair_id)
    bal = balanced_pair_ids(pairs, exposure)
    rows = _rows_for(pr, ck, cfg, cfg["seed"], "all", None, dirs)
    rows += _rows_for(pr, ck, cfg, cfg["seed"], "balanced", bal, dirs)
    print(f"[exposure_rank] s{cfg['seed']}: {len(files)} checkpoint files, {pr.pair_id.nunique()} pairs")
    return pd.DataFrame(rows)
