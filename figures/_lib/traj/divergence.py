"""Cross-model divergence by layer on the matched word pairs (Fig 8).

Port of the paper's ``traj_divergence.py``. For one seed, load the final-checkpoint
reps of the cs and noswitch models, take each pair's E_word (embedded) and C_word
(never-embedded control) in the source language, and compute per layer the cosine
between the two models' representations of the same word. Reports per language
and layer the mean cosine for both groups, Cohen's d, a paired t-test p, and the
same on L2 distance.

Output schema (== results/word_divergence.csv, minus the seed column added by the
caller): lang, condition, layer, n_pairs, emb_cos, ctl_cos, cohen_d_cos, p_cos,
emb_l2, ctl_l2, cohen_d_l2, p_l2.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CKPT = "stage3_ep10"


def cohen_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / (sp + 1e-12))


def _load(seed_dir, arm):
    z = np.load(seed_dir / f"reps_{arm}_{CKPT}.npz", allow_pickle=True)
    assert str(z["arm"]) == arm
    return {w: i for i, w in enumerate(z["words"])}, z["vecs"], z["counts"], str(z["set_hash"])


def divergence_layerwise(cfg, pairs_path=None) -> pd.DataFrame:
    d = Path(cfg["out_dir"])
    pairs = pd.read_parquet(pairs_path or d / "measurement_pairs_mono_freq.parquet")
    idx_cs, V_cs, cnt_cs, h_cs = _load(d, "cs")
    idx_ns, V_ns, cnt_ns, h_ns = _load(d, "noswitch")
    assert h_cs == h_ns, "cs and noswitch reps built from different context sets"
    assert idx_cs == idx_ns

    def key(w, lang):
        return f"{lang}:{w}"

    keep = []
    for r in pairs.itertuples(index=False):
        ke, kc = key(r.E_word, r.src_lang), key(r.C_word, r.src_lang)
        ks = (ke, kc, key(r.equiv_E, r.tgt_lang), key(r.equiv_C, r.tgt_lang))
        if all(k in idx_cs and cnt_cs[idx_cs[k]] > 0 and cnt_ns[idx_ns[k]] > 0 for k in ks):
            keep.append((r.src_lang, idx_cs[ke], idx_cs[kc]))
    print(f"[divergence] s{cfg['seed']}: {len(keep)}/{len(pairs)} pairs with contexts for both members")

    rows = []
    for lang in ("eng", "nld", "zho"):
        ie = np.array([e for l, e, c in keep if l == lang])
        ic = np.array([c for l, e, c in keep if l == lang])
        for L in range(V_cs.shape[1]):
            A_cs, A_ns = V_cs[:, L, :], V_ns[:, L, :]
            cos = np.einsum("ij,ij->i", A_cs, A_ns) / (
                np.linalg.norm(A_cs, axis=1) * np.linalg.norm(A_ns, axis=1) + 1e-12)
            l2 = np.linalg.norm(A_cs - A_ns, axis=1)
            ce, cc, le, lc = cos[ie], cos[ic], l2[ie], l2[ic]
            rows.append(dict(
                lang=lang, condition="curriculum", layer=L, n_pairs=len(ie),
                emb_cos=ce.mean(), ctl_cos=cc.mean(), cohen_d_cos=cohen_d(ce, cc),
                p_cos=stats.ttest_rel(ce, cc).pvalue,
                emb_l2=le.mean(), ctl_l2=lc.mean(), cohen_d_l2=cohen_d(le, lc),
                p_l2=stats.ttest_rel(le, lc).pvalue))
    return pd.DataFrame(rows)
