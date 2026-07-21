"""DiD estimands + cluster bootstrap (ports traj_stats.py).

Estimands (upper-layer aggregate = mean over cfg['layer_aggregate']):
  Delta_arm(t)   = mean_pairs [ m_E - m_C ]        within each arm
  DD(t)          = Delta_cs(t) - Delta_noswitch(t) <- HEADLINE (matched-control DiD)
  Delta_E(t)     = mean_E [ m_E,cs - m_E,noswitch ] (same word, cross-arm; primary)
Reported for m in {pct (primary), raw_sim, cla, cos_centered}. CIs from a cluster
bootstrap that resamples whole PAIR trajectories (all checkpoints together).

Output schema: metric, estimand, stage, epoch, mean, ci_lo, ci_hi, n_pairs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CKPT_ORDER = [("stage1", 0), ("stage1", 1), ("stage1", 5), ("stage1", 10),
              ("stage2", 1), ("stage2", 5), ("stage2", 10),
              ("stage3", 1), ("stage3", 5), ("stage3", 10)]
METRICS = ["pct", "raw_sim", "cla", "cos_centered"]


def _agg_upper(df, layers):
    """Collapse layers -> per (arm,stage,epoch,pair_id,condition) upper-layer mean."""
    d = df[df.layer.isin(layers)]
    return (d.groupby(["arm", "stage", "epoch", "pair_id", "condition"])
             [METRICS].mean().reset_index())


def _paired_delta(up, metric):
    piv = up.pivot_table(index=["arm", "stage", "epoch", "pair_id"],
                         columns="condition", values=metric).reset_index()
    piv["delta"] = piv["E"] - piv["C"]
    return piv


def _boot_ci(mat, n, seed):
    """mat: (n_pairs, n_ckpts) of per-pair deltas. Bootstrap over pairs (rows)."""
    rng = np.random.default_rng(seed)
    means = np.empty((n, mat.shape[1]))
    idx = np.arange(mat.shape[0])
    for b in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        means[b] = np.nanmean(mat[s], axis=0)
    lo, hi = np.nanpercentile(means, [2.5, 97.5], axis=0)
    return np.nanmean(mat, axis=0), lo, hi


def stats_trajectory(cfg, measurements_path=None) -> Path:
    """Compute the trajectory stats CSV under out_dir; return its path."""
    out = cfg["out_dir"] / "stats_trajectory.csv"
    if out.exists():
        return out

    df = pd.read_csv(measurements_path or cfg["out_dir"] / "measurements.csv")
    layers = cfg["layer_aggregate"]
    up = _agg_upper(df, layers)

    ck = [c for c in CKPT_ORDER if ((up.stage == c[0]) & (up.epoch == c[1])).any()]
    results = []

    for metric in METRICS:
        piv = _paired_delta(up, metric)
        arm_mats = {}
        for arm in ("cs", "noswitch"):
            sub = piv[piv.arm == arm]
            w = sub.pivot_table(index="pair_id", columns=["stage", "epoch"], values="delta")
            w = w.reindex(columns=ck)
            arm_mats[arm] = w
        common = arm_mats["cs"].index.intersection(arm_mats["noswitch"].index)
        cs = arm_mats["cs"].loc[common].values
        ns = arm_mats["noswitch"].loc[common].values
        DD = cs - ns
        # within-word cross-arm delta on the E words (needs no control set)
        Epiv = (up[up.condition == "E"]
                .pivot_table(index="pair_id", columns=["arm", "stage", "epoch"],
                             values=metric).reindex(
                    columns=[(a, s, e) for a in ("cs", "noswitch") for s, e in ck]))
        Epiv = Epiv.dropna()
        dE = (Epiv[[("cs", s, e) for s, e in ck]].values
              - Epiv[[("noswitch", s, e) for s, e in ck]].values)
        for name, mat, npairs in [("Delta_E", dE, len(Epiv)),
                                  ("Delta_cs", cs, len(common)),
                                  ("Delta_noswitch", ns, len(common)),
                                  ("DD", DD, len(common))]:
            m, lo, hi = _boot_ci(mat, cfg["bootstrap_n"], cfg["bootstrap_seed"])
            for i, (st, ep) in enumerate(ck):
                results.append({"metric": metric, "estimand": name, "stage": st,
                                "epoch": ep, "mean": m[i], "ci_lo": lo[i],
                                "ci_hi": hi[i], "n_pairs": npairs})

    res = pd.DataFrame(results)
    res.to_csv(out, index=False)

    anchor = res[(res.metric == "pct") & (res.estimand == "DD")
                 & (res.stage == "stage1") & (res.epoch == 0)]
    if len(anchor):
        print(f"[stats] DD(init) pct = {anchor['mean'].iloc[0]:+.5f} (should be ~0)")
    print(f"[stats] wrote {out}")
    return out
