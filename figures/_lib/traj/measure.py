"""Per-(pair, layer) metrics from the reps (ports traj_measure.py).

For every matched pair and every layer:
  s_l            = cosine(r_l(w), r_l(w_bar))                     [raw]
  mu_l           = mean cosine over ALL non-equivalent cross-lingual pairs
                   (exhaustive null, re-estimated at every checkpoint)
  cla_l          = (s_l - mu_l) / (1 - mu_l)
  pct_l          = percentile rank of s_l in the null distribution [PRIMARY]
  z_l            = (s_l - mu_l) / std(null)
  cos_centered_l = cosine after subtracting the per-layer mean rep

pct is primary: in the paired difference mu cancels, and anisotropy (mu) grows
over training, so percentile rank (monotone-invariant) is immune to the drift
that could otherwise manufacture a trajectory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def measure(cfg, pairs_path=None) -> Path:
    """Turn reps_*.npz + matched pairs into measurements.csv; return its path."""
    out = cfg["out_dir"] / "measurements.csv"
    if out.exists():
        return out

    pairs = pd.read_parquet(
        pairs_path or cfg["out_dir"] / "measurement_pairs_mono_freq.parquet")

    rows = []
    rep_files = sorted(cfg["out_dir"].glob("reps_*.npz"))
    if not rep_files:
        raise SystemExit("no reps_*.npz found -- run extract_reps first")
    print(f"[measure] found {len(rep_files)} checkpoint rep files")

    for f in rep_files:
        z = np.load(f, allow_pickle=True)
        words = list(z["words"]); vecs = z["vecs"]; cnt = z["counts"]
        arm, stage, epoch = str(z["arm"]), str(z["stage"]), int(z["epoch"])
        step = int(z["step"]) if "step" in z.files else 0
        widx = {w: i for i, w in enumerate(words)}
        n_layers = vecs.shape[1]

        def key(w, l):
            return f"{l}:{w}"

        ok = []
        for r in pairs.itertuples(index=False):
            ks = [key(r.E_word, r.src_lang), key(r.C_word, r.src_lang),
                  key(r.equiv_E, r.tgt_lang), key(r.equiv_C, r.tgt_lang)]
            if all(k in widx and cnt[widx[k]] > 0 for k in ks):
                ok.append((r, ks))
        if not ok:
            raise SystemExit("no usable pairs")

        for L in range(n_layers):
            V = vecs[:, L, :]
            U = _unit(V)
            Uc = _unit(V - V.mean(0, keepdims=True))   # centered variant

            null_vals = []
            for (sl, tl), g in pairs.groupby(["src_lang", "tgt_lang"]):
                si = [widx[key(w, sl)] for w in set(g.E_word) | set(g.C_word)
                      if key(w, sl) in widx]
                ti = [widx[key(w, tl)] for w in set(g.equiv_E) | set(g.equiv_C)
                      if key(w, tl) in widx]
                if not si or not ti:
                    continue
                S = U[si] @ U[ti].T
                eq = {(widx[key(r.E_word, r.src_lang)], widx[key(r.equiv_E, r.tgt_lang)])
                      for r in g.itertuples(index=False)
                      if key(r.E_word, r.src_lang) in widx and key(r.equiv_E, r.tgt_lang) in widx}
                eq |= {(widx[key(r.C_word, r.src_lang)], widx[key(r.equiv_C, r.tgt_lang)])
                       for r in g.itertuples(index=False)
                       if key(r.C_word, r.src_lang) in widx and key(r.equiv_C, r.tgt_lang) in widx}
                sset = {v: i for i, v in enumerate(si)}; tset = {v: i for i, v in enumerate(ti)}
                mask = np.ones_like(S, dtype=bool)
                for a, b in eq:
                    if a in sset and b in tset:
                        mask[sset[a], tset[b]] = False
                null_vals.append(S[mask].ravel())
            null = np.concatenate(null_vals)
            mu = float(null.mean())
            sd = float(null.std())
            null_sorted = np.sort(null)

            for r, ks in ok:
                for cond, wk, ek in (("E", ks[0], ks[2]), ("C", ks[1], ks[3])):
                    s = float(U[widx[wk]] @ U[widx[ek]])
                    sc = float(Uc[widx[wk]] @ Uc[widx[ek]])
                    pct = float(np.searchsorted(null_sorted, s) / len(null_sorted))
                    rows.append({
                        "arm": arm, "stage": stage, "epoch": epoch, "step": step,
                        "layer": L, "pair_id": r.pair_id, "condition": cond,
                        "word": wk, "equiv": ek,
                        "src_lang": r.src_lang, "tgt_lang": r.tgt_lang,
                        "pos": r.pos, "n_tok_bpe": r.n_tok_bpe,
                        "raw_sim": s, "cos_centered": sc, "pct": pct,
                        "cla": (s - mu) / (1 - mu), "s_minus_mu": s - mu,
                        "z": (s - mu) / sd, "null_std": sd,
                        "mu": mu, "mu_high_flag": mu > 0.9,
                        "ortho": r.E_ortho if cond == "E" else r.C_ortho,
                        "freq": r.E_freq if cond == "E" else r.C_freq,
                        "freq_intra": r.E_freq_intra if cond == "E" else r.C_freq_intra,
                        "freq_noswitch": r.E_freq_noswitch if cond == "E" else r.C_freq_noswitch,
                    })
        print(f"[measure]  {f.name}: {len(ok)} pairs x {n_layers} layers "
              f"(mu@L{n_layers-1}={mu:.3f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"[measure] wrote {len(df)} rows -> {out}")
    return out
