"""Static (``wte``) word-level cross-lingual alignment measurement.

Ported from ``align_static.py``. For each model checkpoint, over single-token
translation pairs, computes:

  1. Per-pair cosine in three variants (anisotropy controls):
     ``raw`` / ``centered`` (mean-removed) / ``pc2`` (top-2-PC removed), where
     mean and PCs are estimated on the frozen BLI candidate vocabulary (same
     estimation vocab for every model, so variants are comparable across models).
  2. A per-pair frequency-matched null: ``K_NULL`` random cross-lingual pairs
     drawn from the same log2-frequency bins *in the model's own training
     corpus* (``freq_intra`` for code-switched models, ``freq_noswitch`` for
     unilingual models — the corpora differ in word frequencies precisely
     because of switching, and the null must absorb that). The reported
     probability-of-superiority is ``pct_<variant>`` = P(pair cosine > matched
     random pair) over the K samples; the figures average ``pct_raw`` per tier.
  3. (``bli``, optional) CSLS(k=10) / raw-NN retrieval over the frozen
     candidate lists.

The checkpoint set is passed in as ``model_labels`` — a list of
``(run, freq_cond, label, ckpt_dir)`` tuples — instead of the old baked
``MANIFEST``; callers derive ``ckpt_dir`` from ``common.model_dir`` +
``common.final_ckpt`` (final checkpoints) or per-checkpoint stage dirs
(trajectory).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open

LANGS = ("eng", "nld", "zho")
K_NULL = 100
CSLS_K = 10

# safetensors keys that may hold the input embedding matrix, in order of
# preference (GPT-2 uses transformer.wte.weight).
_WTE_KEYS = (
    "transformer.wte.weight",
    "wte.weight",
    "model.embed_tokens.weight",
    "transformer.word_embeddings.weight",
)


def load_wte(ckpt_dir) -> np.ndarray:
    """Return the static input-embedding matrix (float32) from a checkpoint dir.

    ``ckpt_dir`` is a directory holding ``model.safetensors`` (a resolved HF
    checkpoint — pass ``common.final_ckpt(model_dir(...))``).
    """
    path = Path(ckpt_dir)
    st = path / "model.safetensors" if path.is_dir() else path
    with safe_open(str(st), framework="numpy") as f:
        keys = set(f.keys())
        key = next((k for k in _WTE_KEYS if k in keys), None)
        if key is None:  # fall back: any key ending in the embedding suffix
            key = next((k for k in keys if k.endswith("wte.weight")
                        or k.endswith("embed_tokens.weight")), None)
        if key is None:
            raise KeyError(f"no input-embedding tensor in {st} (keys: {sorted(keys)[:8]}...)")
        return f.get_tensor(key).astype(np.float32)


def unit(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def remove_top_pcs(vecs: np.ndarray, basis_from: np.ndarray, n_pc=2):
    """All-but-the-top: subtract mean + top-``n_pc`` PCs (estimated on
    ``basis_from``, the frozen candidate cloud) from ``vecs``."""
    mu = basis_from.mean(0)
    Xc = basis_from - mu
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = vt[:n_pc]
    out = vecs - mu
    return out - (out @ pcs.T) @ pcs


def _log2bin(v):
    return int(np.log2(max(v, 1)))


def _build_cand_bins(cands, freq_maps):
    """Pre-bin candidate token ids per (lang, freq-condition) by log2 frequency."""
    cand_bins = {}
    for lang in LANGS:
        for cond in ("intra", "noswitch"):
            fm = freq_maps[cond]
            bins = {}
            for w, tid in zip(cands[lang]["word"], cands[lang]["token_id"]):
                bins.setdefault(_log2bin(fm.get(w, 0)), []).append(tid)
            cand_bins[(lang, cond)] = {b: np.array(v) for b, v in bins.items()}
    return cand_bins


def _sample_null_ids(words, langs_, cond, cand_bins, freq_maps, rng, k_null=K_NULL):
    fm = freq_maps[cond]
    out = np.empty((len(words), k_null), dtype=np.int64)
    for i, (w, lg) in enumerate(zip(words, langs_)):
        bins = cand_bins[(lg, cond)]
        b = _log2bin(fm.get(w, 0))
        pool = None
        for db in (0, -1, 1, -2, 2, -3, 3):
            pool = bins.get(b + db)
            if pool is not None and len(pool) >= 5:
                break
        if pool is None:
            pool = np.concatenate(list(bins.values()))
        out[i] = rng.choice(pool, size=k_null, replace=True)
    return out


def _freq_maps(corpus_freq: pd.DataFrame):
    return {
        "intra": dict(zip(corpus_freq["word"], corpus_freq["freq_intra"])),
        "noswitch": dict(zip(corpus_freq["word"], corpus_freq["freq_noswitch"])),
    }


def wordlevel_pairs(model_labels, lex, cands, corpus_freq, seed=0, k_null=K_NULL):
    """Per-pair, per-checkpoint alignment measure -> long DataFrame.

    Parameters
    ----------
    model_labels : list of (run, freq_cond, label, ckpt_dir)
        ``freq_cond`` is ``"intra"`` (code-switched models) or ``"noswitch"``
        (unilingual models); ``ckpt_dir`` holds ``model.safetensors``.
    lex : DataFrame
        the evaluation lexicon; only single-token pairs are used
        (``src_single & tgt_single``). Must carry ``src_token_id``,
        ``tgt_token_id``, ``src_lang``, ``tgt_lang``, ``src_word``, ``tgt_word``,
        ``tier``, ``source``.
    cands : {lang: DataFrame}   with columns ``word``, ``token_id``.
    corpus_freq : DataFrame     with ``word``, ``freq_intra``, ``freq_noswitch``.

    Returns a DataFrame with one row per (checkpoint, pair): the input columns
    plus ``cos_{raw,centered,pc2}``, ``null_{...}``, ``null_std_{...}``,
    ``pct_{...}``, ``run``, ``ckpt``. ``pct_raw`` is the probability-of-
    superiority the figures average.
    """
    lex = lex[lex["src_single"] & lex["tgt_single"]].reset_index(drop=True)
    rng = np.random.default_rng(seed)

    cand_ids_all = np.unique(np.concatenate([c["token_id"].values for c in cands.values()]))
    freq_maps = _freq_maps(corpus_freq)
    cand_bins = _build_cand_bins(cands, freq_maps)

    # fixed null samples shared across all checkpoints (same pairs, same nulls
    # per freq-condition -> checkpoint differences are purely weight differences)
    null_src = {c: _sample_null_ids(lex["src_word"], lex["src_lang"], c,
                                    cand_bins, freq_maps, rng, k_null)
                for c in ("intra", "noswitch")}
    null_tgt = {c: _sample_null_ids(lex["tgt_word"], lex["tgt_lang"], c,
                                    cand_bins, freq_maps, rng, k_null)
                for c in ("intra", "noswitch")}

    sids = lex["src_token_id"].values
    tids = lex["tgt_token_id"].values
    frames = []
    for run, cond, label, ckpt in model_labels:
        wte = load_wte(ckpt)
        basis = wte[cand_ids_all]
        variants = {"raw": wte, "centered": wte - basis.mean(0),
                    "pc2": remove_top_pcs(wte, basis, 2)}
        rec = {
            "src_lang": lex["src_lang"], "tgt_lang": lex["tgt_lang"],
            "src_word": lex["src_word"], "tgt_word": lex["tgt_word"],
            "source": lex["source"], "tier": lex["tier"],
        }
        for col in ("exposure_identical", "ortho_jaccard"):
            if col in lex.columns:
                rec[col] = lex[col]
        for vname, W in variants.items():
            U = unit(W)
            cos_v = np.einsum("ij,ij->i", U[sids], U[tids])
            rec[f"cos_{vname}"] = cos_v
            ns_, nt_ = null_src[cond], null_tgt[cond]
            null_mean = np.empty(len(lex), dtype=np.float32)
            null_std = np.empty(len(lex), dtype=np.float32)
            null_pct = np.empty(len(lex), dtype=np.float32)
            for lo in range(0, len(lex), 1000):  # chunked: full gather is ~12GB
                hi = min(lo + 1000, len(lex))
                nc = np.einsum("ikj,ikj->ik", U[ns_[lo:hi]], U[nt_[lo:hi]])
                null_mean[lo:hi] = nc.mean(1)
                null_std[lo:hi] = nc.std(1)
                null_pct[lo:hi] = (nc < cos_v[lo:hi, None]).mean(1)
            rec[f"null_{vname}"] = null_mean
            rec[f"null_std_{vname}"] = null_std
            rec[f"pct_{vname}"] = null_pct
        df = pd.DataFrame(rec)
        df["run"], df["ckpt"] = run, label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def bli(model_labels, lex, cands, variant="centered"):
    """CSLS(k=10) / raw-NN retrieval per direction, per checkpoint.

    Returns a DataFrame with columns ``run, ckpt, direction, method,
    n_queries, p1, p5, mrr, med_rank, n_candidates``. Not used by Figs 4/6;
    provided for parity with ``align_static.py`` and reuse by the dose-response
    measurement.
    """
    lex = lex[lex["src_single"] & lex["tgt_single"]].reset_index(drop=True)
    cand_ids_all = np.unique(np.concatenate([c["token_id"].values for c in cands.values()]))
    rows = []
    for run, _cond, label, ckpt in model_labels:
        wte = load_wte(ckpt)
        basis = wte[cand_ids_all]
        W = {"raw": wte, "centered": wte - basis.mean(0),
             "pc2": remove_top_pcs(wte, basis, 2)}[variant]
        U = unit(W)
        for (sl, tl), sub in lex.groupby(["src_lang", "tgt_lang"]):
            cand_t = cands[tl]
            cid = cand_t["token_id"].values
            cword_idx = {w: i for i, w in enumerate(cand_t["word"])}
            gold = {}
            for s, t in zip(sub["src_word"], sub["tgt_word"]):
                if t in cword_idx:
                    gold.setdefault(s, set()).add(cword_idx[t])
            if not gold:
                continue
            qwords = sorted(gold)
            qids = {w: i for w, i in zip(sub["src_word"], sub["src_token_id"])}
            Q = U[[qids[w] for w in qwords]]
            Cm = U[cid]
            sim = Q @ Cm.T
            src_cloud = U[cands[sl]["token_id"].values]
            r_t = np.sort(Cm @ src_cloud.T, axis=1)[:, -CSLS_K:].mean(1)
            r_s = np.sort(sim, axis=1)[:, -CSLS_K:].mean(1)
            csls = 2 * sim - r_t[None, :] - r_s[:, None]
            for name, S in (("csls", csls), ("nn", sim)):
                order = np.argsort(-S, axis=1)
                p1 = p5 = mrr = 0.0
                best_ranks = []
                for i, w in enumerate(qwords):
                    g = gold[w]
                    ranks = [r for r, c in enumerate(order[i]) if c in g]
                    best = min(ranks) if ranks else None
                    if best is not None:
                        best_ranks.append(best + 1)
                        p1 += best == 0
                        p5 += best < 5
                        mrr += 1 / (best + 1)
                nq = len(qwords)
                rows.append({
                    "run": run, "ckpt": label, "direction": f"{sl}->{tl}",
                    "method": name, "n_queries": nq,
                    "p1": round(p1 / nq, 4), "p5": round(p5 / nq, 4),
                    "mrr": round(mrr / nq, 4),
                    "med_rank": float(np.median(best_ranks)) if best_ranks else None,
                    "n_candidates": len(cid),
                })
    return pd.DataFrame(rows)
