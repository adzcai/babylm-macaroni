"""Sentence-level cross-lingual alignment: FLORES+ bitext retrieval (Figs 2, 3, 6, 7).

Port of the paper's ``scripts/analysis/represent_align.py`` (the single scoring
site for every retrieval number in the paper). Inference only. For each model
it mean-pools per-layer hidden states on the FLORES+ dev set (eng/nld/zho, 997
sentences aligned by id, held out) and computes per layer and language pair:

  retrieval_p1     bitext retrieval P@1 under CSLS (k=10) ranking, both directions averaged  [headline]
  retrieval_p1_nn  the same under plain cosine nearest neighbour (hubness check)
  median_pct_rank, mean_pct_rank, mrr, retrieval_p5, median_rank   rank companions (CSLS)
  cos_gap, cka, centroid_dist                                       corroborating measures

FLORES+ (``openlanguagedata/flores_plus``) is a GATED Hub dataset: accept its terms
and set ``HF_TOKEN`` (or ``huggingface-cli login``) before the first run. Pooled
hidden states can be cached per model label (``emb_cache``) so re-scoring is
CPU-only.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

import common

PAIRS = [("eng", "nld"), ("eng", "zho"), ("nld", "zho")]
FLORES_PLUS_CFG = {"eng": "eng_Latn", "nld": "nld_Latn", "zho": "cmn_Hans"}
CSLS_K = 10
FIELDS = ["model", "group", "seed", "layer", "pair",
          "retrieval_p1", "retrieval_p1_nn", "cos_gap", "cka", "centroid_dist",
          "median_pct_rank", "mean_pct_rank", "mrr", "retrieval_p5", "median_rank"]


def load_flores_plus(n=997, split="dev"):
    """FLORES+ ``split`` for eng/nld/zho, aligned by sentence id -> (texts, n)."""
    from datasets import load_dataset
    per = {}
    for lang, cfg in FLORES_PLUS_CFG.items():
        d = load_dataset("openlanguagedata/flores_plus", cfg, split=split)
        per[lang] = {row["id"]: row["text"] for row in d}
    ids = sorted(set.intersection(*[set(p.keys()) for p in per.values()]))
    if n:
        ids = ids[:n]
    return {lang: [per[lang][i] for i in ids] for lang in per}, len(ids)


@torch.no_grad()
def embed(model, tok, sents, device, batch=32):
    """Return ``[n_layers+1, N, hid]`` attention-masked mean-pooled hidden states."""
    per_layer = None
    for i in range(0, len(sents), batch):
        chunk = sents[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=128).to(device)
        out = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        denom = mask.sum(1).clamp(min=1)
        pooled = [(h * mask).sum(1) / denom for h in out.hidden_states]
        pooled = torch.stack(pooled, 0).float().cpu().numpy()
        per_layer = pooled if per_layer is None else np.concatenate([per_layer, pooled], axis=1)
    return per_layer


def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    xx = np.linalg.norm(X.T @ X, "fro")
    yy = np.linalg.norm(Y.T @ Y, "fro")
    return float(xy / (xx * yy + 1e-12))


def _gold_ranks(S):
    gold = np.diag(S)[:, None]
    return (S > gold).sum(1) + 1


def rank_metrics(S):
    """Rank-based retrieval stats over both directions (pct_rank 1.0 = gold first, 0.5 = chance)."""
    n = S.shape[0]
    r = np.concatenate([_gold_ranks(S), _gold_ranks(S.T)])
    pct = 1.0 - (r - 1) / (n - 1)
    return {"median_pct_rank": float(np.median(pct)),
            "mean_pct_rank": float(pct.mean()),
            "mrr": float((1.0 / r).mean()),
            "retrieval_p5": float((r <= 5).mean()),
            "median_rank": float(np.median(r))}


def csls(S, k=CSLS_K):
    """Cross-domain similarity local scaling of a cosine matrix (Conneau et al. 2018)."""
    if k <= 0:
        return S
    k = min(k, S.shape[0], S.shape[1])
    r_t = np.sort(S, axis=1)[:, -k:].mean(1)
    r_s = np.sort(S, axis=0)[-k:, :].mean(0)
    return 2.0 * S - r_t[:, None] - r_s[None, :]


def p_at_1(S):
    n = S.shape[0]
    return float(0.5 * ((S.argmax(1) == np.arange(n)).mean() + (S.argmax(0) == np.arange(n)).mean()))


def pair_metrics(E1, E2, rng, k=CSLS_K):
    """``E1,E2``: ``[N, hid]`` for one layer -> dict of every metric in FIELDS."""
    n = E1.shape[0]
    N1 = E1 / (np.linalg.norm(E1, axis=1, keepdims=True) + 1e-12)
    N2 = E2 / (np.linalg.norm(E2, axis=1, keepdims=True) + 1e-12)
    S = N1 @ N2.T
    C = csls(S, k)
    parallel = np.diag(S).mean()
    perm = rng.permutation(n)
    while np.any(perm == np.arange(n)):
        perm = rng.permutation(n)
    rand = S[np.arange(n), perm].mean()
    out = {"retrieval_p1": p_at_1(C), "retrieval_p1_nn": p_at_1(S),
           "cos_gap": float(parallel - rand), "cka": linear_cka(E1, E2),
           "centroid_dist": float(np.linalg.norm(N1.mean(0) - N2.mean(0)))}
    out.update(rank_metrics(C))
    return out


def group_seed(label):
    """Paper label convention: group = switched / noswitch; seed from ``-sNN``.

    Handles final labels (``curriculum-s43``) and trajectory labels
    (``cur_noswitch-s43-s2e5``).
    """
    m = re.search(r"-s(\d+)(?:-s\d+e\d+)?$", label)
    seed = int(m.group(1)) if m else 42
    return ("noswitch" if common.is_noswitch(label) else "switched"), seed


def embed_cached(label, path, texts, device, cache_dir=None, batch=32):
    """lang -> [L+1, N, hid]; served from ``cache_dir/<label>/<lang>.npy`` if present."""
    if cache_dir is not None:
        d = Path(cache_dir) / label
        files = {lang: d / f"{lang}.npy" for lang in texts}
        if all(f.exists() for f in files.values()):
            emb = {lang: np.load(f) for lang, f in files.items()}
            if all(e.shape[1] == len(texts[lang]) for lang, e in emb.items()):
                print(f"[{label}] embeddings from cache {d}", flush=True)
                return emb
    print(f"[{label}] loading {path}", flush=True)
    tok = common.load_tokenizer(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = common.load_model(path, device=device, dtype=torch.float32)
    emb = {lang: embed(model, tok, texts[lang], device, batch=batch) for lang in texts}
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    if cache_dir is not None:
        d = Path(cache_dir) / label
        d.mkdir(parents=True, exist_ok=True)
        for lang, e in emb.items():
            tmp = d / f"{lang}.npy.tmp"
            with open(tmp, "wb") as fh:
                np.save(fh, e.astype(np.float32))
            tmp.rename(d / f"{lang}.npy")
    return emb


def retrieval_per_layer(model_labels, device="cuda", flores_cache=None, n_sents=997,
                        csls_k=CSLS_K, emb_cache=None, rng_seed=0):
    """Score ``{label: model_dir}`` -> long-format DataFrame with columns FIELDS."""
    import pandas as pd
    texts, n = flores_cache if flores_cache is not None else load_flores_plus(n_sents)
    print(f"probe: FLORES+ dev, {n} aligned sentences x {list(texts)}; CSLS k={csls_k}", flush=True)
    rng = np.random.default_rng(rng_seed)
    rows = []
    for label, path in model_labels.items():
        grp, seed = group_seed(label)
        emb = embed_cached(label, path, texts, device, emb_cache)
        for layer in range(emb["eng"].shape[0]):
            for a, b in PAIRS:
                mt = pair_metrics(emb[a][layer], emb[b][layer], rng, k=csls_k)
                rows.append({"model": label, "group": grp, "seed": seed,
                             "layer": layer, "pair": f"{a}-{b}", **mt})
    return pd.DataFrame(rows, columns=FIELDS)
