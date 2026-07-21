"""Cross-lingual representation-alignment compute (paper RQ2 / Figs 2, 3, 5).

Ported from the paper's ``scripts/analysis/represent_align.py``. Inference only
(no training). For each model checkpoint it mean-pools per-layer hidden states on
a held-out parallel probe set (FLORES+ dev, eng/nld/zho, 997 aligned sentences the
models never saw as pairs) and computes, per layer and language pair:

  - retrieval_p1  : bitext retrieval P@1 (nearest-neighbour translation, avg of
                    both directions) -- the headline metric for Figs 2/3/5;
  - cos_gap       : mean cos(parallel pair) - mean cos(random cross-lingual pair);
  - cka           : linear CKA between the two languages' representation matrices;
  - centroid_dist : || mean(E_L1) - mean(E_L2) ||  on L2-normalised embeddings.

FLORES+ TOKEN REQUIREMENT
-------------------------
The probe set is loaded from the HuggingFace Hub dataset
``openlanguagedata/flores_plus``, which is GATED: you must (a) accept its terms on
the Hub with your account, and (b) expose a token so ``datasets`` can authenticate.
Set ``HF_TOKEN`` in the environment (or run ``huggingface-cli login``) before
calling ``retrieval_per_layer``. To run fully offline against a pre-downloaded
cache, set ``HF_DATASETS_OFFLINE=1`` (and ``HF_HOME`` to the cache root), or pass a
pre-loaded probe set via the ``flores_cache`` argument (see below) so no network
access is attempted at all.

PUBLIC API
----------
``retrieval_per_layer(model_labels, device="cuda", flores_cache=None, ...)`` ->
``pandas.DataFrame`` with columns
``model, group, seed, layer, pair, retrieval_p1, cos_gap, cka, centroid_dist``.

  * ``model_labels`` : ``dict[str, str | Path]`` mapping a label -> a model dir /
    HF id. The label encodes the condition (and optionally seed / trajectory
    milestone); ``group`` (arm), ``seed`` are inferred from it.
  * ``group``        : the ordering ARM inferred from the label -- ``"curriculum"``
    for curriculum / curriculum_noswitch (labels containing ``curriculum`` or
    ``cur_``), else ``"random"`` (switch / noswitch).
  * ``pair``         : formatted like ``"eng-nld"``.
  * ``flores_cache`` : optional pre-loaded probe set as a ``(texts, n)`` tuple
    (``texts`` = ``{lang: [sentences]}``); if ``None`` the probe is loaded from the
    Hub once. Passing it lets several figures share one download.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

# figures/ is sys.path[0] when a figNN_*.py runs, so plain `import common` works.
import common

PAIRS = [("eng", "nld"), ("eng", "zho"), ("nld", "zho")]
# FLORES+ (openlanguagedata/flores_plus) config per language; Chinese Simplified = cmn_Hans.
FLORES_PLUS_CFG = {"eng": "eng_Latn", "nld": "nld_Latn", "zho": "cmn_Hans"}


# --------------------------------------------------------------------------- #
# Probe set
# --------------------------------------------------------------------------- #
def load_flores_plus(n=997, split="dev"):
    """Load FLORES+ ``split`` for eng/nld/zho, aligned by sentence id (intersection).

    Requires HF auth for the gated ``openlanguagedata/flores_plus`` dataset -- see
    the module docstring. Returns ``(texts, n_aligned)`` where ``texts`` maps each
    language code to a list of ``n_aligned`` parallel sentences.
    """
    from datasets import load_dataset

    per = {}
    for lang, cfg in FLORES_PLUS_CFG.items():
        d = load_dataset("openlanguagedata/flores_plus", cfg, split=split)
        per[lang] = {row["id"]: row["text"] for row in d}
    ids = sorted(set.intersection(*[set(p.keys()) for p in per.values()]))
    if n:
        ids = ids[:n]
    return {lang: [per[lang][i] for i in ids] for lang in per}, len(ids)


# --------------------------------------------------------------------------- #
# Embedding + metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def embed(model, tok, sents, device, batch=32):
    """Return ``[n_layers+1, N, hid]`` mean-pooled hidden states."""
    per_layer = None
    for i in range(0, len(sents), batch):
        chunk = sents[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=128).to(device)
        out = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"].unsqueeze(-1).float()        # [B, seq, 1]
        denom = mask.sum(1).clamp(min=1)
        hs = out.hidden_states                                    # tuple len L+1, each [B, seq, hid]
        pooled = [(h * mask).sum(1) / denom for h in hs]          # each [B, hid]
        pooled = torch.stack(pooled, 0).float().cpu().numpy()     # [L+1, B, hid]
        per_layer = pooled if per_layer is None else np.concatenate([per_layer, pooled], axis=1)
    return per_layer                                              # [L+1, N, hid]


def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    xx = np.linalg.norm(X.T @ X, "fro")
    yy = np.linalg.norm(Y.T @ Y, "fro")
    return float(xy / (xx * yy + 1e-12))


def pair_metrics(E1, E2, rng):
    """``E1,E2``: ``[N, hid]`` for one layer. Returns a dict of metrics."""
    n = E1.shape[0]
    N1 = E1 / (np.linalg.norm(E1, axis=1, keepdims=True) + 1e-12)
    N2 = E2 / (np.linalg.norm(E2, axis=1, keepdims=True) + 1e-12)
    S = N1 @ N2.T                                                 # [N, N] cosine
    p1 = 0.5 * ((S.argmax(1) == np.arange(n)).mean() + (S.argmax(0) == np.arange(n)).mean())
    parallel = np.diag(S).mean()
    perm = rng.permutation(n)
    while np.any(perm == np.arange(n)):
        perm = rng.permutation(n)
    rand = S[np.arange(n), perm].mean()
    cos_gap = float(parallel - rand)
    cka = linear_cka(E1, E2)
    centroid = float(np.linalg.norm(N1.mean(0) - N2.mean(0)))
    return {"retrieval_p1": float(p1), "cos_gap": cos_gap, "cka": cka,
            "centroid_dist": centroid}


# --------------------------------------------------------------------------- #
# Label -> (arm, seed) inference
# --------------------------------------------------------------------------- #
def arm_of(label: str) -> str:
    """Ordering arm from a model label: 'curriculum' vs 'random'."""
    return "curriculum" if ("curriculum" in label or "cur_" in label) else "random"


def is_unilingual(label: str) -> bool:
    """True for the no-code-switch (unilingual) arm of a label.

    Matches ``noswitch`` / ``_nsw`` / ``-nsw`` so both the fig02_05 condition
    names (``curriculum_noswitch``, ``noswitch``) and the fig03 trajectory labels
    (``cur_noswitch-...``) resolve correctly. Note ``noswitch`` is NOT a substring
    of ``switch``, so ``switch`` correctly reads as code-switched.
    """
    return ("noswitch" in label) or ("nsw" in label)


def seed_of(label: str) -> int:
    """Parse the training seed from a label (42 if absent).

    Matches a ``-s<digits>`` group that is followed by ``-`` or end-of-string, so
    ``curriculum-s43`` -> 43 and ``cur_cs-s42-s1e0`` -> 42 (the trailing ``-s1e0``
    stage token is not followed by ``-``/end and is ignored).
    """
    m = re.findall(r"-s(\d+)(?=-|$)", label)
    return int(m[0]) if m else 42


def group_of(label: str) -> str:
    """The ``group`` column value: the ordering arm (see :func:`arm_of`)."""
    return arm_of(label)


# --------------------------------------------------------------------------- #
# Public compute entry point
# --------------------------------------------------------------------------- #
def retrieval_per_layer(model_labels, device="cuda", flores_cache=None,
                        n_sents=997, batch=32, rng_seed=0):
    """Per-(layer, pair) FLORES+ bitext retrieval P@1 for a set of checkpoints.

    Parameters
    ----------
    model_labels : dict[str, str | Path]
        ``{label: model_dir_or_hf_id}``. ``group``/``seed`` are inferred from each
        label (see module docstring). Models are loaded via ``common.load_model`` /
        ``common.load_tokenizer`` (so ``final_checkpoint`` resolution applies).
    device : str
        Torch device string (``"cuda"`` / ``"cpu"``).
    flores_cache : tuple | None
        Optional pre-loaded ``(texts, n)`` probe set to avoid re-downloading.
    Returns
    -------
    pandas.DataFrame
        columns ``model, group, seed, layer, pair, retrieval_p1, cos_gap, cka,
        centroid_dist``.
    """
    import pandas as pd

    if flores_cache is not None:
        texts, n = flores_cache
    else:
        texts, n = load_flores_plus(n_sents)
    print(f"probe: FLORES+ dev, {n} aligned sentences x {list(texts)}", flush=True)
    rng = np.random.default_rng(rng_seed)

    rows = []
    for label, path in model_labels.items():
        grp, seed = group_of(label), seed_of(label)
        print(f"[{label}] group={grp} seed={seed} loading {path}", flush=True)
        tok = common.load_tokenizer(path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = common.load_model(path, device=device, dtype=torch.float32)
        emb = {lang: embed(model, tok, texts[lang], device, batch=batch) for lang in texts}
        n_layers = emb["eng"].shape[0]
        for layer in range(n_layers):
            for a, b in PAIRS:
                mt = pair_metrics(emb[a][layer], emb[b][layer], rng)
                rows.append({"model": label, "group": grp, "seed": seed,
                             "layer": layer, "pair": f"{a}-{b}", **mt})
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    return pd.DataFrame(rows, columns=["model", "group", "seed", "layer", "pair",
                                       "retrieval_p1", "cos_gap", "cka", "centroid_dist"])
