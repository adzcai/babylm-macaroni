"""Embedded-vs-own-language representation (Figure 9).

Ports ``embown_sample.py`` (context sampler) + ``embown_analyze.py`` (stats),
path-parameterized. Uses :func:`contextual.extract_reps` for the forward pass.

For each single-token embedded word ``w`` (language ``L``) we collect:
  own  = monolingual BabyLM-``L`` sentences containing ``w`` (native matrix),
         split into two disjoint halves own1/own2 for an own-vs-own ceiling;
  emb  = phase_intra sentences where ``w`` is the EMBEDDED token (matrix != L)
         -- the code-switched setting the cs model saw.
Then, per word/layer/model: cross = cos(own, emb), reliab = cos(own1, own2),
shift = reliab - cross.

CORPUS DEPENDENCY FLAGS (see ``contextual.py`` for the full list):
  * ``babylm_dir``     -- monolingual BabyLM corpora (NOT from download_data.py).
  * ``phase_intra``    -- DOWNLOADED ``data/curriculum/1_intra.parquet``.
  * ``attested_path``  -- ``align/attested_lexicon.parquet`` (mining output,
                          NOT from download_data.py).
  * ``tokenizer_json`` -- ``tokenizer.json`` from any downloaded model dir.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import contextual
from .contextual import (LANGS, MINC, MAXC, SENT, UP_LAYERS,
                         babylm_texts, find_offset)

MODELS = ("cur_cs", "cur_nsw")


def _collect(sent_iter, lang, targets, surf, k, ctype, out, remaining):
    """Stream sentences; collect up to k contexts per word (ports collect())."""
    from normalize_zh import to_simplified
    from tokenize_mixed import tokenize as mixed_tokenize

    n_active = sum(1 for w in remaining if remaining[w] > 0)
    for sent in sent_iter:
        if n_active == 0:
            break
        s = to_simplified(sent) if lang == "zho" else sent
        for m in (SENT["zho"] if lang == "zho" else SENT["lat"]).finditer(s):
            seg = m.group().strip()
            if not (MINC <= len(seg) <= MAXC):
                continue
            hits = {t.lower() for t in mixed_tokenize(seg)} & targets
            for sfc in hits:
                w = surf[sfc]
                if remaining[w] <= 0:
                    continue
                pos = find_offset(seg, sfc, lang)
                if pos < 0:
                    continue
                out[w].append({"word": f"{ctype}|{lang}:{w}", "lang": lang,
                               "is_embedded": False, "sentence": seg,
                               "char_start": pos, "char_end": pos + len(sfc)})
                remaining[w] -= 1
                if remaining[w] == 0:
                    n_active -= 1


def sample_contexts(lang, attested_path, phase_intra_path, babylm_dir,
                    tokenizer_json, k=20, out=None):
    """Write ``contexts_{lang}.jsonl`` of own1/own2/emb contexts (ports
    embown_sample.main). Returns the number of records written.

    Word keys are ``"{own1|own2|emb}|{lang}:{word}"`` so a single downstream
    :func:`contextual.extract_reps` keeps them distinct.
    """
    import pandas as pd
    from normalize_zh import to_simplified
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_json))

    att = pd.read_parquet(attested_path)
    words = sorted({w for w in att.loc[att.embedded_lang == lang, "embedded_word"]})

    def single(w):
        return len(tok.encode(w if lang == "zho" else " " + w).ids) == 1
    words = [w for w in words if single(w)]
    surf = {(to_simplified(w) if lang == "zho" else w): w for w in words}
    targets = set(surf)

    # own: 2K monolingual contexts (later split into halves) from BabyLM-lang
    own = defaultdict(list)
    _collect(babylm_texts(babylm_dir, lang), lang, targets, surf, 2 * k, "own",
             own, {w: 2 * k for w in words})

    # emb: K embedded contexts from phase_intra docs whose embedded_lang == lang
    cs = pd.read_parquet(phase_intra_path).rename(columns={"doc-id": "doc_id"})
    cs = cs[(cs.embedded_lang == lang) & (cs.language != lang)]
    emb = defaultdict(list)
    _collect((t for t in cs["text"]), lang, targets, surf, k, "emb",
             emb, {w: k for w in words})

    n = 0
    records = []
    for w in words:
        o, e = own[w], emb[w]
        if len(o) < 4 or len(e) < 4:            # need enough of BOTH context types
            continue
        half = len(o) // 2
        for i, rec in enumerate(o):
            rec["word"] = f"{'own1' if i < half else 'own2'}|{lang}:{w}"
            records.append(rec); n += 1
        for rec in e:
            records.append(rec); n += 1
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def analyze(npz, models=MODELS):
    """Per word/layer/model cross, reliab, shift (ports embown_analyze.main).

    ``npz`` is a result dict from :func:`contextual.extract_reps` (keys
    ``own1|lang:w`` / ``own2|lang:w`` / ``emb|lang:w``). Returns a DataFrame
    with columns model, word, layer, reliab, cross, shift.
    """
    import pandas as pd

    words = list(npz["words"])
    idx = {w: i for i, w in enumerate(words)}
    base = {}
    for k in words:
        ctype, lw = k.split("|", 1)
        base.setdefault(lw, {})[ctype] = k
    usable = [lw for lw, d in base.items() if {"own1", "own2", "emb"} <= set(d)]

    counts = {m: np.asarray(npz[f"{m}_count"]) for m in models}
    n_layers = np.asarray(npz[models[0]]).shape[1]
    rows = []
    for m in models:
        V = np.asarray(npz[m])
        for lw in usable:
            d = base[lw]
            i1, i2, ie = idx[d["own1"]], idx[d["own2"]], idx[d["emb"]]
            if min(counts[m][i1], counts[m][i2], counts[m][ie]) == 0:
                continue
            c1, c2 = counts[m][i1], counts[m][i2]
            for L in range(n_layers):
                v1, v2, ve = _unit(V[i1, L]), _unit(V[i2, L]), _unit(V[ie, L])
                own_all = _unit((V[i1, L] * c1 + V[i2, L] * c2) / (c1 + c2))
                reliab = float(v1 @ v2)
                cross = float(own_all @ ve)
                rows.append({"model": m, "word": lw, "layer": L,
                             "reliab": reliab, "cross": cross,
                             "shift": reliab - cross})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plot -- ports embedded_plots_v2.embedded_own (Fig 9).
# --------------------------------------------------------------------------- #
# Neutral dark/light slate pair for the two arms (line style is spent on
# own-vs-embedded vs the reliability ceiling).
MODEL_SLATE = {"cur_cs": "#334155", "cur_nsw": "#94a3b8"}


def plot_embedded_own(df, out_png):
    """Panel A: own-vs-embedded cosine + own-vs-own ceiling per arm.
    Panel B: cs invariance advantage (shift_nsw - shift_cs) per layer.

    ``df`` has columns seed, model, layer, cross, reliab, shift.
    """
    import common
    plt, DIFF, MUT, INK = common.plt, common.DIFF, common.MUT, common.INK

    g = (df.groupby(["seed", "model", "layer"])[["cross", "reliab", "shift"]]
         .mean().reset_index())
    layers = sorted(g.layer.unique())

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.0))
    for model, col, lab in (("cur_cs", MODEL_SLATE["cur_cs"], "code-switched"),
                            ("cur_nsw", MODEL_SLATE["cur_nsw"], "unilingual baseline")):
        gm = g[g.model == model]
        cross = gm.pivot_table(index="layer", columns="seed", values="cross").reindex(layers)
        rel = gm.pivot_table(index="layer", columns="seed", values="reliab").reindex(layers)
        cm, csd = cross.mean(axis=1).values, cross.std(axis=1, ddof=0).values
        axA.plot(layers, cm, "-o", color=col, lw=2.2, ms=4, label=f"{lab}: own vs. embedded")
        axA.fill_between(layers, cm - csd, cm + csd, color=col, alpha=0.12, lw=0)
        axA.plot(layers, rel.mean(axis=1).values, ":", color=col, lw=1.4, alpha=0.7,
                 label=f"{lab}: own vs. own (ceiling)")
    axA.set_xlabel("layer", color=MUT); axA.set_xticks(layers)
    axA.set_ylabel("cosine similarity", color=INK)
    axA.set_title("A · Is an embedded word encoded like its own-language self?",
                  fontsize=10.5, loc="left")
    axA.legend(frameon=False, fontsize=8, loc="lower left")
    common.style_axes(axA)

    piv = g.pivot_table(index=["seed", "layer"], columns="model", values="shift").reset_index()
    piv["adv"] = piv["cur_nsw"] - piv["cur_cs"]
    a = piv.pivot_table(index="layer", columns="seed", values="adv").reindex(layers)
    m, sd = a.mean(axis=1).values, a.std(axis=1, ddof=0).values
    axB.axhline(0, color=MUT, lw=1, ls="--")
    axB.plot(layers, m, "-o", color=DIFF, lw=2.4, ms=4)
    axB.fill_between(layers, m - sd, m + sd, color=DIFF, alpha=0.15, lw=0)
    axB.set_xlabel("layer", color=MUT); axB.set_xticks(layers)
    axB.set_ylabel("cs invariance advantage\n"
                   "shift$_{\\mathrm{uni}}$ $-$ shift$_{\\mathrm{cs}}$", color=INK)
    axB.set_title("B · Switching keeps embedded words closer to their own-language self",
                  fontsize=10.5, loc="left")
    common.style_axes(axB)
    common.save(fig, out_png)
