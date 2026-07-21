"""Grid experiment (Figure 10): full factorial
   word-language (eng/nld/zho) x sentence-language (eng/nld/zho) x seen-embedded?

Ports ``grid_sample.py`` (context sampler) + ``grid_analyze.py`` (per-cell
stats) + ``grid_plots.py`` (heatmaps + advantage dot plot), path-parameterized.
Uses :func:`contextual.extract_reps` for the forward pass.

Words + equivalents come from the Exp-3 matched pairs (E = seen-embedded,
C = never-embedded, freq/POS/token-count matched). Per sentence-language L:
  own  (diagonal)  monolingual L sentences with w (native), 2*K_OWN -> own1/own2
  mL   (synthetic) monolingual L sentences with w's equivalent e, e replaced by w
  natL (natural)   phase_intra docs, matrix==L, w embedded (seen words only)

CORPUS DEPENDENCY FLAGS (see ``contextual.py``):
  * ``babylm_dir``  -- monolingual BabyLM corpora (NOT from download_data.py).
  * ``phase_intra`` -- DOWNLOADED ``data/curriculum/1_intra.parquet``.
  * ``pairs_path``  -- ``traj/measurement_pairs_mono_freq.parquet``, the Exp-3
                       matched-pair output (NOT from download_data.py).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import contextual
from .contextual import (LANGS, MINC, MAXC, SENT, UP_LAYERS,
                         babylm_texts, find_offset)

MODELS = ("cur_cs", "cur_nsw")           # curriculum arms (figures use these)
K_SYN = 20    # synthetic / natural contexts per (word, cell)
K_OWN = 40    # own contexts (split 20/20 into own1/own2)


def load_words(pairs_path):
    """{(w, lw): {"status": seen|unseen, "equiv": {tgt_lang: e}}} (ports it)."""
    import pandas as pd
    pairs = pd.read_parquet(pairs_path)
    words = {}
    for r in pairs.itertuples():
        for w, e, status in ((r.E_word, r.equiv_E, "seen"),
                             (r.C_word, r.equiv_C, "unseen")):
            rec = words.setdefault((w, r.src_lang), {"status": status, "equiv": {}})
            rec["equiv"][r.tgt_lang] = e
    return words


def _stream_sentences(babylm_dir, lang):
    from normalize_zh import to_simplified
    for text in babylm_texts(babylm_dir, lang):
        s = to_simplified(text) if lang == "zho" else text
        for m in (SENT["zho"] if lang == "zho" else SENT["lat"]).finditer(s):
            seg = m.group().strip()
            if MINC <= len(seg) <= MAXC:
                yield seg


def sample_contexts(lang, pairs_path, phase_intra_path, babylm_dir, out=None):
    """Write ``contexts_grid_{lang}.jsonl`` for sentence-language ``lang``
    (ports grid_sample.main). ``lang`` is the SENTENCE language of this pass.

    Keys ``"{own1|own2|m{L}|nat{L}}|{L_w}:{w}"``. Returns record count.
    """
    import pandas as pd
    from normalize_zh import to_simplified
    from tokenize_mixed import tokenize as mixed_tokenize

    L = lang
    words = load_words(pairs_path)

    own_words = {w for (w, lw) in words if lw == L}
    slot = defaultdict(list)              # equivalent surface (in L) -> [(w, lw)]
    for (w, lw), rec in words.items():
        if lw != L and L in rec["equiv"]:
            e = rec["equiv"][L]
            slot[to_simplified(e) if L == "zho" else e.lower()].append((w, lw))
    own_surf = {(to_simplified(w) if L == "zho" else w.lower()): w for w in own_words}

    remaining_own = {w: K_OWN for w in own_words}
    remaining_syn = {(w, lw): K_SYN for lst in slot.values() for (w, lw) in lst}
    records = []
    n = 0
    for seg in _stream_sentences(babylm_dir, L):
        if not any(v > 0 for v in remaining_own.values()) and \
           not any(v > 0 for v in remaining_syn.values()):
            break
        toks = {t if L == "zho" else t.lower() for t in mixed_tokenize(seg)}
        for sfc in toks & set(own_surf):
            w = own_surf[sfc]
            if remaining_own.get(w, 0) <= 0:
                continue
            pos = find_offset(seg, sfc, L)
            if pos < 0:
                continue
            i = K_OWN - remaining_own[w]
            ctype = "own1" if i < K_OWN // 2 else "own2"
            records.append({"word": f"{ctype}|{L}:{w}", "lang": L,
                            "is_embedded": False, "sentence": seg,
                            "char_start": pos, "char_end": pos + len(sfc)})
            remaining_own[w] -= 1; n += 1
        for es in toks & set(slot):
            pos = find_offset(seg, es, L)
            if pos < 0:
                continue
            for (w, lw) in slot[es]:
                if remaining_syn.get((w, lw), 0) <= 0:
                    continue
                sw = to_simplified(w) if lw == "zho" else w
                new = seg[:pos] + sw + seg[pos + len(es):]
                records.append({"word": f"m{L}|{lw}:{w}", "lang": lw,
                                "is_embedded": True, "sentence": new,
                                "char_start": pos, "char_end": pos + len(sw)})
                remaining_syn[(w, lw)] -= 1; n += 1

    # natural embedded contexts (seen words only), matrix == L
    cs = pd.read_parquet(phase_intra_path).rename(columns={"doc-id": "doc_id"})
    nat_targets = {(w, lw): K_SYN for (w, lw), rec in words.items()
                   if rec["status"] == "seen" and lw != L}
    surf_by_lang = defaultdict(dict)
    for (w, lw) in nat_targets:
        surf_by_lang[lw][to_simplified(w) if lw == "zho" else w.lower()] = w
    for lw, surf in surf_by_lang.items():
        sub = cs[(cs.language == L) & (cs.embedded_lang == lw)]
        targets = set(surf)
        rem = {surf[s]: K_SYN for s in targets}
        for text in sub["text"]:
            if not any(v > 0 for v in rem.values()):
                break
            s = to_simplified(text) if L == "zho" else text
            for m in (SENT["zho"] if L == "zho" else SENT["lat"]).finditer(s):
                seg = m.group().strip()
                if not (MINC <= len(seg) <= MAXC):
                    continue
                toks = {t if lw == "zho" else t.lower() for t in mixed_tokenize(seg)}
                for sfc in toks & targets:
                    w = surf[sfc]
                    if rem.get(w, 0) <= 0:
                        continue
                    pos = find_offset(seg, sfc, lw)
                    if pos < 0:
                        continue
                    records.append({"word": f"nat{L}|{lw}:{w}", "lang": lw,
                                    "is_embedded": True, "sentence": seg,
                                    "char_start": pos, "char_end": pos + len(sfc)})
                    rem[w] -= 1; n += 1

    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def analyze(npz, pairs_path, models=MODELS):
    """Per-cell cross/reliab/shift long table (ports grid_analyze.main core).

    Returns a DataFrame with columns model, word, L_w, L_s, status, source
    (syn=m | nat), layer, cross, reliab, shift.
    """
    import pandas as pd

    pairs = pd.read_parquet(pairs_path)
    status = {}
    for r in pairs.itertuples():
        status[(r.E_word, r.src_lang)] = "seen"
        status[(r.C_word, r.src_lang)] = "unseen"

    keys = list(npz["words"])
    idx = {k: i for i, k in enumerate(keys)}
    per_word = {}
    for k in keys:
        ctype, lw_w = k.split("|", 1)
        lw, w = lw_w.split(":", 1)
        d = per_word.setdefault((w, lw), {})
        if ctype in ("own1", "own2"):
            d[ctype] = k
        elif ctype.startswith("nat"):
            d[("nat", ctype[3:])] = k
        elif ctype.startswith("m"):
            d[("m", ctype[1:])] = k

    n_layers = np.asarray(npz[models[0]]).shape[1]
    counts = {m: np.asarray(npz[f"{m}_count"]) for m in models}
    rows = []
    for m in models:
        V = np.asarray(npz[m])
        for (w, lw), d in per_word.items():
            if "own1" not in d or "own2" not in d:
                continue
            i1, i2 = idx[d["own1"]], idx[d["own2"]]
            c1, c2 = counts[m][i1], counts[m][i2]
            if min(c1, c2) == 0:
                continue
            st = status.get((w, lw))
            for cell, key in d.items():
                if cell in ("own1", "own2"):
                    continue
                src, ls = cell
                ie = idx[key]
                if counts[m][ie] < 4:
                    continue
                for L in range(n_layers):
                    v1, v2 = _unit(V[i1, L]), _unit(V[i2, L])
                    own_all = _unit((V[i1, L] * c1 + V[i2, L] * c2) / (c1 + c2))
                    ve = _unit(V[ie, L])
                    reliab = float(v1 @ v2)
                    cross = float(own_all @ ve)
                    rows.append({"model": m, "word": w, "L_w": lw, "L_s": ls,
                                 "status": st, "source": src, "layer": L,
                                 "cross": cross, "reliab": reliab,
                                 "shift": reliab - cross})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots -- port grid_plots.py.
# --------------------------------------------------------------------------- #
LABEL = {"seen": "seen embedded", "unseen": "never embedded"}
MIN_N = 10


def _w_mean(df, up_layers=UP_LAYERS):
    upd = df[df.layer.isin(up_layers) & (df.source == "m")]
    return (upd.groupby(["model", "L_w", "L_s", "status", "word"])["shift"]
               .mean().reset_index())


def _cell_matrix(w_mean, model, status):
    M = np.full((3, 3), np.nan)
    N = np.zeros((3, 3), int)
    for i, lw in enumerate(LANGS):
        for j, ls in enumerate(LANGS):
            if lw == ls:
                M[i, j] = 0.0
                continue
            g = w_mean[(w_mean.model == model) & (w_mean.status == status) &
                       (w_mean.L_w == lw) & (w_mean.L_s == ls)]
            if len(g) >= MIN_N:
                M[i, j] = g["shift"].mean()
                N[i, j] = len(g)
    return M, N


def plot_heatmaps(df, out_png, up_layers=UP_LAYERS):
    """Fig A: 3x3 upper-layer context-language shift heatmaps, {seen,unseen} x
    {cs, nsw} + (nsw - cs) advantage columns."""
    import common
    plt, INK, MUT = common.plt, common.INK, common.MUT

    def draw_heat(ax, M, N, title, vmax, cmap="Reds", center0=False):
        vmin = -vmax if center0 else 0
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
        for i in range(3):
            for j in range(3):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center", color=MUT)
                elif np.isnan(M[i, j]):
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, hatch="///",
                                               fill=True, fc="#f0f0f0", ec="#cccccc"))
                    ax.text(j, i, "no\ncontrols", ha="center", va="center",
                            fontsize=6.5, color=MUT)
                else:
                    ax.text(j, i, f"{M[i,j]:+.3f}\n(n={N[i,j]})", ha="center",
                            va="center", fontsize=7.5,
                            color="white" if abs(M[i, j]) > 0.6 * vmax else INK)
        ax.set_xticks(range(3), LANGS)
        ax.set_yticks(range(3), LANGS)
        ax.set_title(title, fontsize=10, loc="left")
        return im

    w_mean = _w_mean(df, up_layers)
    mats = {(m, st): _cell_matrix(w_mean, m, st)
            for m in ("cur_cs", "cur_nsw") for st in ("seen", "unseen")}
    vmax = np.nanmax([np.nanmax(M) for M, _ in mats.values()])
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.6))
    for r, st in enumerate(("seen", "unseen")):
        Mc, Nc = mats[("cur_cs", st)]
        Mn, Nn = mats[("cur_nsw", st)]
        im0 = draw_heat(axes[r, 0], Mc, Nc, f"{LABEL[st]} · code-switched", vmax)
        draw_heat(axes[r, 1], Mn, Nn, f"{LABEL[st]} · no-switch", vmax)
        D = Mn - Mc
        imd = draw_heat(axes[r, 2], D, Nc, f"{LABEL[st]} · cs advantage (nsw$-$cs)",
                        np.nanmax(np.abs(D)) * 1.05 + 1e-9, cmap="PuOr_r", center0=True)
        fig.colorbar(im0, ax=axes[r, :2], fraction=0.03, pad=0.02)
        fig.colorbar(imd, ax=axes[r, 2], fraction=0.06, pad=0.04)
    for ax in axes[:, 0]:
        ax.set_ylabel("word language", color=INK)
    for ax in axes[1, :]:
        ax.set_xlabel("sentence language", color=MUT)
    fig.suptitle("Context-language shift by grid cell (upper layers; synthetic switches;\n"
                 "shift = own-vs-own ceiling $-$ cos(own, in-sentence-language); "
                 "lower = more language-invariant)", fontsize=11.5)
    common.save(fig, out_png)


def plot_advantage(df, out_png, up_layers=UP_LAYERS):
    """Fig B: cs advantage (nsw - cs shift) per direction, seen vs unseen, with
    95% bootstrap CIs over words."""
    import common
    plt, CS, DIFF, INK, MUT = common.plt, common.CS, common.DIFF, common.INK, common.MUT

    w_mean = _w_mean(df, up_layers)
    rng = np.random.default_rng(0)
    rows = []
    for (lw, ls, st), g in w_mean.groupby(["L_w", "L_s", "status"]):
        cs = g[g.model == "cur_cs"].set_index("word")["shift"]
        ns = g[g.model == "cur_nsw"].set_index("word")["shift"]
        common_idx = cs.index.intersection(ns.index)
        if len(common_idx) < MIN_N:
            continue
        d = (ns.loc[common_idx] - cs.loc[common_idx]).values
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        rows.append({"dir": f"{lw}$\\to${ls}", "status": st, "mean": d.mean(),
                     "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5),
                     "n": len(common_idx)})
    import pandas as pd
    res = pd.DataFrame(rows)
    dirs = sorted(res["dir"].unique())
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for st, col, off in (("seen", CS, -0.13), ("unseen", DIFF, +0.13)):
        s = res[res.status == st].set_index("dir").reindex(dirs)
        x = np.arange(len(dirs)) + off
        ax.errorbar(x, s["mean"], yerr=[s["mean"] - s["lo"], s["hi"] - s["mean"]],
                    fmt="o", color=col, ms=7, lw=2, capsize=4, label=LABEL[st])
        for xi, (mv, n) in enumerate(zip(s["mean"], s["n"])):
            if not np.isnan(mv):
                ax.annotate(f"n={int(n)}", (xi + off, mv), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=7, color=MUT)
    ax.axhline(0, color=MUT, lw=1, ls="--")
    ax.set_xticks(range(len(dirs)), dirs)
    ax.set_ylabel("cs advantage:  shift$_{\\mathrm{nsw}}$ $-$ shift$_{\\mathrm{cs}}$\n"
                  "(> 0: cs keeps word closer to own-language self)", color=INK)
    ax.set_title("Is the context-language invariance specific to words seen embedded in training?\n"
                 "seen-embedded vs never-embedded words, identical synthetic-switch construction; "
                 "95% bootstrap CIs over words", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    common.style_axes(ax)
    common.save(fig, out_png)
