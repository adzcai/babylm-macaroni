"""Contextual (forward-pass) representation extraction + analysis.

Ported, path-parameterized versions of the paper's
``extract_contextual.py`` + ``analyze_contextual.py`` (and the shared
context-sampling / word-set helpers they depend on). Feeds Figures 8/9/10.

The measurement is: for a target word in a context sentence, run the model
with ``output_hidden_states``, locate the word's sub-word span via the
tokenizer offset mapping, mean-pool the span at every layer (embeddings +
12 blocks = 13 layers), then average across a word's contexts to get one
vector per (word, model, layer).

Every hard-coded ``/n/.../drooryck/lmkiddo`` path from the sources is replaced
by an explicit argument that the ``figNN_*.py`` scripts resolve from
``--models-dir`` / ``--data-dir`` / ``--out`` (see ``figures/common.py``).

CORPUS DEPENDENCY FLAGS (not satisfied by ``download_data.py``, which fetches
only the eight code-switched training subsets):
  * ``babylm_dir`` -- the monolingual BabyLM eng/nld/zho corpora (arrow files
    ``babylm-{lang}-train-*.arrow``). Needed for every "own-language" context
    and for monolingual frequency counting. These are the public
    ``BabyLM-community/babylm-{eng,nld,zho}`` datasets; download them
    separately (e.g. into ``data/babylm/``).
  * ``align_dir`` -- ``attested_lexicon.parquet`` + ``exposure_sets.parquet``,
    the outputs of the span-mining / exposure pipeline
    (extract_pragmatic0630 -> clean_spans -> build_vocab + the alignment
    stage). That upstream is a separate multi-stage experiment and is NOT
    ported here; ``build_wordsets`` consumes its outputs.
  * ``freq_dir`` -- ``freq_{lang}.tsv`` monolingual frequency tables, produced
    by :func:`count_monolingual_freq` over ``babylm_dir``.
  * phase_intra -- maps to the DOWNLOADED ``data/curriculum/1_intra.parquet``
    (satisfied by ``download_data.py``).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from math import erfc, sqrt
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Constants (from config.yaml: n_layers=13, layer_aggregate=[6..11]).
# --------------------------------------------------------------------------- #
N_LAYERS = 13
UP_LAYERS = [6, 7, 8, 9, 10, 11]     # upper layers, excl. next-token layer 12
LANGS = ("eng", "nld", "zho")

# curriculum arm labels -> training conditions (public condition names).
# The two arms these figures contrast.
CONDITION_FOR_LABEL = {
    "cur_cs": "curriculum",
    "cur_nsw": "curriculum_noswitch",
    "rnd_cs": "switch",
    "rnd_nsw": "noswitch",
}
# cs<->noswitch pairs sharing a training-order condition.
PAIRS = {
    "curriculum": ("cur_cs", "cur_nsw"),
    "random": ("rnd_cs", "rnd_nsw"),
}

# sentence splitters / length gate (config.yaml min/max_sent_chars).
SENT = {"zho": re.compile(r"[^。！？\n]+[。！？]?"),
        "lat": re.compile(r"[^.!?\n]+[.!?]?")}
MINC, MAXC = 10, 300


# --------------------------------------------------------------------------- #
# Corpus helpers (BabyLM monolingual arrow files + offset finding).
# --------------------------------------------------------------------------- #
def babylm_texts(babylm_dir, lang):
    """Yield raw document texts from the monolingual BabyLM-``lang`` corpus.

    ``babylm_dir`` is searched recursively for ``babylm-{lang}-train-*.arrow``
    so it works whether you point at the dataset root or the HF snapshot dir.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    root = Path(babylm_dir)
    files = sorted(root.rglob(f"babylm-{lang}-train-*.arrow"))
    if not files:
        raise FileNotFoundError(
            f"no babylm-{lang}-train-*.arrow under {root} -- supply the "
            f"monolingual BabyLM-{lang} corpus (not fetched by download_data.py)"
        )
    for p in files:
        with pa.memory_map(str(p), "r") as src:
            for batch in ipc.open_stream(src):
                for t in batch.column("text"):
                    yield t.as_py()


def find_offset(sent, surface, lang):
    """Char offset of ``surface`` in ``sent`` (substring for zho, \\b-word else)."""
    if lang == "zho":
        return sent.find(surface)
    m = re.search(r"\b" + re.escape(surface) + r"\b", sent, re.IGNORECASE)
    return m.start() if m else -1


def iter_sentences(text, lang):
    splitter = SENT["zho"] if lang == "zho" else SENT["lat"]
    for m in splitter.finditer(text):
        s = m.group().strip()
        if MINC <= len(s) <= MAXC:
            yield s


# --------------------------------------------------------------------------- #
# Word-set + frequency utilities (ported from build_control_words / stopwords /
# count_monolingual_freq / v2_build_wordsets).
# --------------------------------------------------------------------------- #
def load_freq(path):
    """Read a ``word\\tcount`` monolingual frequency table into a dict."""
    freq = {}
    with open(path) as f:
        for line in f:
            word, count = line.rstrip("\n").split("\t")
            freq[word] = int(count)
    return freq


def log_bucket(count):
    import math
    return int(math.log2(max(count, 1)))


# closed-class additions stopwordsiso misses (from stopwords.py).
_STOPWORD_EXTRA = {
    "nld": {"eenmaal", "hebbend", "meeste", "zeer"},
    "zho": {"应该", "没有", "没", "非常", "很多", "同样", "嗎"},
    "eng": set(),
}
_ISO = {"eng": "en", "nld": "nl", "zho": "zh"}


def _load_stopwords():
    import stopwordsiso as _sw
    return {lang: {w.lower() for w in _sw.stopwords(iso)} | _STOPWORD_EXTRA.get(lang, set())
            for lang, iso in _ISO.items()}


def count_monolingual_freq(babylm_dir, lang, out=None):
    """Word-frequency table over monolingual BabyLM-``lang`` (ports
    count_monolingual_freq.py). zho is normalized to Simplified first.

    Writes a ``word\\tcount`` tsv if ``out`` is given; returns the Counter.
    """
    from collections import Counter
    from normalize_zh import to_simplified            # opencc
    from tokenize_mixed import tokenize as mixed_tokenize

    word_re = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)
    counts = Counter()
    for text in babylm_texts(babylm_dir, lang):
        if lang == "zho":
            text = to_simplified(text)
            toks = mixed_tokenize(text)
        else:
            toks = word_re.findall(text)
        counts.update(t.lower() for t in toks)
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for word, c in counts.most_common():
                f.write(f"{word}\t{c}\n")
    return counts


def build_wordsets(align_dir, freq_dir, controls_per_word=3, out_dir=None):
    """Embedded content-word vocab + frequency-matched controls.

    WRAPPER around ``v2_build_wordsets.py`` (its span-mining upstream --
    extract_pragmatic0630 -> clean_spans -> build_vocab + the alignment/exposure
    stage -- is too entangled with the CS-vs-noswitch diffing pipeline to port
    for a figures repo, so it is NOT ported; this consumes its outputs).

    Reads (all under ``align_dir`` unless noted):
      * ``exposure_sets.parquet``   [word, lang, ever_embedded, ever_replaced]
      * ``attested_lexicon.parquet``[embedded_word, embedded_lang, count]
      * ``freq_{lang}.tsv``         (under ``freq_dir``; see
        :func:`count_monolingual_freq`)

    Returns ``(vocab_df, control_df)`` with the v1 schemas:
      vocab   -> [word, embedded_lang, frequency]
      control -> [embedded_word, lang, embedded_word_native_freq,
                  control_word, control_word_freq]
    If ``out_dir`` given, also writes the two parquet files there.
    """
    import pandas as pd

    align_dir, freq_dir = Path(align_dir), Path(freq_dir)
    stop = _load_stopwords()

    def is_fw(word, lang):
        return word.strip().lower() in stop.get(lang, set())

    exp = pd.read_parquet(align_dir / "exposure_sets.parquet")
    att = pd.read_parquet(align_dir / "attested_lexicon.parquet")
    freqs = {lang: load_freq(freq_dir / f"freq_{lang}.tsv") for lang in LANGS}

    # --- vocab: embedded content-word types + embedded-occurrence counts
    emb_freq = (att.groupby(["embedded_word", "embedded_lang"], as_index=False)["count"].sum()
                   .rename(columns={"embedded_word": "word", "count": "frequency"}))
    emb_types = exp[exp.ever_embedded][["word", "lang"]].rename(columns={"lang": "embedded_lang"})
    vocab = emb_types.merge(emb_freq, on=["word", "embedded_lang"], how="left")
    vocab["frequency"] = vocab["frequency"].fillna(1).astype("int64")
    vocab = vocab[~vocab.apply(lambda r: is_fw(r["word"], r["embedded_lang"]), axis=1)]

    # --- controls: never embedded AND never replaced, same lang + log2-freq bucket
    rows = []
    for lang in LANGS:
        f = freqs[lang]
        sub = exp[exp.lang == lang]
        exposed = set(sub["word"])
        emb_words = [w for w in sub[sub.ever_embedded]["word"]
                     if w in f and not is_fw(w, lang)]
        pool_words = [w for w in f if w not in exposed and not is_fw(w, lang)]
        pool = defaultdict(list)
        for w in pool_words:
            pool[log_bucket(f[w])].append(w)
        used = defaultdict(int)
        for w in emb_words:
            b = log_bucket(f[w])
            cands = pool.get(b, [])
            if not cands:
                continue
            for j in range(min(controls_per_word, len(cands))):
                c = cands[(used[b] + j) % len(cands)]
                rows.append({"embedded_word": w, "lang": lang,
                             "embedded_word_native_freq": f[w],
                             "control_word": c, "control_word_freq": f[c]})
            used[b] += controls_per_word
    ctrl = pd.DataFrame(rows)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        vocab.to_parquet(out_dir / "vocab_embedded_words_content_only.parquet", index=False)
        ctrl.to_parquet(out_dir / "control_words.parquet", index=False)
    return vocab, ctrl


def sample_monolingual_contexts(vocab_df, control_df, lang, babylm_dir,
                                tokenizer_json, k=15, out=None):
    """Fig-8 context sampler (ports context_sampler.py, targets rebuilt from
    :func:`build_wordsets` outputs instead of the static-analysis npz).

    Targets = single-BPE-token embedded words (``vocab_df``) + single-token
    control words (``control_df``) for ``lang``. Streams the monolingual
    BabyLM-``lang`` corpus and collects up to ``k`` whole-word-matched
    sentences per target, recording char offsets for span alignment.

    Writes a ``contexts_{lang}.jsonl`` (keyed by bare surface ``word``,
    ``is_embedded`` flag) if ``out`` given; returns the list of records.
    """
    from normalize_zh import to_simplified
    from tokenize_mixed import tokenize as mixed_tokenize
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_json))

    def single(w):
        return len(tok.encode(w if lang == "zho" else " " + w).ids) == 1

    emb = [w for w in vocab_df.loc[vocab_df.embedded_lang == lang, "word"].unique()
           if single(w)]
    ctl = [w for w in control_df.loc[control_df.lang == lang, "control_word"].unique()
           if single(w)]
    words = list(dict.fromkeys(emb + ctl))          # embedded first, dedup
    word_emb = {w: (w in set(emb)) for w in words}
    surf = {(to_simplified(w) if lang == "zho" else w): w for w in words}
    target_surfaces = set(surf)

    remaining = {w: k for w in words}
    collected = defaultdict(list)
    n_active = len(remaining)
    for text in babylm_texts(babylm_dir, lang):
        if n_active == 0:
            break
        norm = to_simplified(text) if lang == "zho" else text
        for sent in iter_sentences(norm, lang):
            hits = {t.lower() for t in mixed_tokenize(sent)} & target_surfaces
            for surface in hits:
                w = surf[surface]
                if remaining[w] <= 0:
                    continue
                pos = find_offset(sent, surface, lang)
                if pos < 0:
                    continue
                collected[w].append({
                    "word": w, "lang": lang, "is_embedded": word_emb[w],
                    "sentence": sent, "char_start": pos,
                    "char_end": pos + len(surface),
                })
                remaining[w] -= 1
                if remaining[w] == 0:
                    n_active -= 1

    records = [rec for w in words for rec in collected[w]]
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


# --------------------------------------------------------------------------- #
# Extraction (GPU forward pass) -- ports extract_contextual.py.
# --------------------------------------------------------------------------- #
def load_contexts(path):
    """Read a contexts jsonl into ``(by_word, is_embedded)`` (ports the loader)."""
    by_word = defaultdict(list)
    meta = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            by_word[d["word"]].append(d)
            meta[d["word"]] = d.get("is_embedded", False)
    return by_word, meta


def _extract_for_model(model, tokenizer, contexts_by_word, words, device,
                       batch_size, n_layers):
    """Sum -> average per-layer mean-pooled span vectors per word."""
    import torch

    dim = getattr(model.config, "n_embd", None) or model.config.hidden_size
    acc = np.zeros((len(words), n_layers, dim), dtype=np.float64)
    cnt = np.zeros(len(words), dtype=np.int64)
    word_idx = {w: i for i, w in enumerate(words)}

    flat = [(w, c["sentence"], c["char_start"], c["char_end"])
            for w in words for c in contexts_by_word[w]]

    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            chunk = flat[start:start + batch_size]
            sents = [c[1] for c in chunk]
            enc = tokenizer(sents, return_offsets_mapping=True, return_tensors="pt",
                            padding=True, truncation=True, max_length=256)
            offsets = enc.pop("offset_mapping")
            attn = enc["attention_mask"]
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True)
            hs = torch.stack(out.hidden_states, dim=1)          # (B, L, T, D)
            hs = hs.to(torch.float32).cpu().numpy()

            for b, (w, sent, cs, ce) in enumerate(chunk):
                offs = offsets[b].numpy()
                mask = attn[b].numpy().astype(bool)
                tok_sel = [j for j in range(len(offs))
                           if mask[j] and offs[j][1] > offs[j][0]
                           and offs[j][0] < ce and offs[j][1] > cs]
                if not tok_sel:
                    continue
                span = hs[b].take(tok_sel, axis=1).mean(axis=1)  # (L, D)
                i = word_idx[w]
                acc[i] += span
                cnt[i] += 1

    cnt_safe = np.maximum(cnt, 1)[:, None, None]
    return (acc / cnt_safe).astype(np.float32), cnt


def extract_reps(model_paths, contexts_jsonl, tokenizer_dir, device="cuda",
                 batch_size=64, out=None):
    """Extract per-(word, model, layer) contextual reps over one contexts jsonl.

    Parameters
    ----------
    model_paths : dict {label -> model dir / HF checkpoint dir}
        e.g. ``{"cur_cs": <curriculum final ckpt>, "cur_nsw": <...>}``. The
        ``figNN_*.py`` scripts resolve these via
        ``common.final_ckpt(common.model_dir(condition, seed, models_dir))``.
    contexts_jsonl : path to a jsonl with rows {word, sentence, char_start,
        char_end, is_embedded}.
    tokenizer_dir : model dir holding the shared HF tokenizer (fast, for
        offset mapping). Use any of the ``model_paths`` values.
    out : optional ``.npz`` path to save the result dict.

    Returns a dict: ``words``, ``is_embedded``, and per label ``<label>`` (an
    ``(n_words, n_layers, D)`` float32 array) + ``<label>_count`` (n_words).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA not available; contextual extraction needs a GPU.")

    contexts_by_word, is_emb = load_contexts(contexts_jsonl)
    words = sorted(contexts_by_word)

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    result = {
        "words": np.array(words),
        "is_embedded": np.array([is_emb[w] for w in words]),
    }
    for name, ckpt_dir in model_paths.items():
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_dir), output_hidden_states=True).to(device).eval()
        n_layers = getattr(model.config, "n_layer", None)
        n_layers = (n_layers + 1) if n_layers is not None else (model.config.num_hidden_layers + 1)
        vecs, cnt = _extract_for_model(model, tokenizer, contexts_by_word, words,
                                       device, batch_size, n_layers)
        result[name] = vecs
        result[f"{name}_count"] = cnt
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **result)
    return result


# --------------------------------------------------------------------------- #
# Analysis -- ports analyze_contextual.py.
# --------------------------------------------------------------------------- #
def cohen_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / (sp + 1e-12)


def welch_p(a, b):
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (ma - mb) / np.sqrt(va + vb + 1e-12)
    return erfc(abs(t) / sqrt(2)), t


def cos_rows(A, B):
    na = np.linalg.norm(A, axis=1)
    nb = np.linalg.norm(B, axis=1)
    return np.sum(A * B, axis=1) / (na * nb + 1e-12)


def contextual_layerwise(npz_by_lang, pairs=None):
    """Per-layer embedded-vs-control cs<->noswitch contrast (returns DataFrame).

    Parameters
    ----------
    npz_by_lang : dict {lang -> result dict from :func:`extract_reps`} (or a
        loaded ``.npz`` mapping with the same keys).
    pairs : dict {condition -> (cs_label, nsw_label)}; default :data:`PAIRS`.

    Columns: lang, condition, layer, n_embedded, n_control, emb_cos, ctl_cos,
    cohen_d_cos, p_cos, emb_l2, ctl_l2, cohen_d_l2, p_l2.
    """
    import pandas as pd

    pairs = pairs or PAIRS
    labels = [m for pr in pairs.values() for m in pr]
    rows = []
    for lang, npz in npz_by_lang.items():
        is_emb = np.asarray(npz["is_embedded"]).astype(bool)
        valid = np.ones(len(is_emb), dtype=bool)
        for name in labels:
            valid &= np.asarray(npz[f"{name}_count"]) > 0
        is_emb_v = is_emb[valid]
        first = np.asarray(npz[labels[0]])
        n_layers = first.shape[1]

        for cond, (cs, nsw) in pairs.items():
            A = np.asarray(npz[cs])[valid]
            B = np.asarray(npz[nsw])[valid]
            for L in range(n_layers):
                cos = cos_rows(A[:, L], B[:, L])
                l2 = np.linalg.norm(A[:, L] - B[:, L], axis=1)
                ce, cc = cos[is_emb_v], cos[~is_emb_v]
                le, lc = l2[is_emb_v], l2[~is_emb_v]
                rows.append({
                    "lang": lang, "condition": cond, "layer": L,
                    "n_embedded": int(is_emb_v.sum()), "n_control": int((~is_emb_v).sum()),
                    "emb_cos": ce.mean(), "ctl_cos": cc.mean(),
                    "cohen_d_cos": cohen_d(ce, cc), "p_cos": welch_p(ce, cc)[0],
                    "emb_l2": le.mean(), "ctl_l2": lc.mean(),
                    "cohen_d_l2": cohen_d(le, lc), "p_l2": welch_p(le, lc)[0],
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plot -- ports embedded_plots_v2.repr_divergence (Fig 8).
# --------------------------------------------------------------------------- #
def plot_repr_divergence(df, out_png):
    """Cross-arm cosine (cs vs. unilingual, same word) per layer, embedded
    (solid) vs. matched controls (dashed), per language; 3-seed mean +- 1 SD.

    ``df`` must have columns lang, layer, seed, emb_cos, ctl_cos (already
    filtered to the ``curriculum`` condition).
    """
    import common
    plt, LANG, DIFF, MUT, INK = (common.plt, common.LANG, common.DIFF,
                                 common.MUT, common.INK)
    LANG_LAB = {"eng": "English", "nld": "Dutch", "zho": "Chinese"}

    layers = sorted(df.layer.unique())
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for lang, col in LANG.items():
        for kind, ls, lab in (("emb_cos", "-", "embedded"), ("ctl_cos", "--", "control")):
            piv = (df[df.lang == lang]
                   .pivot_table(index="layer", columns="seed", values=kind)
                   .reindex(layers))
            m, sd = piv.mean(axis=1).values, piv.std(axis=1, ddof=0).values
            ax.plot(layers, m, ls, color=col, lw=2.2 if kind == "emb_cos" else 1.6,
                    marker="o" if kind == "emb_cos" else None, ms=4,
                    alpha=1.0 if kind == "emb_cos" else 0.7,
                    label=f"{LANG_LAB[lang]} {lab}")
            ax.fill_between(layers, m - sd, m + sd, color=col, alpha=0.10, lw=0)
    ax.set_xlabel("layer", color=MUT)
    ax.set_ylabel("cross-arm cosine  (cs vs. unilingual, same word)\n"
                  "1 $=$ unchanged by switching", color=INK)
    ax.set_title("How far code-switching moves a word's representation, by layer\n"
                 "embedded words (solid) vs. matched never-embedded controls "
                 "(dashed); 3 seeds, mean $\\pm$ 1 SD", fontsize=10.5)
    ax.set_xticks(layers)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    common.style_axes(ax)
    common.save(fig, out_png)
