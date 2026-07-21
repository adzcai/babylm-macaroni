"""Lexicon + candidate + corpus-frequency builders (ported, path-parameterized).

Ports the three upstream scripts that ``align_static.py`` consumes:

  * ``build_corpus_stats``  <- align_mine_phase_intra.py
  * ``build_candidates``    <- align_build_candidates.py
  * ``build_lexicon``       <- align_build_lexicon.py

plus ``prepare_lexicon`` — a build-or-load convenience the figure scripts call to
get ``(eval_lexicon, candidates, corpus_freq)``, caching intermediates under a
directory (default ``<out>/align_cache``) so re-runs skip the expensive corpus
mine.

Public-repo path mapping (vs the old hardcoded lmkiddo layout):
  * ``phase_intra`` / ``phase_intra_noswitch``  ->  the curriculum stage-1
    corpora ``curriculum/1_intra.parquet`` and ``curriculum_noswitch/1_intra.parquet``
    under ``data_dir`` (see the SHARED CONTRACT phase-name map).
  * per-language monolingual frequency tables ``freq_{lang}.tsv``  ->  ``data_dir``.
  * MUSE bilingual dictionaries (en-nl, nl-en, en-zh, zh-en)       ->  ``data_dir/muse/``.
  * tokenizer  ->  a HuggingFace tokenizer loaded from a downloaded model dir
    (``common.load_tokenizer(model_dir(...))``); the token-id convention is
    eng/nld leading-space (" word"), zho bare — identical to the paper.

``build_corpus_stats`` additionally needs ``opencc``, ``jieba`` and
``stopwordsiso`` (imported lazily by ``align.text``).
"""
import difflib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from . import text as T

LANGS = ("eng", "nld", "zho")
MUSE_LANG = {"en": "eng", "nl": "nld", "zh": "zho"}


# --------------------------------------------------------------------------- #
# Tokenizer adapter: works with a HF (fast) tokenizer or a raw tokenizers.Tokenizer
# --------------------------------------------------------------------------- #
def _encode_ids(tokenizer, text):
    """Return the list of token ids for ``text`` (no special tokens)."""
    # raw tokenizers.Tokenizer
    if hasattr(tokenizer, "encode") and not hasattr(tokenizer, "backend_tokenizer"):
        enc = tokenizer.encode(text)
        if hasattr(enc, "ids"):
            return list(enc.ids)
    # HF PreTrainedTokenizer(Fast)
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _word_token_ids(tokenizer, word, lang):
    """Token ids under the canonical convention: zho bare, eng/nld leading-space."""
    return _encode_ids(tokenizer, word if lang == "zho" else " " + word)


# --------------------------------------------------------------------------- #
# A1+A2: mine the actual training corpora (code-switched vs unilingual)
# --------------------------------------------------------------------------- #
def _phase_paths(data_dir):
    """Locate the stage-1 code-switched / unilingual corpora under data_dir.

    Public layout: curriculum/1_intra.parquet vs curriculum_noswitch/1_intra.parquet.
    Falls back to the old phase_intra/data.parquet layout if present.
    """
    data_dir = Path(data_dir)
    cs = data_dir / "curriculum" / "1_intra.parquet"
    ns = data_dir / "curriculum_noswitch" / "1_intra.parquet"
    if cs.is_file() and ns.is_file():
        return cs, ns
    cs_old = data_dir / "phase_intra" / "data.parquet"
    ns_old = data_dir / "phase_intra_noswitch" / "data.parquet"
    if cs_old.is_file() and ns_old.is_file():
        return cs_old, ns_old
    raise FileNotFoundError(
        f"could not find the code-switched/unilingual stage-1 corpora under {data_dir} "
        f"(looked for curriculum/1_intra.parquet + curriculum_noswitch/1_intra.parquet). "
        f"Run download_data.py --only curriculum curriculum_noswitch.")


def build_corpus_stats(data_dir, min_count=3, min_share=0.6, limit=None,
                       align_dir=None):
    """Mine the training corpora -> {attested, corpus_freq, exposure} DataFrames.

    Diffs the code-switched stage-1 corpus against its unilingual counterfactual
    doc-by-doc to produce, in one pass over exactly the text the models trained on:
    per-word per-corpus frequencies, the attested single-word translation
    lexicon, and exposure sets (ever_embedded / ever_replaced) for tiering.
    If ``align_dir`` is given, writes the three parquets there.
    """
    cs_path, ns_path = _phase_paths(data_dir)
    cs = pd.read_parquet(cs_path).rename(columns={"doc-id": "doc_id"})
    ns = pd.read_parquet(ns_path).rename(columns={"doc-id": "doc_id"})
    ns_by_id = dict(zip(ns["doc_id"], ns["text"]))
    assert set(cs["doc_id"]) == set(ns["doc_id"]), "doc-id sets differ"

    freq_intra = Counter()
    freq_noswitch = Counter()
    pair_counts = defaultdict(Counter)   # (matrix, embedded, emb_word) -> Counter(replaced_word)
    ever_embedded = set()                # (word, embedded_lang)
    ever_replaced = set()                # (word, matrix_lang)

    n = 0
    for row in cs.itertuples(index=False):
        if limit is not None and n >= limit:
            break
        n += 1
        matrix, embedded = row.language, row.embedded_lang
        cs_toks = [t.lower() for t in T.tokenize(T.to_simplified(row.text))]
        ns_toks = [t.lower() for t in T.tokenize(T.to_simplified(ns_by_id[row.doc_id]))]
        freq_intra.update(cs_toks)
        freq_noswitch.update(ns_toks)
        sm = difflib.SequenceMatcher(None, ns_toks, cs_toks, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag not in ("replace", "insert"):
                continue
            emb_span = cs_toks[j1:j2]
            rep_span = ns_toks[i1:i2]
            for w in emb_span:
                ever_embedded.add((w, embedded))
            for w in rep_span:
                ever_replaced.add((w, matrix))
            if tag == "replace" and len(emb_span) == 1 and len(rep_span) == 1:
                e, r = emb_span[0], rep_span[0]
                if e == r:
                    continue
                if not (any(c.isalpha() for c in e) and any(c.isalpha() for c in r)):
                    continue
                if T.is_function_word(e, embedded) or T.is_function_word(r, matrix):
                    continue
                pair_counts[(matrix, embedded, e)][r] += 1

    rows = []
    for (matrix, embedded, e), ctr in pair_counts.items():
        total = sum(ctr.values())
        r, c = ctr.most_common(1)[0]
        if not T.pair_script_ok(e, embedded, r, matrix):
            continue
        if total >= min_count and c / total >= min_share:
            rows.append({
                "matrix_lang": matrix, "embedded_lang": embedded,
                "embedded_word": e, "replaced_word": r,
                "count": c, "total": total, "majority_share": round(c / total, 3),
            })
    attested = pd.DataFrame(rows).sort_values("count", ascending=False)

    words = sorted(set(freq_intra) | set(freq_noswitch))
    corpus_freq = pd.DataFrame({
        "word": words,
        "freq_intra": [freq_intra.get(w, 0) for w in words],
        "freq_noswitch": [freq_noswitch.get(w, 0) for w in words],
    })

    exp_words = sorted(ever_embedded | ever_replaced)
    exposure = pd.DataFrame({
        "word": [w for w, _ in exp_words],
        "lang": [l for _, l in exp_words],
        "ever_embedded": [(w, l) in ever_embedded for w, l in exp_words],
        "ever_replaced": [(w, l) in ever_replaced for w, l in exp_words],
    })

    if align_dir is not None:
        align_dir = Path(align_dir)
        align_dir.mkdir(parents=True, exist_ok=True)
        attested.to_parquet(align_dir / "attested_lexicon.parquet", index=False)
        corpus_freq.to_parquet(align_dir / "corpus_freq.parquet", index=False)
        exposure.to_parquet(align_dir / "exposure_sets.parquet", index=False)
    return {"attested": attested, "corpus_freq": corpus_freq, "exposure": exposure}


# --------------------------------------------------------------------------- #
# frozen per-language BLI candidate vocabularies
# --------------------------------------------------------------------------- #
def _load_freqs(data_dir):
    freqs = {}
    for lang in LANGS:
        d = {}
        with open(Path(data_dir) / f"freq_{lang}.tsv") as f:
            for line in f:
                w, c = line.rstrip("\n").split("\t")
                d[w] = int(c)
        freqs[lang] = d
    return freqs


def build_candidates(data_dir, tokenizer, n=5000, ambig_top=20000, ratio=9.0,
                     align_dir=None):
    """Top-N single-BPE-token, language-exclusive word types per language.

    Reads ``data_dir/freq_{lang}.tsv``; returns ``{lang: DataFrame(word,
    token_id, freq)}``. The SAME frozen lists are used for every model. If
    ``align_dir`` is given, also writes ``bli_candidates_{lang}.jsonl`` there.
    """
    freqs = _load_freqs(data_dir)
    topsets = {
        lang: set(sorted(freqs[lang], key=freqs[lang].get, reverse=True)[:ambig_top])
        for lang in LANGS
    }
    out = {}
    for lang in LANGS:
        others = [o for o in LANGS if o != lang]
        picked = []
        for w in sorted(freqs[lang], key=freqs[lang].get, reverse=True):
            if len(picked) >= n:
                break
            if not any(c.isalpha() for c in w):
                continue
            f = freqs[lang][w]
            ambiguous = False
            for o in others:
                fo = freqs[o].get(w, 0)
                if w in topsets[o] and f < ratio * max(fo, 1):
                    ambiguous = True
                    break
            if ambiguous:
                continue
            ids = _word_token_ids(tokenizer, w, lang)
            if len(ids) != 1:
                continue
            picked.append((w, ids[0], f))
        out[lang] = pd.DataFrame(picked, columns=["word", "token_id", "freq"])
        if align_dir is not None:
            align_dir = Path(align_dir)
            align_dir.mkdir(parents=True, exist_ok=True)
            with open(align_dir / f"bli_candidates_{lang}.jsonl", "w") as fh:
                for w, tid, fr in picked:
                    fh.write(json.dumps({"word": w, "token_id": int(tid), "freq": int(fr)},
                                        ensure_ascii=False) + "\n")
    return out


# --------------------------------------------------------------------------- #
# unified, tiered evaluation lexicon (attested + MUSE)
# --------------------------------------------------------------------------- #
def _tri_jaccard(a, b):
    ta = {a[i:i + 3] for i in range(max(len(a) - 2, 1))}
    tb = {b[i:i + 3] for i in range(max(len(b) - 2, 1))}
    return len(ta & tb) / max(len(ta | tb), 1)


def build_lexicon(data_dir, tokenizer, corpus_stats=None, muse_dir=None,
                  align_dir=None):
    """Build the tiered evaluation lexicon DataFrame (attested + MUSE).

    ``corpus_stats`` is the dict from ``build_corpus_stats`` (attested, exposure,
    corpus_freq); if ``None``, they are read from ``align_dir``. ``muse_dir``
    defaults to ``data_dir/muse``. Every pair is normalized, near-identical
    pairs dropped, and tiered attested / pair_unseen / word_unseen with
    single-token status + token ids and per-corpus frequencies.
    """
    data_dir = Path(data_dir)
    muse_dir = Path(muse_dir) if muse_dir is not None else data_dir / "muse"
    if corpus_stats is None:
        if align_dir is None:
            raise ValueError("provide corpus_stats or align_dir to load them from")
        align_dir = Path(align_dir)
        corpus_stats = {
            "attested": pd.read_parquet(align_dir / "attested_lexicon.parquet"),
            "exposure": pd.read_parquet(align_dir / "exposure_sets.parquet"),
            "corpus_freq": pd.read_parquet(align_dir / "corpus_freq.parquet"),
        }
    attested = corpus_stats["attested"]
    exposure = corpus_stats["exposure"]
    cfreq = corpus_stats["corpus_freq"]

    def norm(w):
        return T.to_simplified(w.strip().lower())

    freq_i = dict(zip(cfreq["word"], cfreq["freq_intra"]))
    freq_n = dict(zip(cfreq["word"], cfreq["freq_noswitch"]))
    ever_emb = set(zip(exposure.loc[exposure["ever_embedded"], "word"],
                       exposure.loc[exposure["ever_embedded"], "lang"]))
    ever_rep = set(zip(exposure.loc[exposure["ever_replaced"], "word"],
                       exposure.loc[exposure["ever_replaced"], "lang"]))
    att_pairs = set()
    for r in attested.itertuples(index=False):
        e, rp = norm(r.embedded_word), norm(r.replaced_word)
        att_pairs.add((r.embedded_lang, e, r.matrix_lang, rp))
        att_pairs.add((r.matrix_lang, rp, r.embedded_lang, e))

    rows = []

    def add_pair(src_lang, src, tgt_lang, tgt, source):
        src, tgt = norm(src), norm(tgt)
        if not src or not tgt or src == tgt:
            return
        if " " in src or " " in tgt:
            return
        sids = _word_token_ids(tokenizer, src, src_lang)
        tids = _word_token_ids(tokenizer, tgt, tgt_lang)
        if sids == tids:
            return
        if (src_lang, src, tgt_lang, tgt) in att_pairs:
            tier = "attested"
        elif (src, src_lang) in ever_emb or (tgt, tgt_lang) in ever_emb:
            tier = "pair_unseen"
        else:
            tier = "word_unseen"
        exposure_identical = (
            tier == "word_unseen"
            and (src, src_lang) not in ever_rep and (tgt, tgt_lang) not in ever_rep
        )
        rows.append({
            "src_lang": src_lang, "tgt_lang": tgt_lang,
            "src_word": src, "tgt_word": tgt, "source": source, "tier": tier,
            "exposure_identical": exposure_identical,
            "src_single": len(sids) == 1, "tgt_single": len(tids) == 1,
            "src_token_id": sids[0] if len(sids) == 1 else -1,
            "tgt_token_id": tids[0] if len(tids) == 1 else -1,
            "src_freq_intra": freq_i.get(src, 0), "src_freq_noswitch": freq_n.get(src, 0),
            "tgt_freq_intra": freq_i.get(tgt, 0), "tgt_freq_noswitch": freq_n.get(tgt, 0),
            "ortho_jaccard": round(_tri_jaccard(src, tgt), 3),
        })

    for r in attested.itertuples(index=False):
        add_pair(r.embedded_lang, r.embedded_word, r.matrix_lang, r.replaced_word,
                 "attested_mined")

    for fname in ["en-nl", "nl-en", "en-zh", "zh-en"]:
        s, t = fname.split("-")
        path = muse_dir / f"{fname}.txt"
        if not path.is_file():
            print(f"  [build_lexicon] MUSE dict {path} missing; skipping "
                  f"(attested-only lexicon for direction {s}->{t})")
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) != 2:
                    continue
                add_pair(MUSE_LANG[s], parts[0], MUSE_LANG[t], parts[1], "muse")

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["src_lang", "tgt_lang", "src_word", "tgt_word"])
    if align_dir is not None:
        align_dir = Path(align_dir)
        align_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(align_dir / "eval_lexicon.parquet", index=False)
    return df


# --------------------------------------------------------------------------- #
# build-or-load convenience for the figure scripts
# --------------------------------------------------------------------------- #
def _load_candidates(align_dir):
    return {lang: pd.read_json(Path(align_dir) / f"bli_candidates_{lang}.jsonl", lines=True)
            for lang in LANGS}


def prepare_lexicon(data_dir, tokenizer, cache_dir, rebuild=False, **mine_kw):
    """Return ``(eval_lexicon, candidates, corpus_freq)``, building + caching
    them under ``cache_dir`` on first use and loading the cache thereafter.

    Set ``rebuild=True`` to force a rebuild. Extra keyword args go to
    ``build_corpus_stats`` (e.g. ``min_count``, ``limit``).
    """
    cache = Path(cache_dir)
    lex_p = cache / "eval_lexicon.parquet"
    cf_p = cache / "corpus_freq.parquet"
    have_cands = all((cache / f"bli_candidates_{l}.jsonl").is_file() for l in LANGS)
    if not rebuild and lex_p.is_file() and cf_p.is_file() and have_cands:
        return (pd.read_parquet(lex_p), _load_candidates(cache), pd.read_parquet(cf_p))

    cache.mkdir(parents=True, exist_ok=True)
    stats = build_corpus_stats(data_dir, align_dir=cache, **mine_kw)
    cands = build_candidates(data_dir, tokenizer, align_dir=cache)
    lex = build_lexicon(data_dir, tokenizer, corpus_stats=stats, align_dir=cache)
    return lex, cands, stats["corpus_freq"]
