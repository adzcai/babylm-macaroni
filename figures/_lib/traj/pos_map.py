"""Type -> majority-POS map (ports traj_build_pos_map.py + allcontent_tag.py).

The paper first tagged every token of the monolingual stage-1 corpus into a
private ``data/allcontent/tagged.jsonl.gz`` (spaCy UPOS for eng/nld, jieba flags
for zho), then reduced it to a per-type majority tag in a second script. That
intermediate is NOT part of the published download, but it is fully DERIVABLE
from the downloaded corpus: it was built from ``phase_intra_noswitch`` == the
monolingual curriculum's stage-1 parquet, i.e.
``data/curriculum_noswitch/1_intra.parquet``. So this module folds the tagging
in and streams straight from that parquet — no external file needed.

If a pre-tagged ``data/allcontent/tagged.jsonl.gz`` IS present it is used as-is
(matching the paper byte-for-byte); otherwise tagging runs on the parquet.

Taggers require spaCy (+ ``en_core_web_sm``, ``nl_core_news_sm``) and jieba,
which are NOT in the base requirements.txt. See the friendly error below.

The two tagsets (UPOS vs jieba) are incompatible, so POS matching downstream is
WITHIN-LANGUAGE only — which is all that is needed (controls are same-language).
"""
from __future__ import annotations

from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd

from . import config as _cfg

SPACY_MODEL = {"eng": "en_core_web_sm", "nld": "nl_core_news_sm"}


def _tag_zho(texts):
    """Yield [(token, pos_flag), ...] per text using jieba.posseg."""
    try:
        import jieba.posseg as pseg
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "jieba is required to tag Chinese for the POS map "
            "(pip install jieba)") from e
    for text in texts:
        out = []
        for word, flag in pseg.cut(text):
            if any(c.isalnum() for c in word):
                out.append((word, flag))
        yield out


def _tag_spacy(texts, lang):
    """Yield [(token, UPOS), ...] per text using spaCy."""
    try:
        import spacy
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "spaCy is required to tag eng/nld for the POS map "
            "(pip install spacy && python -m spacy download %s)"
            % SPACY_MODEL[lang]) from e
    try:
        nlp = spacy.load(SPACY_MODEL[lang], disable=["parser", "ner", "lemmatizer"])
    except OSError as e:  # pragma: no cover
        raise OSError(
            f"spaCy model {SPACY_MODEL[lang]} not installed: "
            f"python -m spacy download {SPACY_MODEL[lang]}") from e
    nlp.max_length = 4_000_000
    for doc in nlp.pipe(texts, batch_size=64):
        out = []
        for t in doc:
            if any(c.isalnum() for c in t.text):
                out.append((t.text, t.pos_))
        yield out


def _iter_tagged_from_gz(path):
    """Yield (lang, [(token, pos), ...]) from a pre-built tagged.jsonl.gz."""
    import gzip
    import json
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            yield d["language"], [(tok, pos) for tok, _s, _e, pos, _c in d["tokens"]]


def _iter_tagged_from_corpus(parquet):
    """Yield (lang, [(token, pos), ...]) by tagging the stage-1 noswitch parquet."""
    df = pd.read_parquet(parquet)
    for lang in ("eng", "nld", "zho"):
        sub = df[df["language"] == lang]
        if not len(sub):
            continue
        texts = sub["text"].tolist()
        tagger = _tag_zho(texts) if lang == "zho" else _tag_spacy(texts, lang)
        for toks in tagger:
            yield lang, toks


def build_pos_map(cfg) -> Path:
    """Build pos_map.parquet under ``out_dir`` and return its path.

    Prefers a pre-built ``data/allcontent/tagged.jsonl.gz``; otherwise tags
    ``data/curriculum_noswitch/1_intra.parquet`` inline.
    """
    out = cfg["out_dir"] / "pos_map.parquet"
    if out.exists():
        return out

    tagged_gz = cfg["data_dir"] / "allcontent" / "tagged.jsonl.gz"
    if tagged_gz.is_file():
        src = _iter_tagged_from_gz(tagged_gz)
        print(f"[pos_map] using pre-built {tagged_gz}")
    else:
        parquet = _cfg.phase_parquet(cfg, "curriculum_noswitch", "stage1")
        if not parquet.is_file():
            raise FileNotFoundError(
                f"neither {tagged_gz} nor the stage-1 monolingual corpus "
                f"{parquet} is present. Download the corpus first: "
                f"python download_data.py --only curriculum_noswitch")
        src = _iter_tagged_from_corpus(parquet)
        print(f"[pos_map] tagging {parquet} (spaCy/jieba)")

    counts = defaultdict(Counter)  # (lang, word) -> Counter(pos)
    n = 0
    for lang, toks in src:
        for tok, pos in toks:
            counts[(lang, tok.lower())][pos] += 1
        n += 1
        if n % 20000 == 0:
            print(f"  {n} docs...", flush=True)

    rows = []
    for (lang, word), ctr in counts.items():
        total = sum(ctr.values())
        pos, c = ctr.most_common(1)[0]
        rows.append({"lang": lang, "word": word, "pos": pos,
                     "purity": c / total, "n_tok": total})
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"[pos_map] docs={n} types={len(df)} -> {out}")
    return out
