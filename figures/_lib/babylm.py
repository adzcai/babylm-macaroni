"""The monolingual BabyBabelLM corpora (eng/nld/zho): text streaming + frequency tables.

The word-level analyses (Figs 4 and 8) draw their measurement sentences and their
monolingual word frequencies from the ORIGINAL monolingual BabyLM corpora
(``BabyLM-community/babylm-{eng,nld,zho}``), not from the code-switched training
corpora. ``download_aux.py`` fetches them into ``data/babylm/<lang>/``; this
module streams their ``text`` column (parquet as published on the Hub, or the
``*.arrow`` shards of a ``datasets`` cache) and builds ``freq_{lang}.tsv``.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

LANGS = ("eng", "nld", "zho")
_WORD = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)


def babylm_texts(babylm_dir, lang):
    """Yield every training document of ``lang`` under ``babylm_dir/<lang>/``."""
    root = Path(babylm_dir) / lang
    parquets = sorted(root.rglob("*.parquet"))
    arrows = sorted(root.rglob(f"babylm-{lang}-train-*.arrow"))
    if parquets:
        import pyarrow.parquet as pq
        for p in parquets:
            pf = pq.ParquetFile(p)
            for batch in pf.iter_batches(columns=["text"], batch_size=4096):
                for t in batch.column("text"):
                    yield t.as_py()
    elif arrows:
        import pyarrow as pa
        import pyarrow.ipc as ipc
        for p in arrows:
            with pa.memory_map(str(p), "r") as src:
                for batch in ipc.open_stream(src):
                    for t in batch.column("text"):
                        yield t.as_py()
    else:
        raise FileNotFoundError(
            f"no BabyLM {lang} corpus under {root} (expected *.parquet or "
            f"babylm-{lang}-train-*.arrow). Fetch it: python download_aux.py --babylm")


def count_monolingual_freq(babylm_dir, lang, out):
    """Write ``out`` (word<TAB>count, descending) over the whole ``lang`` corpus.

    Verbatim port of the paper's count_monolingual_freq.py: eng/nld use a
    Unicode word regex; zho is normalised to Simplified and segmented with the
    hybrid jieba tokenizer.
    """
    from .align.text import to_simplified, tokenize as tokenize_mixed
    counts = Counter()
    n_docs = 0
    for text in babylm_texts(babylm_dir, lang):
        n_docs += 1
        if lang == "zho":
            toks = tokenize_mixed(to_simplified(text))
        else:
            toks = _WORD.findall(text)
        counts.update(t.lower() for t in toks)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for w, c in counts.most_common():
            f.write(f"{w}\t{c}\n")
    print(f"[freq] {lang}: {n_docs} docs, {len(counts)} types -> {out}")
    return out


def ensure_freq_tables(babylm_dir, freq_dir):
    """Build any missing ``freq_{lang}.tsv`` under ``freq_dir``; return their paths."""
    paths = {}
    for lang in LANGS:
        p = Path(freq_dir) / f"freq_{lang}.tsv"
        if not p.is_file():
            count_monolingual_freq(babylm_dir, lang, p)
        paths[lang] = p
    return paths
