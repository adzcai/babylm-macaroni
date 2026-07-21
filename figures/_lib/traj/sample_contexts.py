"""Sample the frozen measurement context set (ports traj_sample_contexts.py).

For every word in the matched pairs (E, C, and BOTH translation equivalents),
sample K monolingual sentences from that word's OWN language, recording exact
char offsets. The set is persisted once and sha256'd so every checkpoint measures
the byte-identical set.

CORPUS REPOINT. The paper read raw BabyLM ``*.arrow`` shards from a hard-coded HF
cache. Here we stream the monolingual ``text`` column from the downloaded corpus
instead: the ``noswitch`` / ``curriculum_noswitch`` subsets ARE the monolingual
twin (no code-switching), filtered by their ``language`` column. Prefers
``data/noswitch/data.parquet`` (single file, all 3 languages), falling back to the
staged ``data/curriculum_noswitch/{1_intra,2_sentence,3_mono}.parquet``.

CAVEAT (documented, not fixed): these are training sentences the models saw. The
E-vs-C contrast is matched so exposure is symmetric; the cs-vs-noswitch DiD nets
out the rest. Because the corpus source differs from the paper's arrow shards, the
sampled set (and its hash) will differ from the paper's — the pipeline is
internally consistent, not byte-identical to the private run.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import config as _cfg
from .text import to_simplified, tokenize as mixed_tokenize

_SENT = {"zho": re.compile(r"[^。！？\n]+[。！？]?"),
         "lat": re.compile(r"[^.!?\n]+[.!?]?")}


def _mono_parquets(cfg):
    """Ordered list of monolingual corpus parquets to stream text from."""
    single = cfg["data_dir"] / "noswitch" / "data.parquet"
    if single.is_file():
        return [single]
    staged = [_cfg.phase_parquet(cfg, "curriculum_noswitch", s)
              for s in ("stage1", "stage2", "stage3")]
    staged = [p for p in staged if p.is_file()]
    if not staged:
        raise FileNotFoundError(
            "no monolingual corpus found. Download it first: "
            "python download_data.py --only noswitch curriculum_noswitch")
    return staged


def _mono_texts(cfg, lang):
    """Yield monolingual ``text`` strings for ``lang`` from the corpus parquets."""
    for p in _mono_parquets(cfg):
        df = pd.read_parquet(p, columns=["text", "language"])
        for t in df.loc[df.language == lang, "text"]:
            yield t


def _sample_lang(cfg, pairs, lang):
    """Sample contexts for one language; write contexts_<lang>.jsonl; return path."""
    K = cfg["k_contexts"]
    need = set()
    for r in pairs.itertuples(index=False):
        need.add((r.E_word, r.src_lang)); need.add((r.C_word, r.src_lang))
        need.add((r.equiv_E, r.tgt_lang)); need.add((r.equiv_C, r.tgt_lang))
    words = sorted(w for w, l in need if l == lang)
    surf = {to_simplified(w): w for w in words} if lang == "zho" else {w: w for w in words}
    targets = set(surf)
    remaining = {w: K for w in words}
    collected = defaultdict(list)
    n_active = len(remaining)
    print(f"[contexts] lang={lang} words={len(words)} K={K}", flush=True)

    sp = _SENT["zho"] if lang == "zho" else _SENT["lat"]
    for text in _mono_texts(cfg, lang):
        if n_active == 0:
            break
        norm = to_simplified(text) if lang == "zho" else text
        for m in sp.finditer(norm):
            s = m.group().strip()
            if not (cfg["min_sent_chars"] <= len(s) <= cfg["max_sent_chars"]):
                continue
            hits = {t.lower() for t in mixed_tokenize(s)} & targets
            for sf in hits:
                w = surf[sf]
                if remaining[w] <= 0:
                    continue
                if lang == "zho":
                    pos = s.find(sf)
                else:
                    mm = re.search(r"\b" + re.escape(sf) + r"\b", s, re.IGNORECASE)
                    pos = mm.start() if mm else -1
                if pos < 0:
                    continue
                collected[w].append({"word": f"{lang}:{w}", "lang": lang,
                                     "is_embedded": False, "sentence": s,
                                     "char_start": pos, "char_end": pos + len(sf)})
                remaining[w] -= 1
                if remaining[w] == 0:
                    n_active -= 1

    out = cfg["out_dir"] / f"contexts_{lang}.jsonl"
    n = 0
    with open(out, "w") as f:
        for w in words:                       # deterministic order -> stable hash
            for r in collected[w]:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
                n += 1
    got = sum(1 for w in words if collected[w])
    full = sum(1 for w in words if len(collected[w]) >= K)
    print(f"[contexts] wrote {n} -> {out} | >=1 ctx: {got}/{len(words)} | full K: {full}")
    print(f"           sha256={_cfg.sha256_file(out)[:16]}")
    return out


def sample_contexts(cfg, pairs_path=None):
    """Sample+freeze the measurement set for all 3 languages; return the 3 paths."""
    pairs = pd.read_parquet(
        pairs_path or cfg["out_dir"] / "measurement_pairs_mono_freq.parquet")
    paths = []
    for lang in ("eng", "nld", "zho"):
        out = cfg["out_dir"] / f"contexts_{lang}.jsonl"
        if out.exists():
            print(f"[contexts] skip (exists) {out.name}")
            paths.append(out)
            continue
        paths.append(_sample_lang(cfg, pairs, lang))
    return paths
