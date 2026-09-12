"""Build matched embedded/control (E/C) pairs (ports traj_build_sets.py).

E = word types embedded during the CS curriculum; C = never-embedded controls
matched 1:1 on (direction, majority POS, identical subword-token count, nearest
monolingual log-frequency within a caliper) via optimal (Hungarian) assignment.

INPUTS (under ``data_dir``, NOT produced by download_data.py — these are the
lexicon/exposure artifacts of the alignment experiment; supply them alongside
the corpus):
  - align/eval_lexicon.parquet    (translation pairs: mined + MUSE)
  - align/exposure_sets.parquet   (ever_embedded / ever_replaced per word)
  - freq_{eng,nld,zho}.tsv        (monolingual corpus frequency tables)
plus ``pos_map.parquet`` (built by pos_map.build_pos_map) under ``out_dir`` and
the shared tokenizer.json (resolved from the curriculum model).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from . import config as _cfg


def _load_freq(path):
    return {ln.split("\t")[0]: int(ln.split("\t")[1])
            for ln in open(path).read().splitlines() if "\t" in ln}


def _match_direction(E, C, caliper, freq_col):
    """Optimal 1:1 matching within (POS, token-count) blocks, caliper on log10 freq."""
    pairs = []
    for (pos, ntok), cg in C.groupby(["pos", "n_tok_bpe"]):
        eg = E[(E.pos == pos) & (E.n_tok_bpe == ntok)]
        if len(eg) == 0:
            continue
        cl = np.log10(cg[freq_col].values + 1.0)
        el = np.log10(eg[freq_col].values + 1.0)
        cost = np.abs(cl[:, None] - el[None, :])
        cost[cost > caliper] = 1e6  # forbid out-of-caliper matches
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            if cost[r, c] >= 1e6:
                continue
            pairs.append({
                "E_word": eg.iloc[c].src_word, "C_word": cg.iloc[r].src_word,
                "equiv_E": eg.iloc[c].tgt_word, "equiv_C": cg.iloc[r].tgt_word,
                "src_lang": eg.iloc[c].src_lang, "tgt_lang": eg.iloc[c].tgt_lang,
                "pos": pos, "n_tok_bpe": ntok,
                "E_freq": eg.iloc[c][freq_col], "C_freq": cg.iloc[r][freq_col],
                "E_ortho": eg.iloc[c].ortho_jaccard, "C_ortho": cg.iloc[r].ortho_jaccard,
                "E_freq_intra": eg.iloc[c].src_freq_intra,
                "E_freq_noswitch": eg.iloc[c].src_freq_noswitch,
                "C_freq_intra": cg.iloc[r].src_freq_intra,
                "C_freq_noswitch": cg.iloc[r].src_freq_noswitch,
                "dlogf": abs(cl[r] - el[c]),
            })
    return pairs


def _require(path, what):
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"{what} not found at {path}. This is an alignment-experiment input "
            f"not covered by download_data.py; supply it under data/ before "
            f"running measure_wordlevel.py (it builds them via _lib.align.prepare_lexicon).")
    return path


def build_measurement_pairs(cfg, freq_col: str = "mono_freq") -> Path:
    """Build measurement_pairs_<freq_col>.parquet under out_dir; return its path.

    ``freq_col`` in {"mono_freq" (primary), "src_freq_intra" (exposure-confound
    test)}. The trajectory figure uses the mono_freq set.
    """
    from tokenizers import Tokenizer

    out = cfg["out_dir"] / f"measurement_pairs_{freq_col}.parquet"
    if out.exists():
        return out

    K = cfg["k_contexts"]
    tok = Tokenizer.from_file(str(_cfg.tokenizer_json(cfg)))

    align = cfg["data_dir"] / "align"
    lex = pd.read_parquet(_require(align / "eval_lexicon.parquet", "eval_lexicon.parquet"))
    exp = pd.read_parquet(_require(align / "exposure_sets.parquet", "exposure_sets.parquet"))
    pos_map = pd.read_parquet(cfg["out_dir"] / "pos_map.parquet")
    pos_map = pos_map[pos_map.purity >= cfg["pos_purity_min"]]
    pos_of = {(l, w): p for l, w, p in
              zip(pos_map.lang, pos_map.word, pos_map.pos)}

    emb = set(zip(exp.loc[exp.ever_embedded, "word"], exp.loc[exp.ever_embedded, "lang"]))
    rep = set(zip(exp.loc[exp.ever_replaced, "word"], exp.loc[exp.ever_replaced, "lang"]))
    F = {l: _load_freq(_require(cfg["data_dir"] / f"freq_{l}.tsv", f"freq_{l}.tsv"))
         for l in ("eng", "nld", "zho")}

    lex = lex.copy()

    def n_bpe(word, lang):
        # canonical convention: leading-space form for eng/nld, bare for zho
        return len(tok.encode(word if lang == "zho" else " " + word).ids)

    lex["mono_freq"] = [F[l].get(w, 0) for w, l in zip(lex.src_word, lex.src_lang)]
    lex["tgt_mono_freq"] = [F[l].get(w, 0) for w, l in zip(lex.tgt_word, lex.tgt_lang)]
    lex["n_tok_bpe"] = [n_bpe(w, l) for w, l in zip(lex.src_word, lex.src_lang)]
    lex["pos"] = [pos_of.get((l, w)) for w, l in zip(lex.src_word, lex.src_lang)]

    # both members need enough corpus occurrences to supply K contexts
    lex = lex[(lex.mono_freq >= K) & (lex.tgt_mono_freq >= K) & lex.pos.notna()]
    # deterministic equivalent: highest-frequency tgt per (src_lang,src_word,tgt_lang)
    lex = (lex.sort_values("tgt_mono_freq", ascending=False)
              .drop_duplicates(["src_lang", "src_word", "tgt_lang"]))

    key = list(zip(lex.src_word, lex.src_lang))
    lex["is_E"] = [k in emb for k in key]
    lex["is_C"] = [(k not in emb) and (k not in rep) for k in key]

    all_pairs = []
    for (sl, tl), g in lex.groupby(["src_lang", "tgt_lang"]):
        E, C = g[g.is_E], g[g.is_C]
        if len(C) == 0:
            print(f"  {sl}->{tl}: E={len(E)} C=0 -> UNUSABLE (no never-embedded controls)")
            continue
        p = _match_direction(E, C, cfg["caliper_log10"], freq_col)
        print(f"  {sl}->{tl}: E={len(E)} C={len(C)} -> matched {len(p)}")
        all_pairs.extend(p)

    df = pd.DataFrame(all_pairs)
    df.insert(0, "pair_id", [f"p{i:05d}" for i in range(len(df))])
    df.to_parquet(out, index=False)
    print(f"[build_sets] TOTAL matched pairs: {len(df)} -> {out}")
    return out
