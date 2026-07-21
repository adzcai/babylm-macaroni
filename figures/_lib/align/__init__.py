"""Word-level cross-lingual alignment measurement (static ``wte`` embeddings).

Ported, path-parameterized version of the paper's ``align_static.py`` and its
three upstream lexicon builders (``align_mine_phase_intra.py``,
``align_build_lexicon.py``, ``align_build_candidates.py``). All hardcoded
``/n/.../lmkiddo`` roots are replaced by the portable ``data_dir`` / ``models_dir``
resolution in ``figures/common.py``; the baked ``MANIFEST`` of checkpoint paths is
replaced by checkpoint discovery via ``common.model_dir`` + ``common.final_ckpt``.

Public API (see submodules for full docstrings):

  Lexicon construction (each returns a DataFrame / dict of DataFrames, and can
  cache to / load from ``align_dir``):
    * ``build_corpus_stats(data_dir, ...)  -> {attested, corpus_freq, exposure}``
        mines the code-switched vs unilingual corpora (curriculum stage 1:
        ``curriculum/1_intra.parquet`` vs ``curriculum_noswitch/1_intra.parquet``).
    * ``build_candidates(data_dir, tokenizer, ...) -> {lang: DataFrame}``
        frozen per-language BLI candidate vocabularies from ``freq_{lang}.tsv``.
    * ``build_lexicon(data_dir, tokenizer, ...) -> DataFrame``
        the tiered evaluation lexicon (attested + MUSE), single-token annotated.
    * ``prepare_lexicon(data_dir, tokenizer, cache_dir, ...) -> (lex, cands,
        corpus_freq)`` — build-or-load convenience wrapper used by the figures.

  Measurement:
    * ``load_wte(ckpt_dir) -> np.ndarray`` — static input embedding matrix.
    * ``wordlevel_pairs(model_labels, lex, cands, corpus_freq, ...) -> DataFrame``
        per-pair, per-checkpoint cosine + frequency-matched null percentile
        (``pct_raw`` is the probability-of-superiority the figures average),
        with ``tier``, ``run``, ``ckpt`` columns. This is the compute half of
        Figs 4 and 6.
    * ``bli(model_labels, lex, cands, ...) -> DataFrame`` — CSLS / raw-NN
        retrieval scores (not used by Figs 4/6; kept for parity and reuse by
        the dose-response measurement).

NOTE the text-mining helpers (``build_corpus_stats``) additionally require
``opencc``, ``jieba``, and ``stopwordsiso`` (imported lazily inside that
function only); the measurement path (``wordlevel_pairs``) needs none of them.
"""
from .measure import load_wte, unit, remove_top_pcs, wordlevel_pairs, bli
from .lexicon import (
    build_corpus_stats,
    build_candidates,
    build_lexicon,
    prepare_lexicon,
)

__all__ = [
    "load_wte",
    "unit",
    "remove_top_pcs",
    "wordlevel_pairs",
    "bli",
    "build_corpus_stats",
    "build_candidates",
    "build_lexicon",
    "prepare_lexicon",
]
