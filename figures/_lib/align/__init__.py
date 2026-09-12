"""Lexicon mining for the word-level analyses (ported from the paper's align_* scripts).

  * ``build_corpus_stats(data_dir)``  diff the code-switched vs non-CS stage-1 corpora
    (``curriculum/1_intra.parquet`` vs ``curriculum_noswitch/1_intra.parquet``) ->
    ``attested_lexicon`` (single-word embedded<->replaced pairs), ``corpus_freq``,
    ``exposure_sets`` (which words were ever embedded / ever replaced).
  * ``build_lexicon(data_dir, tokenizer, ...)``  the evaluation lexicon: attested pairs
    + the MUSE en-nl / nl-en / en-zh / zh-en dictionaries (``data/muse/``), normalised
    and annotated.
  * ``prepare_lexicon(data_dir, tokenizer, cache_dir)``  build-or-load wrapper.

Needs ``opencc``, ``jieba``, ``stopwordsiso`` (see requirements.txt).
"""
from .lexicon import build_corpus_stats, build_lexicon, prepare_lexicon

__all__ = ["build_corpus_stats", "build_lexicon", "prepare_lexicon"]
