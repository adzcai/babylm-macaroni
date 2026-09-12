"""Measurement code behind the paper's figures (models -> CSV), path-parameterized.

  represent_align   FLORES+ bitext retrieval per layer (Figs 2, 3, 6, 7)
  align/            lexicon + exposure-set mining from the training corpora
  traj/             word-level pipeline: matched pairs, contexts, reps, Fig 4 / Fig 8 measures
  babylm            monolingual BabyLM corpus reader + frequency tables

``measure_retrieval.py`` and ``measure_wordlevel.py`` (in ``figures/``) are the
entry points; the ``figN_*.py`` scripts only read the resulting CSVs.
"""
