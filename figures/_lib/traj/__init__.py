"""Word-level measurement pipeline (Figs 4 and 8), ported from the paper's ``traj_*`` scripts.

Per seed:  pos_map -> build_sets -> sample_contexts -> extract -> exposure_rank / divergence

  * ``pos_map``          type -> majority part of speech, tagged from the monolingual
                         stage-1 corpus (spaCy for eng/nld, jieba for zho)
  * ``build_sets``       seen-embedded (E) vs never-embedded (C) source words, matched 1:1
                         on POS, subword-token count and log10 monolingual frequency
                         (Hungarian assignment, caliper 0.1), each with its translation
  * ``sample_contexts``  K=20 monolingual BabyLM sentences per word, frozen + hashed
  * ``extract``          per-(word, layer) mean contextual representation per checkpoint
  * ``exposure_rank``    Fig 4: median percentile rank of the translation, CS vs non-CS
  * ``divergence``       Fig 8: per-layer cosine between the two models' word representations

Inputs beyond ``download_data.py``: the monolingual BabyLM corpora + MUSE
dictionaries (``download_aux.py``); the lexicon/exposure sets are mined from the
downloaded corpora by ``_lib.align``.
"""
from .config import (ARM_CONDITION, CKPT_ORDER, CheckpointsMissing, build_config,
                     discover_checkpoints, final_checkpoints)
from .pos_map import build_pos_map
from .build_sets import build_measurement_pairs
from .sample_contexts import sample_contexts
from .extract import extract_reps
from .exposure_rank import exposure_rank
from .divergence import divergence_layerwise

__all__ = ["ARM_CONDITION", "CKPT_ORDER", "CheckpointsMissing", "build_config",
           "discover_checkpoints", "final_checkpoints", "build_pos_map",
           "build_measurement_pairs", "sample_contexts", "extract_reps",
           "exposure_rank", "divergence_layerwise"]
