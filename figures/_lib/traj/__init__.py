"""Ported alignment-trajectory (DiD) pipeline for Fig 7.

This package is a path-parameterized port of the paper's ``traj_*`` scripts
(shared config via ``traj_common.py``). It measures how the cross-lingual
alignment of *embedded* words (E) relative to matched never-embedded controls (C)
evolves over training, and contrasts the code-switched curriculum arm (``cs`` ->
condition ``curriculum``) against its monolingual twin (``noswitch`` ->
``curriculum_noswitch``) as a difference-in-differences.

Pipeline (per seed):  pos_map -> build_sets -> sample_contexts -> extract ->
measure -> stats.  ``dd_trajectory`` runs it for a set of seeds and pools.

    >>> from _lib import traj
    >>> df = traj.dd_trajectory(models_dir, data_dir, out_dir, seeds=[42, 43, 44])

============================ IMPORTANT: DATA GAPS ============================
This figure needs the model at TEN points along training for BOTH arms (init,
then epochs 1/5/10 within each of the 3 curriculum stages). The published Hub
repo ships only each condition's FINAL checkpoint, and the no-CS arm's
intermediate checkpoints are NOT on the Hub at all. If they are absent,
``extract`` raises ``config.CheckpointsMissing`` with instructions (re-train each
arm saving milestone checkpoints, or supply the intermediates under
``models/<condition>[-sNN]/<stage>/checkpoint-<step>/``).

Two further inputs are not produced by ``download_data.py``:
  * ``pos_map`` is derived on the fly by tagging the downloaded monolingual
    corpus (``data/curriculum_noswitch/1_intra.parquet``) — so the paper's
    private ``data/allcontent/tagged.jsonl.gz`` is NOT required (it is used if
    present, else re-derived). Needs spaCy + jieba (see pos_map.py).
  * ``build_sets`` needs the alignment-experiment artifacts
    ``data/align/{eval_lexicon,exposure_sets}.parquet`` and
    ``data/freq_{eng,nld,zho}.tsv``. These are supplied alongside the corpus;
    a clear FileNotFoundError names any that is missing.
=============================================================================
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .config import (ARM_CONDITION, CKPT_ORDER, CheckpointsMissing, build_config,
                     discover_checkpoints)
from .pos_map import build_pos_map
from .build_sets import build_measurement_pairs
from .sample_contexts import sample_contexts
from .extract import extract_reps
from .measure import measure
from .stats import stats_trajectory

__all__ = ["build_config", "discover_checkpoints", "CheckpointsMissing",
           "ARM_CONDITION", "CKPT_ORDER", "run_seed", "dd_trajectory"]


def run_seed(models_dir, data_dir, out_dir, seed, n_layers=12,
             layer_aggregate=None) -> Path:
    """Run the full pipeline for ONE seed; return its stats_trajectory.csv path.

    ``out_dir`` here is the per-seed working directory (intermediates + stats).
    """
    cfg = build_config(models_dir, data_dir, out_dir, seed,
                       n_layers=n_layers, layer_aggregate=layer_aggregate)
    build_pos_map(cfg)
    pairs = build_measurement_pairs(cfg, freq_col="mono_freq")
    sample_contexts(cfg, pairs)
    extract_reps(cfg)               # raises CheckpointsMissing if models absent
    measure(cfg, pairs)
    return stats_trajectory(cfg, cfg["out_dir"] / "measurements.csv")


def dd_trajectory(models_dir, data_dir, out_dir, seeds=(42, 43, 44),
                  n_layers=12, layer_aggregate=None) -> pd.DataFrame:
    """Run the trajectory pipeline for each seed and pool the results.

    Parameters
    ----------
    models_dir, data_dir : path-like
        Repo ``models/`` and ``data/`` roots.
    out_dir : path-like
        Root for pipeline intermediates; each seed uses ``out_dir/traj_s<seed>/``.
    seeds : iterable[int]
        Model seeds to run and pool (default 42, 43, 44).

    Returns
    -------
    pandas.DataFrame
        The concatenation of every seed's stats_trajectory rows (schema
        ``metric, estimand, stage, epoch, mean, ci_lo, ci_hi, n_pairs``) with an
        added ``seed`` column — a superset of the stats_trajectory schema that
        preserves per-seed values so the plot can show the 3-seed mean +- 1 SD
        band. Rows are computed once and cached per seed under
        ``out_dir/traj_s<seed>/stats_trajectory.csv``.
    """
    out_root = Path(out_dir)
    frames = []
    for seed in seeds:
        seed_out = out_root / f"traj_s{seed}"
        stats_csv = run_seed(models_dir, data_dir, seed_out, seed,
                             n_layers=n_layers, layer_aggregate=layer_aggregate)
        frames.append(pd.read_csv(stats_csv).assign(seed=int(seed)))
    return pd.concat(frames, ignore_index=True)
