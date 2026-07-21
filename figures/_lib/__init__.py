"""Ported, path-parameterized measurement modules for the figure pipeline.

Each ``figNN_*.py`` script (in the parent ``figures/`` directory) imports the
compute half of its figure from here, runs it over ``models/`` + ``data/`` to
produce an intermediate CSV under ``out/figures/``, then plots it. These modules
are adapted from the paper's analysis code with all hardcoded ``/n/...`` paths
replaced by the ``models_dir`` / ``data_dir`` / ``out`` roots resolved in
``figures/common.py``.
"""
