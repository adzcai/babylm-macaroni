#!/usr/bin/env bash
# The README workflow as actually executed for the trial run in this folder (Sep 2026,
# Harvard FASRC cluster). CPU steps ran in the foreground; GPU steps went through the
# Slurm scripts in test/slurm/. Logs are in test/logs/, the report in test/REPORT.md.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_TOKEN=...                                   # read-only token; gated FLORES+ / BabyLM

PYTHON=/usr/bin/python3.12 bash setup.sh              # -> test/logs/01_setup.log
source .venv/bin/activate
python download_data.py   --data-dir test/data        # -> test/logs/02_download_data.log
python download_models.py --models-dir test/models    # -> test/logs/03_download_models.log
python download_aux.py    --data-dir test/data        # -> test/logs/04_download_aux.log

python figures/make_all.py --out test/out/figures_from_paper_csvs   # plots from results/ (no GPU)
bash figures/tex/build.sh ../../test/out/figures_from_paper_csvs     # Figs 1 and 5

sbatch test/slurm/10_train_smoke.sbatch        # train.py, 3 conditions at 0.2% data, + Fig 3 pipeline on them
sbatch test/slurm/20_measure_retrieval.sbatch  # Figs 2/6/7 on the 24 Hub models; Fig 3 on the paper's checkpoints
sbatch test/slurm/30_measure_wordlevel.sbatch  # Figs 4/8 on the Hub models (seeds 42-44)
sbatch test/slurm/40_evaluate.sbatch           # evaluate.py on models/curriculum
