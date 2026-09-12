#!/bin/bash
#SBATCH --job-name=mac-wordlevel
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h200
#SBATCH --gres=gpu:1 --cpus-per-task=16 --mem=96G --time=08:00:00
#SBATCH --output=test/logs/30_measure_wordlevel_%j.log
# Figs 4/8 measurement on the downloaded final checkpoints, seeds 42-44: lexicon mining,
# frequency tables, POS map, matched pairs, contexts, extraction, both measures.
source test/slurm/_header.sh
python figures/measure_wordlevel.py --seeds 42 43 44 --models-dir test/models --data-dir test/data --results-dir test/out/results --out test/out
python figures/make_all.py --results-dir test/out/results --out test/out/figures_regenerated --only fig4_exposure_generalization fig8_word_divergence
echo DONE
