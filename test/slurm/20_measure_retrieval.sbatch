#!/bin/bash
#SBATCH --job-name=mac-retrieval
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h200
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00
#SBATCH --output=test/logs/20_measure_retrieval_%j.log
# Figs 2/6/7 measurement on the 24 downloaded final checkpoints (seeds 42-44), and the
# Fig 3 measurement on the paper's real per-stage checkpoints (seed 42, symlinked into
# test/models_traj/ in the layout train.py writes).
source test/slurm/_header.sh
python figures/measure_retrieval.py --set finals   --seeds 42 43 44 --models-dir test/models --results-dir test/out/results --emb-cache test/out/emb_cache
python figures/measure_retrieval.py --set controls --seeds 42 43 44 --models-dir test/models --results-dir test/out/results --emb-cache test/out/emb_cache
python figures/measure_retrieval.py --set trajectory --seeds 42 --models-dir test/models_traj --results-dir test/out/results --emb-cache test/out/emb_cache
python figures/make_all.py --results-dir test/out/results --out test/out/figures_regenerated --only fig2_retrieval_per_layer fig3_retrieval_trajectory fig6_retrieval_median_rank fig7_retrieval_all_conditions
echo DONE
