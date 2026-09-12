#!/bin/bash
#SBATCH --job-name=mac-train-smoke
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h200
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=03:00:00
#SBATCH --output=test/logs/10_train_smoke_%j.log
# Smoke-scale training of BOTH curriculum arms + one single-run condition, following
# the README workflow but on 0.2% of each corpus, so the whole thing takes minutes.
# Verifies: data staging, the vendored trainer, 3-stage carry-forward, checkpoint layout.
source test/slurm/_header.sh
python train.py --corpus curriculum          --seed 42 --data-dir test/data --models-dir test/models_smoke --epochs 10 --dataset-fraction 0.002
python train.py --corpus curriculum_noswitch --seed 42 --data-dir test/data --models-dir test/models_smoke --epochs 10 --dataset-fraction 0.002
python train.py --corpus switch              --seed 42 --data-dir test/data --models-dir test/models_smoke --epochs 10 --dataset-fraction 0.002
echo "--- checkpoint layout ---"; find test/models_smoke -maxdepth 3 -name "checkpoint-*" | sort
# The smoke trajectories feed the trajectory measurement end to end (values are meaningless).
python figures/measure_retrieval.py --set trajectory --seeds 42 --models-dir test/models_smoke --results-dir test/out/results_smoke
python figures/fig3_retrieval_trajectory.py --results-dir test/out/results_smoke --out test/out/figures_smoke
echo DONE
