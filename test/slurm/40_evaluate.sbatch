#!/bin/bash
#SBATCH --job-name=mac-eval
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h200
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=06:00:00
#SBATCH --output=test/logs/40_evaluate_%j.log
# The BabyLM evaluation suite (zero-shot + fine-tuning + POS) on the downloaded curriculum model.
source test/slurm/_header.sh
python evaluate.py --model test/models/curriculum --out test/out --tasks finetune   # zero-shot done in the first run
python evaluate.py --model test/models/curriculum --out test/out --tasks pos
echo DONE
