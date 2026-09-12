#!/bin/bash
#SBATCH --job-name=mac-retrieval2
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h200
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00
#SBATCH --output=test/logs/21_measure_retrieval_rerun_%j.log
# Re-run of 20_measure_retrieval after pinning transformers<5 (the first run, on
# transformers 5.17, differed from the paper's numbers by ~2 pp mean P@1).
source test/slurm/_header.sh
python -c "import transformers; print('transformers', transformers.__version__)"
rm -rf test/out/emb_cache
python figures/measure_retrieval.py --set finals controls --seeds 42 43 44 --models-dir test/models --results-dir test/out/results --emb-cache test/out/emb_cache
python figures/measure_retrieval.py --set trajectory --seeds 42 --models-dir test/models_traj --results-dir test/out/results --emb-cache test/out/emb_cache
python figures/make_all.py --results-dir test/out/results --out test/out/figures_regenerated --only fig2_retrieval_per_layer fig3_retrieval_trajectory fig6_retrieval_median_rank fig7_retrieval_all_conditions
echo DONE
