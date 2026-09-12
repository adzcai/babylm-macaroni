# Common preamble for the trial-run Slurm jobs (sourced by each .sbatch).
set -euo pipefail
REPO=/n/netscratch/kdbrantley_lab/Lab/adzcai/multilingual/multilingual-macaroni
cd "$REPO"
source .venv/bin/activate
export HF_HOME="$REPO/test/hf-cache"          # keep the gated datasets' cache local to the trial
export TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "python: $(which python)  $(python --version)"
