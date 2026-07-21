#!/usr/bin/env bash
# (1) SET UP — create a venv, install deps, and clone+pin the two external repos.
#
#   bash setup.sh
#
# Overridable via env:
#   PYTHON            python to build the venv with           (default: python3)
#   TORCH_INDEX_URL   CUDA wheel index for torch              (default: cu124)
#   TRAINING_URL / TRAINING_REF   trainer repo + commit
#   EVAL_URL / EVAL_REF           evaluation repo + commit
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

# External repos, pinned to the commits used for the paper.
TRAINING_URL="${TRAINING_URL:-https://github.com/llam11/multilingual-training.git}"
TRAINING_REF="${TRAINING_REF:-fec6db2c65b0ffd9b05cee62c8fbf0b6a7c6975f}"
EVAL_URL="${EVAL_URL:-https://github.com/babylm-org/babylm-eval.git}"
EVAL_REF="${EVAL_REF:-6f825c291e2c4c78ad33b1935fd64d45f52642dc}"

# --- 1a. venv + python deps -------------------------------------------------
if [ ! -d .venv ]; then
  echo ">> creating .venv"
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

echo ">> installing torch from $TORCH_INDEX_URL"
pip install torch --index-url "$TORCH_INDEX_URL"

echo ">> installing requirements"
pip install -r requirements.txt

# --- 1b. clone + pin external repos ----------------------------------------
clone_pin () {  # $1=url  $2=ref  $3=dir
  local url="$1" ref="$2" dir="$3"
  if [ ! -d "$dir/.git" ]; then
    echo ">> cloning $url -> $dir"
    git clone "$url" "$dir"
  fi
  echo ">> pinning $dir @ $ref"
  git -C "$dir" fetch --all --tags --quiet || true
  if ! git -C "$dir" checkout --quiet "$ref"; then
    echo "ERROR: commit $ref not found in $url ($dir)." >&2
    echo "       See README 'Pins' — confirm the correct commit and re-run, e.g.:" >&2
    echo "       TRAINING_REF=<sha> bash setup.sh" >&2
    exit 1
  fi
}

clone_pin "$TRAINING_URL" "$TRAINING_REF" "multilingual-training"
clone_pin "$EVAL_URL"     "$EVAL_REF"     "babylm-eval"

echo
echo ">> setup complete. Activate with:  source .venv/bin/activate"
echo ">> next:  python download_data.py   (and/or)   python download_models.py"
