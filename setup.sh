#!/usr/bin/env bash
# (1) SET UP — create a venv, install deps, clone+pin the BabyLM evaluator.
#
#   bash setup.sh
#
# Overridable via env:
#   PYTHON            python to build the venv with           (default: python3)
#   TORCH_INDEX_URL   CUDA wheel index for torch              (default: cu124)
#   EVAL_URL / EVAL_REF           evaluation repo + commit
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

# The evaluator, pinned to the commit used for the paper (the trainer is vendored
# in training/, see training/NOTICE.md).
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
echo ">> installing spaCy models (POS tagging for the word-level figures)"
python -m spacy download en_core_web_sm
python -m spacy download nl_core_news_sm

# --- 1b. clone + pin the evaluator -----------------------------------------
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
    echo "       Confirm the commit and re-run, e.g.:  EVAL_REF=<sha> bash setup.sh" >&2
    exit 1
  fi
}

clone_pin "$EVAL_URL" "$EVAL_REF" "babylm-eval"
# POS fine-tuning fix: newer `datasets` load UD `upos` as strings; cast back to the
# 17-tag ClassLabel (otherwise the POS task fails). Idempotent.
if git -C babylm-eval apply --check ../patches/babylm-eval-pos-upos-classlabel.patch 2>/dev/null; then
  echo ">> patching babylm-eval (UD upos ClassLabel)"
  git -C babylm-eval apply ../patches/babylm-eval-pos-upos-classlabel.patch
fi

echo
echo ">> setup complete. Activate with:  source .venv/bin/activate"
echo ">> next:  python download_data.py && python download_models.py   (see README)"
