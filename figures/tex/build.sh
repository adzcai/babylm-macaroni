#!/usr/bin/env bash
# Build the two hand-drawn paper figures (Fig 1, Fig 5) as standalone PDFs.
# Needs xelatex with fontspec, tikz, booktabs and a Han-capable font
# (Droid Sans Fallback or Noto Sans CJK SC).
#
#   bash figures/tex/build.sh [OUT_DIR]      (default OUT_DIR = out/figures)
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-../../out/figures}"
mkdir -p "$OUT"
for f in fig1_overview fig5_corpus_examples; do
  xelatex -interaction=nonstopmode -halt-on-error "$f.tex" > "$f.log.txt" 2>&1 \
    || { echo "xelatex failed for $f (see figures/tex/$f.log.txt)"; exit 1; }
  mv "$f.pdf" "$OUT/$f.pdf"
  rm -f "$f.aux" "$f.log"
  echo "wrote $OUT/$f.pdf"
done
