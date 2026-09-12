# multilingual-macaroni

Code to reproduce **"Feeding BabyLMs Macaroni: Code-Switching Curricula Cause Cross-Lingual
Convergence"** (BabyLM 2026, multilingual track).

We train GPT-2 models from scratch on English/Dutch/Chinese under the 100M-word BabyLM budget,
on a monolingual corpus and on a synthetic **code-switched** (CS) variant of it, shuffled or as a
three-stage curriculum (word-level CS → sentence-level CS → monolingual). This repo sets up the
environment, downloads the corpora and models, trains and evaluates a model, and regenerates every
figure of the paper.

**Artifacts on the HuggingFace Hub**
- Corpora: [`drooryck/multilingual-macaroni-corpus`](https://huggingface.co/datasets/drooryck/multilingual-macaroni-corpus) (8 subsets)
- Models: [`drooryck/multilingual-macaroni-models`](https://huggingface.co/drooryck/multilingual-macaroni-models) (one branch per condition × seed; seeds 42–44 of the paper's 42–49)

The 8 conditions (= corpus subsets = model names): `switch` (CS, shuffled), `noswitch` (its
non-CS twin), `curriculum` / `curriculum_noswitch` (the same data staged as a curriculum), and the
four shuffled-ordering controls `word`, `sent`, `par`, `salad`.

## Quick start

```bash
bash setup.sh && source .venv/bin/activate        # venv, deps, pinned BabyLM evaluator
export HF_TOKEN=...                                # needed for the gated FLORES+ / BabyLM datasets

python download_data.py                            # corpora  -> data/<subset>/
python download_models.py                          # models   -> models/<condition>[-sNN]/

python train.py --corpus curriculum --seed 42      # or train one yourself (1 GPU, ~2-7 h)
python evaluate.py --model models/curriculum       # BabyLM eval suite -> out/eval/curriculum/

python figures/make_all.py                         # every data figure, from the paper's CSVs
```

All scripts take `--models-dir` / `--data-dir` / `--out` (with defaults `models/`, `data/`, `out/`).
Training, evaluation and the measurements require a GPU, but plotting does not.

## Figures

Every figure is produced with a measurement step (models → CSV) and a plotting step (CSV → PDF).
The paper's CSV files are in `results/`, so one can reproduce the plots without any model training.
(`figures/make_all.py` outputs into `out/figures/`, which you can compare with `figures_pdfs/`). To regenerate
the CSVs from models, run the measurement scripts; they overwrite `results/` unless you pass
`--results-dir`.

| Figure | Plot script | CSV in `results/` | Measurement | Models needed |
|---|---|---|---|---|
| 1 | `figures/tex/fig1_overview.tex` (TikZ) | – | – | – |
| 2 | `fig2_retrieval_per_layer.py` | `retrieval_finals.csv` | `measure_retrieval.py --set finals` | 4 main conditions, finals |
| 3 | `fig3_retrieval_trajectory.py` | `retrieval_trajectory.csv` | `measure_retrieval.py --set trajectory` | **per-stage checkpoints** of `curriculum` + `curriculum_noswitch` |
| 4 | `fig4_exposure_generalization.py` | `exposure_rank.csv` | `measure_wordlevel.py` | `curriculum` + `curriculum_noswitch`, finals |
| 5 | `figures/tex/fig5_corpus_examples.tex` (table) | – | – | – |
| 6 | `fig6_retrieval_median_rank.py` | `retrieval_finals.csv` | as Fig 2 | as Fig 2 |
| 7 | `fig7_retrieval_all_conditions.py` | `retrieval_finals.csv` + `retrieval_controls.csv` | `measure_retrieval.py --set finals controls` | all 8 conditions, finals |
| 8 | `fig8_word_divergence.py` | `word_divergence.csv` | `measure_wordlevel.py` | as Fig 4 |

Figures 1 and 5 are hand-drawn LaTeX: `bash figures/tex/build.sh` (xelatex).

**What each measurement needs beyond the corpora and models**
- Figs 2, 3, 6, 7: the gated FLORES+ dev set (accept the terms on the Hub, set `HF_TOKEN`).
- Figs 4, 8: `python download_aux.py` (the monolingual BabyLM corpora, gated, and the MUSE
  dictionaries) plus the spaCy models installed by `setup.sh`. The word-pair set is re-derived
  from these inputs, so the regenerated CSVs will use the same protocol but may not be exactly the same.
- Fig 3: the intermediate checkpoints are **not on the Hub**. `train.py` saves them
  (checkpoint-0 and epochs 1/5/10 of every stage under `models/<condition>/stage{1,2,3}/`), so
  train both curriculum arms for the seeds you want, then run `measure_retrieval.py --set trajectory`.
- Seeds: the paper averages seeds 42–49; the Hub has 42–44. `--seeds` selects what to score;
  train the rest with `train.py --seed N` if you want the full eight.

## Layout

```
setup.sh  requirements.txt  configs/{gpt2-baseline.json, train.yaml}  patches/
download_data.py  download_models.py  download_aux.py  train.py  evaluate.py  final_checkpoint.py
training/        the BabyLM baseline trainer, vendored at the paper's commit (NOTICE.md)
figures/         common.py (palette + paths), measure_*.py, figN_*.py, make_all.py, tex/, _lib/
results/         the paper's measurement CSVs (inputs to the figN scripts)
figures_pdfs/    the paper's figures 1-8 as PDFs
test/            log + report of an end-to-end trial run of this README
models/ data/ out/   gitignored: downloaded or trained artifacts and regenerated outputs
```

A model lives at `models/<condition>/` (seed 42) or `models/<condition>-s43/` etc.; every script
reads and writes models by that name. A downloaded model is a plain HF directory; a trained one
holds `checkpoint-*/` (single run) or `stage{1,2,3}/checkpoint-*/` (curriculum), and the scripts
pick the final checkpoint.

## Citation

Please cite the paper if you use this code or the artifacts. MIT License.