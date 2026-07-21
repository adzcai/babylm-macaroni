# multilingual-macaroni

Reproducibility code for **"Feeding BabyLMs Macaroni: Investigating Code-Switched Curriculum
Learning in Data-Constrained Multilingual Modelling"** (BabyLM 2026, Multilingual track).

We train GPT-2 BabyLMs from scratch on synthetic **code-switched** English/Dutch/Chinese under a
100M-word budget, and find a small downstream boost from a code-switching **curriculum** (word-level
→ sentence-level → monolingual) over a shuffled ordering. This repo lets you set up the environment,
download the corpora, train a model (or pull ours), evaluate it, and regenerate every figure.

**Operating principle:** given the *data* you can train the models; given a *model in the right
place* you can evaluate it and reproduce the figures. Dataset-*synthesis* code is out of scope — the
8 corpora are consumed as finished artifacts from the Hub.

## Artifacts (already public on HuggingFace)
- Corpora — [`drooryck/multilingual-macaroni-corpus`](https://huggingface.co/datasets/drooryck/multilingual-macaroni-corpus) (8 subsets)
- Models — [`drooryck/multilingual-macaroni-models`](https://huggingface.co/drooryck/multilingual-macaroni-models) (24 models on branches: 8 conditions × seeds 42/43/44)

The 8 conditions: `switch` (full CS, shuffled), `noswitch` (monolingual twin), `word` / `sent` /
`par` / `salad` (switch-type ablations), `curriculum` (= the submission `babylm-macaroni`; same data
as `switch`, staged), `curriculum_noswitch` (monolingual staged twin).

---

## The five commands

```bash
# (1) SET UP — venv, deps, and clone+pin the external trainer & evaluator
bash setup.sh
source .venv/bin/activate

# (2) DOWNLOAD CORPORA -> data/<subset>/
python download_data.py                        # all 8; or --only switch curriculum ...

# (3) TRAIN one condition -> models/<condition>[-sNN]/ ...
python train.py --corpus curriculum --seed 42  # curriculum = 3-stage carry-forward
python train.py --corpus switch     --seed 42  # single-run condition
#     ... OR skip training and pull ours:
python download_models.py                       # all -> models/; or --only curriculum switch

# (4) EVALUATE a model (zero-shot + finetune incl. POS) -> out/eval/<label>/
python evaluate.py --model models/curriculum

# (5) REPRODUCE FIGURES (each: models -> CSV -> plot, one command)
python figures/fig02_05_retrieval_per_layer.py --models-dir models --out out/figures
python figures/make_all.py --models-dir models --data-dir data --out out/figures
```

All scripts take `--models-dir` / `--data-dir` / `--out` (defaults `models` / `data` / `out`), also
settable via `MACARONI_MODELS_DIR` / `MACARONI_DATA_DIR` / `MACARONI_OUT`.

---

## Layout

```
setup.sh  requirements.txt  configs/{gpt2-baseline.json, train.yaml}
download_data.py  download_models.py  train.py  evaluate.py  final_checkpoint.py
figures/
  common.py            # palette + path/model resolution (imported by every figNN)
  _lib/                # ported, path-parameterized measurement code
  figNN_*.py           # one per paper figure: load models -> compute CSV -> plot PNG/PDF
  make_all.py          # run every figure
models/  data/  out/   # gitignored: downloaded/trained artifacts + regenerable CSVs/figures
MODEL_CARD.md  DATASET_CARD.md
```

A model lives at `models/<condition>/` (seed 42) or `models/<condition>-s43/` etc. Every figure and
`evaluate.py` reads models by that name; training writes them there.

## Figures ↔ paper

| Script | Paper figure(s) | Needs |
|---|---|---|
| `fig02_05_retrieval_per_layer.py` | 2, 5 (bitext retrieval P@1 per layer) | published finals + FLORES+ |
| `fig03_retrieval_trajectory.py` | 3 (retrieval over the curriculum) | ⚠️ trajectory checkpoints |
| `fig04_06_wordlevel.py` | 4, 6 (word-level alignment) | finals (4); ⚠️ trajectory (6) |
| `fig07_dd_trajectory.py` | 7 (embedded-word DiD over training) | ⚠️ trajectory checkpoints |
| `fig08_repr_divergence.py` | 8 (how far CS moves a word) | finals + monolingual corpora (see Data) |
| `fig09_embedded_own.py` | 9 (embedded vs own-language) | finals + monolingual corpora |
| `fig10_grid_advantage.py` | 10 (context-invariance grid) | finals + monolingual corpora + Exp-3 pairs |
| `fig11_dose_response.py` | 11 (dose-response, rising part only) | `noswitch` + `word` finals; maximal point not reproducible |

Figure 1 is a TikZ schematic (`figures/fig1_directions.tex`), not a data figure.

---

## Requirements & caveats

- **GPU** for training, evaluation, and the figure measurements (representation extraction).
- **HuggingFace token** (`export HF_TOKEN=...`): the retrieval figures (2, 3, 5) probe alignment on
  **FLORES+** (`openlanguagedata/flores_plus`), which is **gated** — accept its terms on the Hub and
  set `HF_TOKEN` before running them.
- **spaCy models** for the POS-based figures: `python -m spacy download en_core_web_sm nl_core_news_sm`.

### Data the figures need beyond `download_data.py` / `download_models.py`
The 8 corpus subsets + 24 final models reproduce Figs **2 and 5** directly (given FLORES+). The
analysis figures (4, 6, 7, 8, 9, 10, 11) additionally read a few inputs that are **not** on the two
Hub repos. Each script fails with a clear message naming the exact missing file — nothing is
fabricated. In three tiers:

1. **Re-derivable from what you download** (the ported builders regenerate these on demand):
   frequency tables `freq_{lang}.tsv`, the attested lexicon / corpus-frequency / exposure sets
   (from `curriculum/1_intra` vs `curriculum_noswitch/1_intra`), and the Fig-7 POS map.
2. **Public, but a separate download:** the **monolingual BabyLM corpora** — needed for the
   "own-language" contexts of Figs 8/9/10 and Fig-7 context sampling. Fetch with
   `python download_babylm.py` (→ `data/babylm/<lang>/`; confirm the repo ids via `--repo-template`),
   then pass `--babylm-dir data/babylm` to those figures.
3. **Author-supplied (not published):** the MUSE bilingual dictionaries (`data/muse/`, for the full
   tiered eval lexicon — without them the lexicon degrades to attested-only), the Exp-3 matched-pair
   table (`measurement_pairs_mono_freq.parquet`, Fig 10), and any pre-computed
   `eval_lexicon`/`exposure_sets` if you want byte-identical inputs.

### ⚠️ Trajectory figures (3, 6, 7)
These need *per-checkpoint* models for **both** curriculum arms, but only the CS-curriculum arm's
milestones are published (as `chck_*M` on `drooryck/babylm-macaroni`); the no-CS arm's per-checkpoint
checkpoints are **not** on HF. To reproduce them, re-train `curriculum`/`curriculum_noswitch` saving
milestone checkpoints (see `train.py --help`) or obtain the intermediates from the authors. Fig 4's
bars still render without them (only Fig 6's trajectory is skipped).

### Notes
- **Single-seed vs 3-seed error bars:** the paper averaged 3 seeds. The static-alignment figures
  (4, 6, 11) run per `--seed`; run them across seeds 42/43/44 and pool for the paper's mean±SD (or
  accept single-seed point estimates without seed-variance bands).
- **Fig 11 dose-response — partial by design.** The paper's Fig 11 is an inverted-U over three doses
  (none → sparse → **maximal**), showing alignment peaks at *sparse* switching. The maximal point
  (`allcontent`) came from a bespoke maximal-relexification corpus/model produced by the
  corpus-synthesis pipeline, which is out of scope here — and it is **not** among the published
  artifacts, so it is **not reproducible** from this repo (and no public condition faithfully stands
  in for it). This repo therefore reproduces only the doses backed by public models — the rising part
  of the curve — with an exact mapping `noswitch→noswitch`, `intra→word`. That shows sparse switching
  improves alignment over none, but **not** the maximal-density turnover. If you train an
  `allcontent`-style model, add it as a third dose with `--dose <key>=<condition>` to recover the full
  inverted-U. Because it is partial, Fig 11 is **opt-in** in `make_all.py` — run it explicitly with
  `python figures/fig11_dose_response.py ...` or `python figures/make_all.py --all`.
- **Pins:** `setup.sh` clones `multilingual-training@fec6db2c…` (trainer) and
  `babylm-eval@6f825c29…` (evaluator). The trainer commit is the current tip of `main` on the
  `llam11/multilingual-training` fork (the canonical source for it — it is not on the `babylm-org`
  upstream). Override the source with `TRAINING_URL=... TRAINING_REF=<sha> bash setup.sh` if needed.

## Citation

If you use this code or the artifacts, please cite the paper.
Released under the MIT License.
