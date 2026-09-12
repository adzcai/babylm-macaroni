# Trial run of the README workflow

Run on 12 Sep 2026 on the Harvard FASRC cluster (login node for CPU steps, one H200 per
Slurm job for GPU steps), from a fresh venv, exactly as `test/run_test.sh` records. Logs are in
`test/logs/`, the Slurm scripts in `test/slurm/`, outputs under `test/out/` (large artifacts
are gitignored; the regenerated figures and CSVs are kept).

## What was run and what happened

| Step | Command | Result |
|---|---|---|
| Setup | `PYTHON=/usr/bin/python3.12 bash setup.sh` | ok after two fixes (below): venv, torch cu124, deps, spaCy models, evaluator clone + POS patch |
| Corpora | `download_data.py --data-dir test/data` | ok, 8 subsets |
| Models | `download_models.py --models-dir test/models` | ok, 24 branches (5.9 GB) |
| Aux inputs | `download_aux.py --data-dir test/data` | ok, BabyLM eng/nld/zho (966 MB) + 4 MUSE dictionaries |
| Plot from paper CSVs | `figures/make_all.py` + `figures/tex/build.sh` | ok; Figs 2, 3, 4, 6, 7 pixel-identical to `figures_pdfs/` at 100 dpi; Fig 8 identical to a fresh run of the paper's own script (see note) |
| Smoke training | `train.py --corpus curriculum / curriculum_noswitch / switch --epochs 10 --dataset-fraction 0.002` | ok: 3-stage carry-forward, `checkpoint-0` + epochs 1/5/10 per stage in the layout the figures read |
| Fig 3 pipeline on the smoke models | `measure_retrieval.py --set trajectory --models-dir test/models_smoke` | ok end to end (numbers meaningless by construction) |
| Retrieval measurement | `measure_retrieval.py --set finals controls` on the 24 Hub models | ok; **P@1 identical to `results/` for every model, layer and pair** |
| Fig 3 measurement on the paper's checkpoints | `measure_retrieval.py --set trajectory --models-dir test/models_traj` (seed 42, the real per-stage checkpoints symlinked into the `train.py` layout) | ok; identical to `results/retrieval_trajectory.csv` |
| Word-level measurement (Figs 4, 8) | `measure_wordlevel.py --seeds 42 43 44` on the Hub models (lexicon mining, frequency tables, POS map, matched pairs, contexts, extraction, both measures; ~1.5 h, one GPU) | ok; 1,057 matched pairs vs the paper's 1,102. Three of the four directions (en→nl, en→zh, zh→en) reproduce the paper's pair counts and Fig 4 medians **exactly**; nl→en has 183 instead of 245 pairs (the paper's Dutch POS map came from an older corpus build, so a few Dutch types get a different majority tag). Pooled retention 66.1 % vs 64.7 %. Fig 8 curves agree to the third decimal. |
| Evaluation | `evaluate.py --model test/models/curriculum` (zero-shot, 22 fine-tuning tasks, POS) | ok; single-seed accuracies within the paper's Table 3 ranges for CS/Curr (e.g. BLiMP-NL 76.9 vs 76.8±0.6, ZhoBLiMP 74.9 vs 74.8±0.4, SIB-200 en/nl/zh 75.0/71.0/76.5 vs 73.6/70.6/78.6, POS en/nl/zh 92.9/94.8/90.1 vs 92.8/94.2/90.1) |

## Problems found and fixed during the trial

1. **The trainer fork is private.** `setup.sh` used to clone `llam11/multilingual-training` at
   the paper's commit; GitHub returns 404 for it, and the commit is not on the public
   `babylm-org/multilingual-training`. The trainer is now vendored in `training/` (NOTICE.md).
2. **The evaluator pin.** The POS fix the paper's numbers depend on (UD `upos` cast back to a
   ClassLabel) exists only as a local commit, not on `babylm-org/babylm-eval`; pinning it broke
   `setup.sh`. The pin is back on the public commit and the fix ships as a patch that `setup.sh` applies.
3. **transformers 5.x.** A fresh install pulled transformers 5.17. The trainer fails on it
   (`TrainingArguments(warmup_ratio=...)` was removed), and the retrieval probe gives different
   numbers on it (mean |ΔP@1| ≈ 2 pp vs the paper, already at the embedding layer, i.e. a
   tokenisation/padding change). `requirements.txt` now pins `transformers<5`; on 4.57 the
   regenerated CSVs match the paper exactly.
4. **No tokenizer without a downloaded model.** Training in an empty `models/` failed because
   `train.py` looked for a tokenizer inside a downloaded model. It now fetches the paper's shared
   tokenizer from the Hub when none is found locally.
5. **W&B login demanded by the fine-tuning scripts.** The evaluator's HF-Trainer scripts pick up
   an installed wandb and stop for a login; `evaluate.py` now sets `WANDB_DISABLED=true` for its
   subprocesses (training already ran W&B offline).
6. Slurm here rejects multi-partition GPU submissions; the trial scripts use one partition. One
   word-level job also died with a bus error after 35 minutes (a netscratch file-reaping symptom);
   the lexicon builder now reuses already-mined corpus statistics so a retry is cheap.

## Notes

- `figures_pdfs/fig8_word_divergence_by_layer.pdf` is the PDF committed in the paper repo. It
  differs in 0.3 % of pixels from what the paper's own generator produces from the current
  `divergence_layerwise.csv` files (and from what this repo produces, which is identical to that
  fresh run). The paper's committed PDF is therefore very slightly stale relative to its CSVs.
- The regenerated retrieval CSVs differ from the paper's only in `cos_gap` (≤ 0.005), which uses
  a random derangement drawn from an RNG shared across the models scored in one run, so it
  depends on the scoring order; every retrieval and rank metric is identical.
