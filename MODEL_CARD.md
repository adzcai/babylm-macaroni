---
language: [en, nl, zh]
license: mit
library_name: transformers
tags:
  - babylm
  - babylm-2026
  - multilingual
  - code-switching
  - gpt2
---

# multilingual-macaroni-models

24 GPT-2 models (~98M params each) trained from scratch for the **BabyLM 2026 Multilingual track**
(English, Dutch, Chinese) under a **100M byte-premium-adjusted-word** budget — one model per
**condition × seed** (8 conditions × seeds 42/43/44).

All conditions share the same architecture, tokenizer, budget, and recipe; they differ only in
which corpus they train on (see the [dataset card](DATASET_CARD.md)). The two headline conditions
are `switch` (full code-switching, shuffled) and `curriculum` (the *same data*, staged as
intrasentential → sentence-level → monolingual) — the paper's core order comparison. The `curriculum`
weights are identical to the leaderboard submission `drooryck/babylm-macaroni`.

## Repository layout (branches / revisions)
One monolithic HF repo, models on branches. `main` = card only.
- `<condition>`          — seed 42 (e.g. `curriculum`, `switch`, `word`)
- `<condition>-s43`, `-s44` — seeds 43, 44
- conditions: `switch noswitch word sent par salad curriculum curriculum_noswitch`

```python
from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained("drooryck/multilingual-macaroni-models", revision="curriculum-s43")
```

Per-checkpoint (learning-curve) revisions `chck_*M` exist **only** for the CS-curriculum arm, on the
separate `drooryck/babylm-macaroni` repo.

## Architecture & recipe
- GPT-2, 12 layers, 768 hidden, 12 heads (~98M params), 1024 context.
- Tokenizer: shared byte-level BPE, vocabulary 16,384 (BabyLM baseline tokenizer).
- LR 5e-5, cosine-with-min-lr, warmup 0.01, batch 16, 10 epochs (per stage for the curriculum arms).

See the paper *Feeding BabyLMs Macaroni: Investigating Code-Switched Curriculum Learning in
Data-Constrained Multilingual Modelling* for method, controls, and evaluation.
