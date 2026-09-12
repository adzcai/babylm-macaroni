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

GPT-2 models (~98M params each) trained from scratch for the **BabyLM 2026 Multilingual track**
(English, Dutch, Chinese) under a **100M byte-premium-adjusted-word** budget — one model per
**condition × seed**. The paper trains 8 conditions × 8 seeds (42–49); the branches here are
seeds 42/43/44 of each condition (24 models).

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

Word-count milestone revisions `chck_*M` exist only for the CS-curriculum condition (seed 42), on the
separate `drooryck/babylm-macaroni` repo.

## Architecture & recipe
- GPT-2, 12 layers, 768 hidden, 12 heads (~98M params), 1024 context.
- Tokenizer: shared byte-level BPE, vocabulary 16,384 (BabyLM baseline tokenizer).
- Adam (no weight decay), LR 5e-5, 1% warmup then cosine decay to 0, batch 16, 10 epochs (per stage for the curriculum conditions; the LR schedule restarts each stage).
- Only final checkpoints are published; intermediate (per-stage) checkpoints can be regenerated with the training code.

See the paper *Feeding BabyLMs Macaroni: Code-Switching Curricula Cause Cross-Lingual
Convergence* for method, controls, and evaluation; code at github.com/drooryck/multilingual-macaroni.
