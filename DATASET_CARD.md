---
license: mit
language: [en, nl, zh]
tags: [babylm, babylm-2026, multilingual, code-switching]
pretty_name: multilingual-macaroni corpus
---

# multilingual-macaroni-corpus

The 8 training corpora for the BabyLM 2026 Multilingual-track paper. All are derived from the
official BabyLM 2026 multilingual corpora (`babylm-{eng,nld,zho}`) — **no external text** — and
synthesized into code-switched form with an instruction-tuned LLM. Each corpus is held to the same
**100M byte-premium-adjusted-word** budget, split equally across English, Dutch, and Chinese.

Each corpus trains exactly one model condition (× 8 seeds in the paper). `switch` vs `curriculum` are the **same
data in a different training order** (the paper's core comparison); `word/sent/par/salad` are the
four controls.

## The 8 subsets
| Subset | Files | Model condition | What it is |
|---|---|---|---|
| `switch`     | `data.parquet` | `switch`     | Full mixed code-switching, **shuffled** order |
| `noswitch`   | `data.parquet` | `noswitch`   | Monolingual twin of `switch`, shuffled |
| `word`       | `data.parquet` | `word`       | **Intrasentential (word-level)** CS only |
| `sent`       | `data.parquet` | `sent`       | **Sentence-level** CS only |
| `par`        | `data.parquet` | `par`        | **Document-translation** control (a translation appended to a third of documents) |
| `salad`      | `data.parquet` | `salad`      | **Word-salad** control (random words replaced by embedded-language words) |
| `curriculum` | `{1_intra,2_sentence,3_mono}.parquet` | `curriculum` | Same CS data as `switch`, staged as a curriculum |
| `curriculum_noswitch` | `{1_intra,2_sentence,3_mono}.parquet` | `curriculum_noswitch` | Monolingual twin, staged |

`switch` is the concatenation of the `curriculum` phases in random order (identical data, different
order); likewise for the no-CS pair. Each row is one document with `text`, `language` (matrix),
`cs_type`, `embedded_lang`, `switched`, and `num-tokens`.

Published at [`drooryck/multilingual-macaroni-corpus`](https://huggingface.co/datasets/drooryck/multilingual-macaroni-corpus).
See the paper for the generation pipeline, quality gate, and budget accounting.
