# Vendored trainer

`train.py`, `data.py`, `tokenizer.py`, `base_train.yaml`, `base_tokenizer.yaml` and
`README_upstream.md` are copied verbatim from the BabyLM multilingual baseline
trainer, https://github.com/babylm-org/multilingual-training, at the fork commit the
paper trained with (`llam11/multilingual-training` @ `fec6db2c65b0ffd9b05cee62c8fbf0b6a7c6975f`,
"Merge pull request #21 from llam11/update-base-train"). The fork adds document packing
(concatenate documents with end-of-text separators, then split at the context length),
an explicit epoch save schedule (`--save_intervals`), an initial `checkpoint-0`, and
resume support; nothing here was modified.

The repo-root `train.py` is a thin wrapper that fills a YAML config per condition and
calls `training/train.py --config <yaml>`. The trainer reports to Weights & Biases;
the wrapper sets `WANDB_MODE=offline` unless you configure it otherwise.
