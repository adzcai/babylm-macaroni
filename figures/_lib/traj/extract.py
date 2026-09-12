"""Per-checkpoint representation extraction (ports traj_extract.py).

Keeps the proven internals verbatim: offset-overlap span selection, ``np.take``
pooling, float64 accumulation, one fresh forward pass per checkpoint, frozen
measurement set (sha256 asserted identical everywhere). fp32 deliberately.

The checkpoint LIST comes from ``config.discover_checkpoints`` (on-disk discovery
under ``model_dir(condition, seed)``) instead of the paper's hard-coded lmkiddo
step list. The per-word buffer is sized from the model's ACTUAL hidden-state
count at runtime (n_transformer_layers + 1), so ``cfg['n_layers']`` is advisory.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from . import config as _cfg


def _load_contexts(paths):
    by_word = defaultdict(list)
    for p in paths:
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                by_word[d["word"]].append(d)
    return by_word


def _extract(model, tokenizer, by_word, words, device, batch_size):
    import torch
    dim = getattr(model.config, "n_embd", None) or model.config.hidden_size
    widx = {w: i for i, w in enumerate(words)}
    flat = [(w, c["sentence"], c["char_start"], c["char_end"])
            for w in words for c in by_word[w]]
    acc = None
    cnt = np.zeros(len(words), dtype=np.int64)

    with torch.no_grad():
        for lo in range(0, len(flat), batch_size):
            chunk = flat[lo:lo + batch_size]
            enc = tokenizer([c[1] for c in chunk], return_offsets_mapping=True,
                            return_tensors="pt", padding=True, truncation=True,
                            max_length=256)
            offsets = enc.pop("offset_mapping")
            attn = enc["attention_mask"]
            out = model(**{k: v.to(device) for k, v in enc.items()},
                        output_hidden_states=True)
            hs = torch.stack(out.hidden_states, dim=1).to(torch.float32).cpu().numpy()
            if acc is None:  # size layer axis from actual hidden-state count
                n_hidden = hs.shape[1]
                acc = np.zeros((len(words), n_hidden, dim), dtype=np.float64)
            for b, (w, _s, cs, ce) in enumerate(chunk):
                offs = offsets[b].numpy(); mask = attn[b].numpy().astype(bool)
                sel = [j for j in range(len(offs))
                       if mask[j] and offs[j][1] > offs[j][0]
                       and offs[j][0] < ce and offs[j][1] > cs]
                if not sel:
                    continue
                acc[widx[w]] += hs[b].take(sel, axis=1).mean(axis=1)
                cnt[widx[w]] += 1
    return (acc / np.maximum(cnt, 1)[:, None, None]).astype(np.float32), cnt


def extract_reps(cfg, ckpt_list=None, device="cuda"):
    """Run a fresh forward pass at every checkpoint in ``ckpt_list`` -> reps_*.npz.

    Writes one ``reps_{arm}_{stage}_ep{epoch}.npz`` per checkpoint under out_dir.
    ``ckpt_list`` defaults to ``config.final_checkpoints(cfg)`` (the two final
    checkpoints Figs 4 and 8 need); pass ``config.discover_checkpoints(cfg)`` for
    the full trajectory.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ctx_paths = [cfg["out_dir"] / f"contexts_{l}.jsonl" for l in ("eng", "nld", "zho")]
    set_hash = _cfg.sha256_strings([_cfg.sha256_file(p) for p in ctx_paths])
    by_word = _load_contexts(ctx_paths)
    words = sorted(by_word)
    word_hash = _cfg.sha256_strings(words)
    print(f"[extract] measurement set: {len(words)} words, "
          f"{sum(len(v) for v in by_word.values())} contexts "
          f"(set {set_hash[:16]} words {word_hash[:16]})")

    todo = ckpt_list if ckpt_list is not None else _cfg.final_checkpoints(cfg)

    tokenizer = AutoTokenizer.from_pretrained(str(todo[0][3]))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ref_cnt = None
    for arm, stage, epoch, ckpt in todo:
        out = cfg["out_dir"] / f"reps_{arm}_{stage}_ep{epoch}.npz"
        if out.exists():
            print(f"[extract] skip (exists) {out.name}"); continue
        print(f"[extract] === {arm} {stage} ep{epoch} ({ckpt}) ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt), output_hidden_states=True).to(device).eval()
        vecs, cnt = _extract(model, tokenizer, by_word, words, device, cfg["batch_size"])
        # span selection is weight-independent -> context counts MUST be identical
        if ref_cnt is None:
            ref_cnt = cnt
        else:
            assert np.array_equal(cnt, ref_cnt), "context counts differ across checkpoints!"
        np.savez_compressed(out, words=np.array(words), vecs=vecs, counts=cnt,
                            set_hash=set_hash, word_hash=word_hash,
                            arm=arm, stage=stage, epoch=epoch)
        print(f"[extract]  -> {out.name}  words with >=1 ctx: {(cnt>0).sum()}/{len(words)}")
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
