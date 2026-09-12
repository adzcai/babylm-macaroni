#!/usr/bin/env python3
"""Download the auxiliary inputs of the word-level analyses (Figs 4 and 8).

    python download_aux.py             # both of the below
    python download_aux.py --babylm    # monolingual BabyBabelLM corpora -> data/babylm/<lang>/
    python download_aux.py --muse      # MUSE bilingual dictionaries    -> data/muse/

* The monolingual corpora `BabyLM-community/babylm-{eng,nld,zho}` at the
  revisions the paper used (b78a9336 / 1aa063f7 / 600a6657) supply the
  measurement sentences and the word-frequency tables. They are gated on the
  Hub: accept their terms and set HF_TOKEN (or `huggingface-cli login`).
* The MUSE en-nl / nl-en / en-zh / zh-en dictionaries (Conneau et al. 2018)
  extend the mined translation lexicon. Public, ~1.5 MB each.

Nothing here is needed for training, evaluation, or the retrieval figures.
"""
import argparse
import os
import urllib.request
from pathlib import Path

BABYLM = {"eng": "b78a9336aacf2d876aeb6d289869191443f8d41f",
          "nld": "1aa063f7e59b84090489bcab6ca1cfa2b911188d",
          "zho": "600a6657f55eabbea5bc56041ef6b8a33440501f"}
MUSE_URL = "https://dl.fbaipublicfiles.com/arrival/dictionaries/{pair}.txt"
MUSE_PAIRS = ["en-nl", "nl-en", "en-zh", "zh-en"]


def download_babylm(data_dir, langs):
    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN")
    for lang in langs:
        dest = Path(data_dir) / "babylm" / lang
        if list(dest.rglob("*.parquet")):
            print(f"[babylm] {lang}: already present under {dest}")
            continue
        repo = f"BabyLM-community/babylm-{lang}"
        print(f"[babylm] {repo}@{BABYLM[lang][:8]} -> {dest}")
        snapshot_download(repo_id=repo, repo_type="dataset", revision=BABYLM[lang],
                          allow_patterns=["data/*.parquet"], local_dir=str(dest), token=token)


def download_muse(data_dir):
    dest = Path(data_dir) / "muse"
    dest.mkdir(parents=True, exist_ok=True)
    for pair in MUSE_PAIRS:
        out = dest / f"{pair}.txt"
        if out.is_file():
            print(f"[muse] {out} already present")
            continue
        print(f"[muse] {MUSE_URL.format(pair=pair)} -> {out}")
        urllib.request.urlretrieve(MUSE_URL.format(pair=pair), out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--babylm", action="store_true")
    ap.add_argument("--muse", action="store_true")
    ap.add_argument("--only", nargs="+", choices=list(BABYLM), default=list(BABYLM),
                    help="BabyLM languages to fetch")
    args = ap.parse_args()
    if not (args.babylm or args.muse):
        args.babylm = args.muse = True
    if args.babylm:
        download_babylm(args.data_dir, args.only)
    if args.muse:
        download_muse(args.data_dir)
    print("done.")


if __name__ == "__main__":
    main()
