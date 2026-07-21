#!/usr/bin/env python3
"""Download the multilingual-macaroni training corpora from the HuggingFace Hub.

Usage: python download_data.py [--only switch noswitch ...] [--data-dir data]

Fetches the 8 corpus subsets of `drooryck/multilingual-macaroni-corpus` into
`data/<subset>/...`, preserving the on-Hub file layout:
  - single-run corpora  -> data/<subset>/data.parquet
  - curriculum corpora  -> data/<subset>/{1_intra,2_sentence,3_mono}.parquet

Depends only on `huggingface_hub` (no `datasets` required).
"""
import argparse
import os
import sys

# The 8 corpus subsets published in the dataset repo.
SUBSETS = [
    "switch",
    "noswitch",
    "word",
    "sent",
    "par",
    "salad",
    "curriculum",
    "curriculum_noswitch",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="SUBSET", choices=SUBSETS,
                    default=SUBSETS,
                    help=f"subsets to download (default: all). choices: {', '.join(SUBSETS)}")
    ap.add_argument("--data-dir", default="data",
                    help="local directory to download into (default: data)")
    ap.add_argument("--repo-id", default="drooryck/multilingual-macaroni-corpus",
                    help="dataset repo id on the Hub")
    ap.add_argument("--revision", default="main",
                    help="dataset revision/branch/tag to download (default: main)")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    token = os.environ.get("HF_TOKEN")
    if token:
        print("using HF_TOKEN from environment")
    else:
        print("no HF_TOKEN in environment; relying on cached login (fine for public repos)")

    subsets = args.only
    print(f"repo-id  : {args.repo_id} (dataset)")
    print(f"revision : {args.revision}")
    print(f"data-dir : {args.data_dir}")
    print(f"subsets  : {', '.join(subsets)}")

    for subset in subsets:
        print(f"\n[{subset}] downloading {subset}/* ...")
        try:
            local = snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                revision=args.revision,
                allow_patterns=[f"{subset}/*"],
                local_dir=args.data_dir,
                local_dir_use_symlinks=False,
                token=token,
            )
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 401:
                print("  ERROR 401 Unauthorized. This repo may require authentication.",
                      file=sys.stderr)
                print("  Set HF_TOKEN=<your token> or run `huggingface-cli login`, then retry.",
                      file=sys.stderr)
            raise
        print(f"  ok -> {os.path.join(local, subset)}")

    print(f"\nDONE. corpora under {os.path.abspath(args.data_dir)}")


if __name__ == "__main__":
    main()
