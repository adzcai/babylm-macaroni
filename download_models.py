#!/usr/bin/env python3
"""Download multilingual-macaroni model branches from the HuggingFace Hub.

Usage: python download_models.py [--only switch curriculum ...] [--seeds 42 43 44]

The models live in ONE repo, `drooryck/multilingual-macaroni-models`, with each
trained model on its own branch (revision). The paper trains seeds 42-49; the Hub
holds the final checkpoints of seeds 42-44 (train the others with train.py).
Branch naming:
  - seed 42 -> `<condition>`          (bare name)
  - seed 43 -> `<condition>-s43`
  - seed 44 -> `<condition>-s44`
For each (condition, seed) this downloads the branch into
`models/<condition>[-sNN]/`. The `main` branch (card only) is never touched.

Depends only on `huggingface_hub`.
"""
import argparse
import os
import sys

# The 8 training conditions == the 8 dataset subset names.
CONDITIONS = [
    "switch",
    "noswitch",
    "word",
    "sent",
    "par",
    "salad",
    "curriculum",
    "curriculum_noswitch",
]
SEEDS = [42, 43, 44]


def revision_for(condition, seed):
    """seed 42 -> bare condition name; 43/44 -> `<condition>-sNN`."""
    return condition if seed == 42 else f"{condition}-s{seed}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="CONDITION", choices=CONDITIONS,
                    default=CONDITIONS,
                    help=f"conditions to download (default: all). choices: {', '.join(CONDITIONS)}")
    ap.add_argument("--seeds", nargs="+", type=int, metavar="SEED", choices=SEEDS,
                    default=SEEDS,
                    help="seeds to download (default: 42 43 44)")
    ap.add_argument("--models-dir", default="models",
                    help="local directory to download into (default: models)")
    ap.add_argument("--repo-id", default="drooryck/multilingual-macaroni-models",
                    help="model repo id on the Hub")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the local dir already has a config.json")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    token = os.environ.get("HF_TOKEN")
    if token:
        print("using HF_TOKEN from environment")
    else:
        print("no HF_TOKEN in environment; relying on cached login (fine for public repos)")

    print(f"repo-id    : {args.repo_id} (model)")
    print(f"models-dir : {args.models_dir}")
    print(f"conditions : {', '.join(args.only)}")
    print(f"seeds      : {', '.join(str(s) for s in args.seeds)}")

    n_done = n_skipped = 0
    for condition in args.only:
        for seed in args.seeds:
            rev = revision_for(condition, seed)
            local_dir = os.path.join(args.models_dir, rev)
            print(f"\n[{condition} seed={seed}] revision={rev} -> {local_dir}")

            if not args.force and os.path.isfile(os.path.join(local_dir, "config.json")):
                print("  skip: config.json already present (use --force to re-download)")
                n_skipped += 1
                continue

            try:
                snapshot_download(
                    repo_id=args.repo_id,
                    repo_type="model",
                    revision=rev,
                    local_dir=local_dir,
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
            print(f"  ok -> {os.path.abspath(local_dir)}")
            n_done += 1

    print(f"\nDONE. downloaded {n_done}, skipped {n_skipped}. models under "
          f"{os.path.abspath(args.models_dir)}")


if __name__ == "__main__":
    main()
