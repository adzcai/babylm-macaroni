#!/usr/bin/env python3
"""Download the monolingual BabyLM 2026 corpora (English/Dutch/Chinese) -> data/babylm/<lang>/.

Several analysis figures (8, 9, 10, and Fig-7 context sampling) need the *original
monolingual* BabyLM corpora — for "own-language" contexts and frequency counting —
in addition to the code-switched subsets fetched by ``download_data.py``. This
helper pulls them and lays them out so the figure scripts' ``--babylm-dir`` can
point at ``data/babylm``.

    python download_babylm.py                       # all three languages
    python download_babylm.py --only eng nld        # subset
    python download_babylm.py --babylm-dir data/babylm

Repo ids follow ``--repo-template`` (default ``BabyLM-community/babylm-{lang}``);
override it if the official ids differ. Reads HF_TOKEN from env if present.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

LANGS = ["eng", "nld", "zho"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=LANGS, default=LANGS,
                    help="languages to fetch (default: all three)")
    ap.add_argument("--babylm-dir", default="data/babylm",
                    help="destination root (default: data/babylm)")
    ap.add_argument("--repo-template", default="BabyLM-community/babylm-{lang}",
                    help="HF dataset id template, {lang} in {eng,nld,zho} "
                         "(default: BabyLM-community/babylm-{lang}) — confirm the official ids")
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download  # lazy import

    token = os.environ.get("HF_TOKEN")
    dest_root = Path(args.babylm_dir)
    for lang in args.only:
        repo_id = args.repo_template.format(lang=lang)
        dest = dest_root / lang
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[{lang}] {repo_id} -> {dest}")
        try:
            snapshot_download(repo_id=repo_id, repo_type="dataset", revision=args.revision,
                              local_dir=str(dest), local_dir_use_symlinks=False, token=token)
        except Exception as e:  # noqa: BLE001
            print(f"  ! failed: {e}")
            if "401" in str(e) or "gated" in str(e).lower():
                print("    (gated/private? set HF_TOKEN or `huggingface-cli login`)")
            print(f"    (confirm the repo id — override with --repo-template)")
    print("done.")


if __name__ == "__main__":
    main()
