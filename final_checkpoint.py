#!/usr/bin/env python3
"""Return the "final" checkpoint directory under a training output dir.

Selection key is (epoch, step, mtime), all descending. The step and mtime
tie-breakers matter when a directory contains checkpoints from more than one
run at the same epoch count: the earlier code broke such ties by arbitrary
glob order, which silently carried a *stale* previous-run checkpoint forward
across curriculum stages (see scripts/run_cscl_curriculum.py). A curriculum
run should also clean its stage dirs first (--fresh) so this never has to
disambiguate across runs, but the robust key here is a second line of defence.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _step(ckpt: Path) -> int:
    m = re.search(r"checkpoint-(\d+)$", ckpt.name)
    return int(m.group(1)) if m else -1


def final_checkpoint(output_dir: str | Path) -> Path | None:
    best: Path | None = None
    best_key: tuple[float, int, float] | None = None
    root = Path(output_dir)
    if not root.is_dir():
        return None
    for ckpt in root.glob("checkpoint-*"):
        state_path = ckpt / "trainer_state.json"
        if not state_path.is_file():
            continue
        epoch = float(json.loads(state_path.read_text()).get("epoch", -1))
        key = (epoch, _step(ckpt), ckpt.stat().st_mtime)
        if best_key is None or key > best_key:
            best_key = key
            best = ckpt
    return best


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} OUTPUT_DIR", file=sys.stderr)
        sys.exit(2)
    ckpt = final_checkpoint(sys.argv[1])
    if ckpt is None:
        sys.exit(1)
    print(ckpt)


if __name__ == "__main__":
    main()
