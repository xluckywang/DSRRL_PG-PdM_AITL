#!/usr/bin/env python
"""Exp10 v4 entry point with mandatory formal Exp7-DSHI trajectories."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
DSHI_TAG = "exp10_v4_formal_dshi"


def main() -> None:
    os.environ["EXP7_DSHI_CACHE_TAG"] = DSHI_TAG
    os.environ["EXP7_DSHI_STRICT"] = "1"
    if "--out-dir" not in sys.argv:
        sys.argv.extend(["--out-dir", str(THIS_DIR / "outputs" / "exp10_v4_dshi_formal")])
    runpy.run_path(str(THIS_DIR / "10_exp10_v3_formal_agent_loop.py"), run_name="__main__")


if __name__ == "__main__":
    main()
