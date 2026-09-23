#!/usr/bin/env python
"""`python scripts/batch_process.py ...` without an install.

Installed, the same thing is the `hypnose-batch-process` command; the code lives in
`hypnose_behavior.cli.batch_process`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hypnose_behavior.cli.batch_process import main

if __name__ == "__main__":
    raise SystemExit(main())
