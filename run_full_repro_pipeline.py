"""Clickable wrapper for the full end-to-end validation reproduction pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation.run_full_repro_pipeline import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
