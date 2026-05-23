"""Run the canonical fresh PyBlinker-vs-Blinker validation for driving_dataset."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation.fresh_compare_subjects import DRIVING_SUBJECTS, main as run_main


DEFAULT_ARGS = [
    "--dataset",
    "driving_dataset",
    "--prefix",
    "drvexp01",
    "--subjects",
    ",".join(DRIVING_SUBJECTS),
    "--restrict-py-to-comparison-channels",
    "--continue-on-failure",
    "--force-rerun",
]


def main(argv: list[str] | None = None) -> int:
    resolved_argv = sys.argv[1:] if argv is None else argv
    return run_main(resolved_argv or DEFAULT_ARGS)


if __name__ == "__main__":
    raise SystemExit(main())
