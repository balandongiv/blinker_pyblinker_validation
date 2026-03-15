"""Run the canonical fresh PyBlinker-vs-Blinker validation for murat_2018."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation.run_murat_full_with_status import main as run_main  # noqa: E402


DEFAULT_ARGS = [
    "--prefix",
    "exp06",
    "--selection",
    "top",
    "--n",
    "74",
    "--force-rerun",
]


def main(argv: list[str] | None = None) -> int:
    resolved_argv = sys.argv[1:] if argv is None else argv
    return run_main(resolved_argv or DEFAULT_ARGS)


if __name__ == "__main__":
    raise SystemExit(main())
