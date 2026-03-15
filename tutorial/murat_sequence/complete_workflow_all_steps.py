"""End-to-end runner for the canonical Murat 2018 reproducibility workflow.

This orchestration reflects the public workflow used in this repository:

1. ``step0_download_dataset.py`` (optional for a clean full reset)
2. ``step1_prepare_dataset.py``
3. ``step2_run_blinker.py``
4. ``step3_validate_pyblinker.py``

The older comparison and visualization scripts are retained under
``tutorial/murat_sequence/legacy`` for archival reference only. They are no
longer the canonical validation path.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path


DATASET_ROOT = Path("D:/dataset/murat_2018")
os.environ.setdefault("MURAT_DATASET_ROOT", str(DATASET_ROOT))

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tutorial.murat_sequence import (  # noqa: E402
    step0_download_dataset,
    step1_prepare_dataset,
    step2_run_blinker,
    step3_validate_pyblinker,
)


LOGGER = logging.getLogger(__name__)


def _ensure_root_exists(root: Path) -> None:
    if root.exists():
        LOGGER.info("Using existing dataset directory: %s", root)
    else:
        LOGGER.info("Creating dataset directory: %s", root)
        root.mkdir(parents=True, exist_ok=True)


def _run_step(name: str, argv: Sequence[str], runner: Callable[[list[str] | None], int]) -> None:
    display_args = " ".join(argv)
    LOGGER.info("Starting %s with arguments: %s", name, display_args or "<none>")
    result = runner(list(argv) if argv else None)
    if result != 0:
        raise RuntimeError(f"{name} failed with exit code {result}")
    LOGGER.info("%s completed successfully", name)


def run_workflow(
    *,
    download_first: bool = False,
    force_prepare: bool = False,
    force_blinker: bool = False,
    validation_args: Sequence[str] | None = None,
) -> None:
    _ensure_root_exists(DATASET_ROOT)

    if download_first:
        _run_step(
            "step0_download_dataset",
            ["--root", str(DATASET_ROOT), "--limit", "-1"],
            step0_download_dataset.main,
        )

    step1_args = ["--root", str(DATASET_ROOT), "--channels", "CH1", "CH2"]
    if force_prepare:
        step1_args.append("--force")
    _run_step("step1_prepare_dataset", step1_args, step1_prepare_dataset.main)

    step2_args = ["--root", str(DATASET_ROOT)]
    if force_blinker:
        step2_args.append("--force")
    _run_step("step2_run_blinker", step2_args, step2_run_blinker.main)

    _run_step(
        "step3_validate_pyblinker",
        list(validation_args or []),
        step3_validate_pyblinker.main,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-first", action="store_true", help="Include the dataset download step.")
    parser.add_argument("--force-prepare", action="store_true", help="Recreate FIF/EDF files.")
    parser.add_argument("--force-blinker", action="store_true", help="Overwrite MATLAB Blinker outputs.")
    parser.add_argument(
        "--validation-arg",
        action="append",
        default=None,
        help="Extra argument to pass through to step3_validate_pyblinker. Repeat as needed.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run_workflow(
            download_first=args.download_first,
            force_prepare=args.force_prepare,
            force_blinker=args.force_blinker,
            validation_args=args.validation_arg,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Workflow failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
