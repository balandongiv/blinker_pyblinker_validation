"""End-to-end runner for the Murat 2018 processing pipeline.

This script orchestrates the five individual workflow steps provided in this
repository:

1. :mod:`murat_sequence.step1_prepare_dataset`
2. :mod:`murat_sequence.step2_pyblinker`
3. :mod:`murat_sequence.step3_run_blinker`
4. :mod:`murat_sequence.step4_compare_`
5. :mod:`murat_sequence.step5_create_ground_truth`

The workflow uses ``D:/dataset/murat_2018`` as the canonical storage location
for both the downloaded dataset and any derived outputs (FIF/EDF files,
pyblinker/blinker results, reports, …). The environment variable
``MURAT_DATASET_ROOT`` is set automatically so that each step resolves its
``download_root``/``raw_downsampled`` paths to that directory.

Unlike the standalone ``step1`` script, which defaults to processing only the
``CH1`` and ``CH2`` channels, the orchestration mirrors that behaviour by
requesting the explicit channel list for those two channels.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Configure the shared dataset location *before* importing any of the step
# modules. They read the ``MURAT_DATASET_ROOT`` environment variable during
# import time to determine their default paths.
DATASET_ROOT = Path("D:/dataset/murat_2018")
os.environ.setdefault("MURAT_DATASET_ROOT", str(DATASET_ROOT))

from murat_sequence import (  # noqa: E402 - import depends on env var above
    step1_prepare_dataset,
    step2_pyblinker,
    step3_run_blinker,
    step4_compare_,
    step5_create_ground_truth,
)

LOGGER = logging.getLogger(__name__)


def _ensure_root_exists(root: Path) -> None:
    """Create ``root`` if required and warn when it already existed."""

    if root.exists():
        LOGGER.info("Using existing dataset directory: %s", root)
    else:
        LOGGER.info("Creating dataset directory: %s", root)
        root.mkdir(parents=True, exist_ok=True)


def _run_step(name: str, argv: Sequence[str], runner: Callable[[list[str] | None], int]) -> None:
    """Execute ``runner`` (a CLI-style ``main`` function) with ``argv``."""

    display_args = " ".join(argv)
    LOGGER.info("Starting %s with arguments: %s", name, display_args or "<none>")
    result = runner(list(argv) if argv else None)
    if result != 0:
        raise RuntimeError(f"{name} failed with exit code {result}")
    LOGGER.info("%s completed successfully", name)


@dataclass
class WorkflowOptions:
    """Runtime configuration shared by all workflow stages."""

    dataset_root: Path
    force_step2: bool = False
    force_step3: bool = False


@dataclass
class WorkflowStage:
    """Description of an individual workflow stage."""

    name: str
    runner: Callable[[list[str] | None], int]
    build_args: Callable[[WorkflowOptions], Sequence[str]]
    enabled: bool = True


def _stage1_args(options: WorkflowOptions) -> list[str]:
    return [
        "--root",
        str(options.dataset_root),
        "--channels",
        "CH1",
        "CH2",
        "--limit",
        "-1",
    ]


def _stage2_args(options: WorkflowOptions) -> list[str]:
    args = ["--root", str(options.dataset_root)]
    if options.force_step2:
        args.append("--force")
    return args


def _stage3_args(options: WorkflowOptions) -> list[str]:
    args = ["--root", str(options.dataset_root)]
    if options.force_step3:
        args.append("--force")
    return args


def _default_stage_args(options: WorkflowOptions) -> list[str]:
    return ["--root", str(options.dataset_root)]


# Toggle individual stages by editing the ``enabled`` flag here instead of commenting
# out code inside :func:`run_workflow`.
WORKFLOW_STAGES: list[WorkflowStage] = [
    WorkflowStage(
        name="step1_prepare_dataset",
        runner=step1_prepare_dataset.main,
        build_args=_stage1_args,
    ),
    WorkflowStage(
        name="step2_pyblinker",
        runner=step2_pyblinker.main,
        build_args=_stage2_args,
    ),
    WorkflowStage(
        name="step3_run_blinker",
        runner=step3_run_blinker.main,
        build_args=_stage3_args,
    ),
    WorkflowStage(
        name="step4_compare_",
        runner=step4_compare_.main,
        build_args=_default_stage_args,
    ),
    WorkflowStage(
        name="step5_create_ground_truth",
        runner=step5_create_ground_truth.main,
        build_args=_default_stage_args,
    ),
]


def run_workflow(*, force_step2: bool = False, force_step3: bool = False) -> None:
    """Run the Murat 2018 end-to-end processing workflow."""

    options = WorkflowOptions(
        dataset_root=DATASET_ROOT,
        force_step2=force_step2,
        force_step3=force_step3,
    )
    _ensure_root_exists(options.dataset_root)

    for stage in WORKFLOW_STAGES:
        if not stage.enabled:
            LOGGER.info("Skipping %s (disabled)", stage.name)
            continue
        args = list(stage.build_args(options))
        _run_step(stage.name, args, stage.runner)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point returning ``0`` on success."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-step2",
        action="store_true",
        help="Overwrite PyBlinker outputs if they already exist.",
    )
    parser.add_argument(
        "--force-step3",
        action="store_true",
        help="Overwrite MATLAB Blinker outputs if they already exist.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run_workflow(force_step2=args.force_step2, force_step3=args.force_step3)
    except Exception as exc:  # noqa: BLE001 - convert to exit status
        LOGGER.error("Workflow failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
