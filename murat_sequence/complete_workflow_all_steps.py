"""End-to-end runner for the Murat 2018 processing pipeline.

This script orchestrates the comparison-focused workflow steps provided in
this repository:

1. :mod:`murat_sequence.step1_prepare_dataset`
2. :mod:`murat_sequence.step2_pyblinker`
3. :mod:`murat_sequence.step3_run_blinker`
4. :mod:`murat_sequence.step4_compare_pyblinker_vs_blinker`
5. :mod:`murat_sequence.step5_compare_viz_vs_pyblinker`
6. :mod:`murat_sequence.step6_compare_viz_vs_blinker`

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
from pathlib import Path

# Configure the shared dataset location *before* importing any of the step
# modules. They read the ``MURAT_DATASET_ROOT`` environment variable during
# import time to determine their default paths.
DATASET_ROOT = Path("D:/dataset/murat_2018")
os.environ.setdefault("MURAT_DATASET_ROOT", str(DATASET_ROOT))

from murat_sequence import (  # noqa: E402,F401 - imported for optional workflow steps
    step1_prepare_dataset,
    step2_pyblinker,
    step3_run_blinker,
    step4_compare_pyblinker_vs_blinker,
    step5_compare_viz_vs_pyblinker,
    step6_compare_viz_vs_blinker,
)
from murat_sequence.step5_compare_viz_vs_pyblinker import (  # noqa: E402,F401 - re-exported for reuse
    DEFAULT_RECORDING_IDS,
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


def _recording_args(
    *,
    mode: str,
    recording_ids: Sequence[str] | None,
) -> list[str]:
    """Build CLI arguments shared by visualization comparison steps."""

    args: list[str] = []
    if mode == "all":
        args.append("--all-recordings")
    else:
        ids = recording_ids if recording_ids else DEFAULT_RECORDING_IDS
        for rec_id in ids:
            args.extend(["--recording-id", rec_id])
    return args


def run_workflow(
    *,
    force_step2: bool = False,
    force_step3: bool = False,
    overwrite_inspected: bool = True,
    recording_mode: str = "top-bottom",
    recording_ids: Sequence[str] | None = None,
    tolerance_samples: int = 20,
) -> None:
    """Run the Murat 2018 end-to-end processing workflow."""

    _ensure_root_exists(DATASET_ROOT)

    # Step 1 – Download/conversion (limit to CH1/CH2 like the standalone script).
    # step1_args = [
    #     "--root",
    #     str(DATASET_ROOT),
    #     "--channels",
    #     "CH1",
    #     "CH2",
    #     "--limit",
    #     "-1",
    # ]
    # _run_step("step1_prepare_dataset", step1_args, step1_prepare_dataset.main)
    #
    # Step 2 – Execute PyBlinker.
    # step2_args = ["--root", str(DATASET_ROOT)]
    # if force_step2:
    #     step2_args.append("--force")
    # _run_step("step2_pyblinker", step2_args, step2_pyblinker.main)
    #
    # # Step 3 – Execute MATLAB Blinker.
    # step3_args = ["--root", str(DATASET_ROOT)]
    # if force_step3:
    #     step3_args.append("--force")
    # _run_step("step3_run_blinker", step3_args, step3_run_blinker.main)

    # Step 4 – Compare PyBlinker ↔ MATLAB Blinker.
    step4_args = [
        "--root",
        str(DATASET_ROOT),
        "--tolerance-samples",
        str(tolerance_samples),
    ]
    _run_step("step4_compare_", step4_args, step4_compare_pyblinker_vs_blinker.main)

    comparison_mode = recording_mode if recording_mode in {"all", "custom"} else "custom"
    comparison_ids = recording_ids if recording_mode == "custom" else None

    step5_args = [
        "--root",
        str(DATASET_ROOT),
        "--tolerance-samples",
        str(tolerance_samples),
    ]
    step5_args.extend(_recording_args(mode=comparison_mode, recording_ids=comparison_ids))
    _run_step(
        "step5_compare_viz_vs_pyblinker",
        step5_args,
        step5_compare_viz_vs_pyblinker.main,
    )

    step6_args = [
        "--root",
        str(DATASET_ROOT),
        "--tolerance-samples",
        str(tolerance_samples),
    ]
    step6_args.extend(_recording_args(mode=comparison_mode, recording_ids=comparison_ids))
    _run_step(
        "step6_compare_viz_vs_blinker",
        step6_args,
        step6_compare_viz_vs_blinker.main,
    )


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
    parser.add_argument(
        "--no-overwrite-inspected",
        dest="overwrite_inspected",
        action="store_false",
        help=(
            "Prevent step5 from overwriting existing *_annot_inspected.csv files "
            "when manual changes are detected."
        ),
    )
    parser.add_argument(
        "--recording-mode",
        choices=["top-bottom", "all", "custom"],
        default="top-bottom",
        help=(
            "Choose whether to run the visualization comparisons across all recordings, "
            "only the predefined top/bottom cohort, or an explicit set supplied with "
            "--recording-id."
        ),
    )
    parser.add_argument(
        "--recording-id",
        action="append",
        help="Recording ID to include when --recording-mode=custom (can be repeated).",
    )
    parser.add_argument(
        "--tolerance-samples",
        type=int,
        default=20,
        help="Blink boundary tolerance (samples) for comparison steps.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run_workflow(
            force_step2=args.force_step2,
            force_step3=args.force_step3,
            overwrite_inspected=args.overwrite_inspected,
            recording_mode=args.recording_mode,
            recording_ids=args.recording_id,
            tolerance_samples=args.tolerance_samples,
        )
    except Exception as exc:  # noqa: BLE001 - convert to exit status
        LOGGER.error("Workflow failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
