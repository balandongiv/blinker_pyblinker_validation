"""Compare PyBlinker outputs with MATLAB Blinker results."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the repository root (which contains the ``src`` package) is importable when
# this script is executed directly via ``python murat_sequence/step4_compare_``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.blink_compare import (  # noqa: E402 - deferred import for path setup
    RecordingComparison,
    build_overall_summary,
    build_summary_frame,
    compare_recordings,
    render_report,
)
from src.utils.config_utils import (  # noqa: E402 - deferred import for path setup
    DEFAULT_CONFIG_PATH,
    get_path_setting,
    load_config,
)

try:  # pragma: no cover - optional dependency during tests
    from pyblinker.utils.evaluation import blink_comparison as _blink_comparison
except ModuleNotFoundError:  # pragma: no cover - pyblinker not installed for tests
    _blink_comparison = None
else:  # pragma: no cover - re-export for mypy / linters
    from pyblinker.utils.evaluation import blink_comparison  # noqa: F401


CONFIG = load_config(DEFAULT_CONFIG_PATH)
DEFAULT_ROOT = get_path_setting(CONFIG, "raw_downsampled", env_var="MURAT_DATASET_ROOT")
TOLERANCE_SAMPLES = 20
LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--tolerance-samples",
        type=int,
        default=TOLERANCE_SAMPLES,
        help="Blink start/end alignment tolerance in samples.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if _blink_comparison is None:
        LOGGER.error(
            "pyblinker.utils.evaluation.blink_comparison is unavailable; install pyblinker to run comparisons",
        )
        return 1

    output_dir = args.root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparisons: list[RecordingComparison] = compare_recordings(
        args.root,
        tolerance_samples=args.tolerance_samples,
        comparator=_blink_comparison,
    )
    if not comparisons:
        LOGGER.warning("No recordings had both pyblinker and blinker outputs")
        return 0

    summary = build_summary_frame(comparisons)
    summary_path = output_dir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)

    overall = build_overall_summary(summary)
    overall_path = None
    if not overall.empty:
        overall_path = output_dir / "summary_metrics_overall.csv"
        overall_frame = (
            overall.to_frame(name="value").reset_index().rename(columns={"index": "metric"})
        )
        overall_frame.to_csv(overall_path, index=False)

    report_path = render_report(summary, output_dir, overall=overall)

    LOGGER.info("Summary metrics saved to %s", summary_path)
    if overall_path:
        LOGGER.info("Overall summary saved to %s", overall_path)
    LOGGER.info("Summary report saved to %s", report_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
