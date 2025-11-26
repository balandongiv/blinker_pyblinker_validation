"""Compare MATLAB Blinker outputs against visualization-derived annotations.

The visualization CSV files produced during manual review are treated as the
ground truth. Each recording directory under ``--root`` is expected to contain
``blinker_results.pkl`` plus either ``<recording_id>_annot_inspected.csv`` or a
``<recording_id>.csv`` fallback alongside the raw ``.fif`` recording.

Results are emitted for the predefined top and bottom performers from
``step4_compare_pyblinker_vs_blinker``. Use ``--all-recordings`` to evaluate the
entire dataset or ``--recording-id`` to target a custom subset.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import mne
import pandas as pd

# Ensure the repository root (which contains the ``src`` package) is importable
# when this script is executed directly via ``python murat_sequence/step6...``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from murat_sequence.step5_compare_viz_vs_pyblinker import (  # noqa: E402 - shared helpers
    BOTTOM_RECORDING_IDS,
    DEFAULT_RECORDING_IDS,
    TOP_RECORDING_IDS,
    find_ground_truth_path,
    iter_selected_recordings,
    load_visual_annotations,
)
from src.utils.blink_compare import (  # noqa: E402 - deferred import for path setup
    RecordingComparison,
    load_pickle,
)
from src.utils.config_utils import (  # noqa: E402 - deferred import for path setup
    DEFAULT_CONFIG_PATH,
    get_path_setting,
    load_config,
)
from src.utils.stat import (  # noqa: E402 - deferred import for path setup
    build_overall_summary,
    build_summary_frame,
)

CONFIG = load_config(DEFAULT_CONFIG_PATH)
DEFAULT_ROOT = get_path_setting(CONFIG, "raw_downsampled", env_var="MURAT_DATASET_ROOT")
TOLERANCE_SAMPLES = 20
LOGGER = logging.getLogger(__name__)


def _load_blinker_events(blinker_payload: Mapping[str, object]) -> pd.DataFrame:
    """Extract MATLAB Blinker blink intervals from the pickle payload."""

    frames = blinker_payload.get("frames", {})
    if isinstance(frames, Mapping):
        blink_fits = pd.DataFrame(frames.get("blinkFits", {}))
    else:
        blink_fits = pd.DataFrame(getattr(frames, "get", lambda *_args, **_kwargs: {})("blinkFits", {}))
    if blink_fits.empty:
        return pd.DataFrame(columns=["start_blink", "end_blink"], dtype=int)

    blinker_events = blink_fits[["leftZero", "rightZero", "maxValue"]].rename(
        columns={"leftZero": "start_blink", "rightZero": "end_blink"}
    )
    return blinker_events.sort_values(by="start_blink").reset_index(drop=True)


def _compare_recording(
    recording_dir: Path,
    comparator,
    *,
    tolerance_samples: int,
) -> RecordingComparison | None:
    blinker_path = recording_dir / "blinker_results.pkl"
    ground_truth_path = find_ground_truth_path(recording_dir)
    fif_path = recording_dir / f"{recording_dir.name}.fif"

    if not blinker_path.exists() or ground_truth_path is None or not fif_path.exists():
        LOGGER.debug(
            "Skipping %s due to missing files (blinker=%s, ground_truth=%s, fif=%s)",
            recording_dir.name,
            blinker_path.exists(),
            ground_truth_path is not None,
            fif_path.exists(),
        )
        return None

    blinker_payload = load_pickle(blinker_path)

    raw = mne.io.read_raw_fif(fif_path, preload=True)
    channel = raw.ch_names[0]
    signal = raw.get_data(picks=[channel])[0]

    blinker_events = _load_blinker_events(blinker_payload)
    viz_events = load_visual_annotations(ground_truth_path, raw.info["sfreq"])

    if viz_events.empty:
        LOGGER.info("[skip] %s has no visualization annotations", recording_dir.name)
        return None

    comparison = comparator.compare_detected_vs_ground_truth(
        blinker_events,
        viz_events,
        raw.info["sfreq"],
        tolerance_samples=tolerance_samples,
        n_preview_rows=10,
        n_diff_rows=20,
        detected_signal=signal,
    )

    return RecordingComparison(
        recording_id=recording_dir.name,
        py_events=blinker_events,
        blinker_events=viz_events,
        metrics=comparison.metrics,
    )


def compare_recordings_viz_vs_blinker(
    root: Path,
    *,
    tolerance_samples: int,
    comparator,
    recording_ids: Sequence[str] | None,
) -> list[RecordingComparison]:
    """Compute comparisons between MATLAB Blinker detections and visualization CSVs."""

    comparisons: list[RecordingComparison] = []
    for recording_dir in iter_selected_recordings(root, recording_ids):
        result = _compare_recording(recording_dir, comparator, tolerance_samples=tolerance_samples)
        if result is not None:
            comparisons.append(result)
    return comparisons


def _write_subset_reports(
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    subset_name: str,
    recording_ids: Sequence[str],
) -> None:
    subset = summary[summary["recording_id"].isin(recording_ids)]
    if subset.empty:
        LOGGER.warning("No results available for %s subset", subset_name)
        return

    subset_path = output_dir / f"summary_metrics_{subset_name}.csv"
    subset.to_csv(subset_path, index=False)

    overall = build_overall_summary(subset)
    overall_path = output_dir / f"summary_metrics_{subset_name}_overall.csv"
    overall.to_frame(name="value").reset_index().rename(columns={"index": "metric"}).to_csv(
        overall_path, index=False
    )
    LOGGER.info("Saved %s subset summary to %s", subset_name, subset_path)
    LOGGER.info("Saved %s subset overall metrics to %s", subset_name, overall_path)


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
        "--recording-id",
        action="append",
        help="Process only the specified recording IDs (can be repeated).",
    )
    parser.add_argument(
        "--all-recordings",
        action="store_true",
        help="Process every recording under the dataset root.",
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

    blink_spec = importlib.util.find_spec("pyblinker.utils.evaluation.blink_comparison")
    if blink_spec is None:
        LOGGER.error("pyblinker is unavailable; install it to run comparisons")
        return 1

    from pyblinker.utils.evaluation import blink_comparison  # type: ignore[attr-defined]

    recording_ids: Sequence[str] | None
    if args.all_recordings:
        recording_ids = None
    elif args.recording_id:
        recording_ids = args.recording_id
    else:
        recording_ids = DEFAULT_RECORDING_IDS

    output_dir = args.root / "reports" / "viz_vs_blinker"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparisons = compare_recordings_viz_vs_blinker(
        args.root,
        tolerance_samples=args.tolerance_samples,
        comparator=blink_comparison,
        recording_ids=recording_ids,
    )
    if not comparisons:
        LOGGER.warning("No recordings had both blinker results and visualization annotations")
        return 0

    summary = build_summary_frame(comparisons)

    target_sets: dict[str, Sequence[str]] = {
        "bottom10": BOTTOM_RECORDING_IDS,
        "top10": TOP_RECORDING_IDS,
        "top_bottom": DEFAULT_RECORDING_IDS,
    }

    if args.recording_id:
        target_sets = {"selected": tuple(args.recording_id)}
    elif args.all_recordings:
        target_sets["all"] = summary["recording_id"].tolist()

    for subset_name, ids in target_sets.items():
        _write_subset_reports(summary, output_dir, subset_name=subset_name, recording_ids=ids)

    LOGGER.info("Comparison complete")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
