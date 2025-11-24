"""Compare PyBlinker outputs against visualization-derived annotations.

This script mirrors :mod:`murat_sequence.step4_compare_pyblinker_vs_blinker`
but treats the visualization-based CSV annotations as the ground truth. The
expected directory layout for each recording is::

    <root>/
      <recording_id>/
        <recording_id>.fif
        pyblinker_results.pkl
        <recording_id>_annot_inspected.csv (preferred, if present)
        <recording_id>.csv (fallback ground-truth annotations)

The CSV annotations may label blink events with codes such as ``"B"``,
``"BD"``, ``"BG"`` or ``"MANUAL"``. Every non-empty annotation label is treated
as a blink for evaluation purposes.

Outputs are restricted to the ten best and ten poorest recordings identified
in :mod:`murat_sequence.step4_compare_pyblinker_vs_blinker`, along with their
combined performance. Use ``--all-recordings`` to process every recording under
``--root`` or ``--recording-id`` to supply a custom subset.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import mne
import pandas as pd

# Ensure the repository root (which contains the ``src`` package) is importable
# when this script is executed directly via ``python murat_sequence/step5...``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.blink_compare import (  # noqa: E402 - deferred import for path setup
    RecordingComparison,
    iter_recordings,
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

BOTTOM_RECORDING_IDS = [
    "12400382",
    "12400388",
    "12400394",
    "12400349",
    "12400343",
    "12400397",
    "12400376",
    "12400346",
]

TOP_RECORDING_IDS = [
    "9636571",
    "9636595",
    "12400406",
    "12400412",
    "9636607",
    "12400409",
    "9636496",
    "9636577",
    "12400256",
]

DEFAULT_RECORDING_IDS: tuple[str, ...] = tuple(BOTTOM_RECORDING_IDS + TOP_RECORDING_IDS)
LOGGER = logging.getLogger(__name__)


def load_visual_annotations(csv_path: Path, sample_rate: float) -> pd.DataFrame:
    """Convert visualization CSV annotations to ``start_blink``/``end_blink`` samples."""

    annotations = pd.read_csv(csv_path)
    if annotations.empty:
        return pd.DataFrame(columns=["start_blink", "end_blink"], dtype=int)

    descriptions = annotations.get("description")
    if descriptions is not None:
        blink_mask = descriptions.astype(str).str.strip().ne("")
        annotations = annotations.loc[blink_mask]

    onset = pd.to_numeric(annotations.get("onset"), errors="coerce")
    duration = pd.to_numeric(annotations.get("duration"), errors="coerce")
    if onset.isna().all() or duration.isna().all():
        return pd.DataFrame(columns=["start_blink", "end_blink"], dtype=int)

    start = onset.to_numpy(dtype=float) * sample_rate
    end = start + duration.to_numpy(dtype=float) * sample_rate
    mask = (end > start) & ~pd.isna(start) & ~pd.isna(end)
    return (
        pd.DataFrame({"start_blink": start[mask], "end_blink": end[mask]})
        .astype(int)
        .sort_values("start_blink", kind="mergesort")
        .reset_index(drop=True)
    )


def load_pyblinker_events(py_payload: Mapping[str, object]) -> pd.DataFrame:
    """Extract PyBlinker blink intervals from the pickle payload."""

    py_events_raw = pd.DataFrame(py_payload.get("events", {}))
    if py_events_raw.empty:
        return pd.DataFrame(columns=["start_blink", "end_blink"], dtype=int)

    py_events = py_events_raw[["left_zero", "right_zero", "max_value"]].rename(
        columns={"left_zero": "start_blink", "right_zero": "end_blink"}
    )
    return py_events.sort_values(by="start_blink").reset_index(drop=True)


def find_ground_truth_path(recording_dir: Path) -> Path | None:
    """Return the preferred visualization CSV path for ``recording_dir``."""

    inspected = recording_dir / f"{recording_dir.name}_annot_inspected.csv"
    if inspected.exists():
        return inspected

    fallback = recording_dir / f"{recording_dir.name}.csv"
    if fallback.exists():
        return fallback

    return None


def iter_selected_recordings(root: Path, recording_ids: Sequence[str] | None) -> Iterable[Path]:
    if recording_ids is None:
        yield from iter_recordings(root)
        return

    id_set = set(recording_ids)
    available = list(iter_recordings(root))
    for recording_dir in available:
        if recording_dir.name in id_set:
            yield recording_dir
    for missing in id_set - {path.name for path in available}:
        LOGGER.warning("[skip] %s not found under %s", missing, root)


def _compare_recording(
    recording_dir: Path,
    comparator,
    *,
    tolerance_samples: int,
) -> RecordingComparison | None:
    py_path = recording_dir / "pyblinker_results.pkl"
    ground_truth_path = find_ground_truth_path(recording_dir)
    fif_path = recording_dir / f"{recording_dir.name}.fif"

    if not py_path.exists() or ground_truth_path is None or not fif_path.exists():
        LOGGER.debug(
            "Skipping %s due to missing files (pyblinker=%s, ground_truth=%s, fif=%s)",
            recording_dir.name,
            py_path.exists(),
            ground_truth_path is not None,
            fif_path.exists(),
        )
        return None

    py_payload = load_pickle(py_path)
    channel = py_payload.get("metrics", {}).get("channel", 0)

    raw = mne.io.read_raw_fif(fif_path, preload=True)
    signal = raw.get_data(picks=[channel])[0]

    py_events = load_pyblinker_events(py_payload)
    viz_events = load_visual_annotations(ground_truth_path, raw.info["sfreq"])

    if viz_events.empty:
        LOGGER.info("[skip] %s has no visualization annotations", recording_dir.name)
        return None

    comparison = comparator.compare_detected_vs_ground_truth(
        py_events,
        viz_events,
        raw.info["sfreq"],
        tolerance_samples=tolerance_samples,
        n_preview_rows=10,
        n_diff_rows=20,
        detected_signal=signal,
    )

    return RecordingComparison(
        recording_id=recording_dir.name,
        py_events=py_events,
        blinker_events=viz_events,
        metrics=comparison.metrics,
    )


def compare_recordings_viz_vs_pyblinker(
    root: Path,
    *,
    tolerance_samples: int,
    comparator,
    recording_ids: Sequence[str] | None,
) -> list[RecordingComparison]:
    """Compute comparisons between PyBlinker detections and visualization CSVs."""

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

    output_dir = args.root / "reports" / "viz_vs_pyblinker"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparisons = compare_recordings_viz_vs_pyblinker(
        args.root,
        tolerance_samples=args.tolerance_samples,
        comparator=blink_comparison,
        recording_ids=recording_ids,
    )
    if not comparisons:
        LOGGER.warning("No recordings had both pyblinker results and visualization annotations")
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
