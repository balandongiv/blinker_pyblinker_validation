from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PYBLINKER_REPO_ROOT = REPO_ROOT / "pyblinker"
if PYBLINKER_REPO_ROOT.exists() and str(PYBLINKER_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(PYBLINKER_REPO_ROOT))

from pyblinker.blinker.pyblinker import BlinkDetector
DEFAULT_ROOT = Path(r"D:\dataset\sustained_attention_driving")
DEFAULT_SUMMARY_PATH = REPO_ROOT / "reports" / "sustained_attention_pyblinker_batch_summary.csv"
PYBLINKER_PICKLE_NAME = "pyblinker_results.pkl"


@dataclass(slots=True)
class RecordingResult:
    set_path: Path
    fif_path: Path
    output_path: Path
    status: str
    reason: str
    selected_channel: str = ""
    annotation_count: int = 0
    good_blink_count: int = 0

    def to_row(self) -> dict[str, object]:
        return {
            "subject_id": self.set_path.parent.parent.name,
            "recording_id": self.set_path.parent.name,
            "set_path": str(self.set_path),
            "fif_path": str(self.fif_path),
            "output_path": str(self.output_path),
            "status": self.status,
            "reason": self.reason,
            "selected_channel": self.selected_channel,
            "annotation_count": int(self.annotation_count),
            "good_blink_count": int(self.good_blink_count),
        }


def configure_logging(verbose: bool) -> None:
    mne.set_log_level("INFO" if verbose else "WARNING")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert sustained_attention_driving EEGLAB recordings to FIF, "
            "run PyBlinker, and save annotation-style pickle outputs beside each recording."
        ),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Dataset root containing subject folders.")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="CSV path for per-recording batch status output.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N discovered recordings.")
    parser.add_argument("--force-fif", action="store_true", help="Regenerate FIF files even if they already exist.")
    parser.add_argument(
        "--force-pyblinker",
        action="store_true",
        help="Regenerate pyblinker_results.pkl even if it already exists.",
    )
    parser.add_argument("--filter-low", type=float, default=0.5, help="PyBlinker lower filter bound in Hz.")
    parser.add_argument("--filter-high", type=float, default=30.0, help="PyBlinker upper filter bound in Hz.")
    parser.add_argument("--resample-rate", type=float, default=100.0, help="PyBlinker resample rate in Hz.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args(argv)


def discover_set_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.set"))


def build_fif_path(set_path: Path) -> Path:
    return set_path.with_suffix(".fif")


def build_output_path(set_path: Path) -> Path:
    return set_path.with_name(PYBLINKER_PICKLE_NAME)


def convert_annotations_to_frame(annotations: mne.Annotations) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "onset": annotations.onset.astype(float, copy=False),
            "duration": annotations.duration.astype(float, copy=False),
            "description": pd.Series(annotations.description, dtype="string"),
        }
    )


def convert_set_to_fif(set_path: Path, fif_path: Path, *, overwrite: bool) -> None:
    if fif_path.exists() and not overwrite:
        print(f"Reusing existing FIF: {fif_path}")
        return

    print(f"Loading EEGLAB file: {set_path}")
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
    print(
        f"Saving FIF: {fif_path} (channels={len(raw.ch_names)}, "
        f"sfreq={float(raw.info['sfreq']):.3f} Hz, "
        f"duration={float(raw.n_times / raw.info['sfreq']):.2f} s)"
    )
    raw.save(fif_path, overwrite=True, verbose="ERROR")


def run_pyblinker_on_fif(
    fif_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
    filter_low: float,
    filter_high: float,
    resample_rate: float,
) -> tuple[str, int, int]:
    if output_path.exists() and not overwrite:
        print(f"Reusing existing PyBlinker output: {output_path}")
        frame = pd.read_pickle(output_path)
        selected_channel = ""
        return selected_channel, int(len(frame)), int(len(frame))

    print(f"Reading FIF for PyBlinker: {fif_path}")
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    raw.pick(picks="eeg")
    print(f"Running PyBlinker on {len(raw.ch_names)} EEG channels at {float(raw.info['sfreq']):.3f} Hz")

    detector = BlinkDetector(
        raw.copy(),
        visualize=False,
        annot_label="blink",
        filter_low=filter_low,
        filter_high=filter_high,
        resample_rate=resample_rate,
        n_jobs=1,
        use_multiprocessing=False,
    )
    annotations, channel, n_good, _blink_details, _fig_data, _selected = detector.get_blink()

    frame = convert_annotations_to_frame(annotations)
    print(f"Saving PyBlinker pickle: {output_path} (selected_channel={channel}, annotations={len(frame)}, good_blinks={int(n_good)})")
    with output_path.open("wb") as handle:
        pickle.dump(frame, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return str(channel), int(len(frame)), int(n_good)


def process_recording(
    set_path: Path,
    *,
    force_fif: bool,
    force_pyblinker: bool,
    filter_low: float,
    filter_high: float,
    resample_rate: float,
) -> RecordingResult:
    fif_path = build_fif_path(set_path)
    output_path = build_output_path(set_path)
    print(f"Processing recording directory: {set_path.parent}")

    try:
        convert_set_to_fif(set_path, fif_path, overwrite=force_fif)
        selected_channel, annotation_count, good_blink_count = run_pyblinker_on_fif(
            fif_path,
            output_path,
            overwrite=force_pyblinker,
            filter_low=filter_low,
            filter_high=filter_high,
            resample_rate=resample_rate,
        )
    except Exception as exc:  # noqa: BLE001 - batch runner should continue
        print(f"ERROR: Failed processing {set_path}: {exc}")
        return RecordingResult(
            set_path=set_path,
            fif_path=fif_path,
            output_path=output_path,
            status="failed",
            reason=str(exc),
        )

    return RecordingResult(
        set_path=set_path,
        fif_path=fif_path,
        output_path=output_path,
        status="processed",
        reason="ok",
        selected_channel=selected_channel,
        annotation_count=annotation_count,
        good_blink_count=good_blink_count,
    )


def maybe_limit(paths: Iterable[Path], limit: int | None) -> list[Path]:
    ordered = list(paths)
    if limit is None:
        return ordered
    return ordered[: max(0, limit)]


def write_summary(summary_path: Path, results: list[RecordingResult]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([result.to_row() for result in results])
    frame.to_csv(summary_path, index=False)
    print(f"Wrote batch summary: {summary_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    print(f"Dataset root: {args.root}")
    set_files = maybe_limit(discover_set_files(args.root), args.limit)
    print(f"Discovered {len(set_files)} EEGLAB recording(s)")
    if not set_files:
        print(f"WARNING: No .set files found below {args.root}")
        return 0

    results: list[RecordingResult] = []
    for index, set_path in enumerate(set_files, start=1):
        print(f"[{index}/{len(set_files)}] {set_path}")
        result = process_recording(
            set_path,
            force_fif=args.force_fif,
            force_pyblinker=args.force_pyblinker,
            filter_low=args.filter_low,
            filter_high=args.filter_high,
            resample_rate=args.resample_rate,
        )
        results.append(result)
        print(
            f"Finished {set_path.parent.name} with status={result.status} "
            f"annotations={result.annotation_count} good_blinks={result.good_blink_count}"
        )

    write_summary(args.summary_path, results)

    processed = sum(result.status == "processed" for result in results)
    failed = sum(result.status == "failed" for result in results)
    print(f"Batch complete: processed={processed} failed={failed} total={len(results)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
