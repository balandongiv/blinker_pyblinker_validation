"""Run the MATLAB Blinker pipeline for every FIF file in the Raja dataset."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import mne
import pandas as pd

try:
    from src.matlab_runner.execute_blinker import (
        BLINKER_KEYS,
        DEFAULT_PROJECT_ROOT,
        run_blinker as matlab_run_blinker,
        start_matlab as matlab_start_matlab,
    )
except ModuleNotFoundError:
    # Support `python raja_sequence/step3_run_blinker.py` from repo root.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.matlab_runner.execute_blinker import (
        BLINKER_KEYS,
        DEFAULT_PROJECT_ROOT,
        run_blinker as matlab_run_blinker,
        start_matlab as matlab_start_matlab,
    )

from src.utils.blink_events import prepare_blinker_frame
from src.utils.config_utils import (
    DEFAULT_CONFIG_PATH,
    get_default_blinker_plugin,
    get_path_setting,
    load_config,
)

CONFIG = load_config(DEFAULT_CONFIG_PATH)
DEFAULT_EEGLAB_ROOT = get_path_setting(CONFIG, "eeglab_root", env_var="EEGLAB_ROOT")
DEFAULT_BLINKER_PLUGIN = get_default_blinker_plugin(CONFIG) or "Blinker1.2.0"
LOGGER = logging.getLogger(__name__)

# Raja dataset default root
DEFAULT_RAJA_ROOT = Path(r"D:\dataset\drowsy_driving_raja_processed")


@dataclass(slots=True)
class BlinkerRunConfig:
    eeglab_root: Path = DEFAULT_EEGLAB_ROOT
    blinker_plugin: str = DEFAULT_BLINKER_PLUGIN
    project_root: Path = DEFAULT_PROJECT_ROOT


def sanitise_metadata(raw: mne.io.BaseRaw) -> None:
    """Make metadata EDF-friendly by replacing spaces in string fields."""
    for field in ("device_info", "subject_info"):
        info_value = raw.info.get(field)
        if isinstance(info_value, dict):
            for key, value in info_value.items():
                if isinstance(value, str):
                    info_value[key] = value.replace(" ", "_")


def convert_fif_to_edf(fif_path: Path, edf_path: Path, overwrite: bool = False) -> bool:
    """Convert a single FIF file to EDF format. Return True on success."""
    if edf_path.exists() and not overwrite:
        LOGGER.info("Skipping existing EDF: %s", edf_path)
        return True

    try:
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
        try:
            # We try to keep only eeg/eog if it doesn't break.
            raw.pick(["eeg", "eog"])
        except Exception:
            pass  # Ignore if picking fails

        sanitise_metadata(raw)

        # Ensure output directory exists before export
        edf_path.parent.mkdir(parents=True, exist_ok=True)
        
        # MNE overwrite=True is required to overwrite existing file
        raw.export(edf_path, fmt="edf", overwrite=True)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to convert FIF -> EDF for %s: %s", fif_path, exc)
        return False
    
    LOGGER.info("Converted FIF -> EDF: %s -> %s", fif_path, edf_path)
    return True


def discover_subjects(root: Path) -> Iterable[Path]:
    """Yield all subject folders S1 through S27 from the Raja dataset."""
    for i in range(1, 28):
        subject_dir = root / f"S{i}"
        yield subject_dir


def run_blinker(eng, edf_path: Path) -> Dict[str, pd.DataFrame]:  # pragma: no cover
    frames: Dict[str, pd.DataFrame] = {}
    output = matlab_run_blinker(eng, edf_path)
    for key in BLINKER_KEYS:
        try:
            frames[key] = prepare_blinker_frame(output.get(key, pd.DataFrame()))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to serialise MATLAB output %s for %s: %s", key, edf_path, exc)
            frames[key] = pd.DataFrame()
    return frames


def persist_results(output_dir: Path, frames: Dict[str, pd.DataFrame], overwrite: bool) -> Path:
    payload = {
        "frames": frames,
        "params": {},
    }

    events = frames.get("blinkFits")
    if events is not None and not events.empty:
        columns = {col.lower(): col for col in events.columns}
        if "latency" in columns:
            payload["events_onset_sec"] = pd.to_numeric(events[columns["latency"]], errors="coerce").tolist()
        if "duration" in columns:
            payload["events_duration_sec"] = (
                pd.to_numeric(events[columns["duration"]], errors="coerce").tolist()
            )

    target = output_dir / "blinker_results.pkl"
    if target.exists() and not overwrite:
        LOGGER.info("Skipping existing blinker results: %s", target)
        return target

    serialisable = {key: frame.reset_index(drop=True) for key, frame in frames.items()}
    payload["frames"] = serialisable
    with target.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metadata_path = output_dir / "blinker_results.json"
    with metadata_path.open("w", encoding="utf8") as handle:
        json.dump({"keys": list(frames)}, handle, indent=2)

    LOGGER.info("Saved blinker outputs → %s", target)
    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_RAJA_ROOT,
        help="Root directory containing Raja dataset subjects (S1-S27).",
    )
    parser.add_argument(
        "--eeglab-root",
        type=Path,
        default=DEFAULT_EEGLAB_ROOT,
        help="Path to the EEGLAB installation.",
    )
    parser.add_argument(
        "--blinker-plugin",
        type=str,
        default=DEFAULT_BLINKER_PLUGIN,
        help="Name of the Blinker plugin folder inside EEGLAB's plugins directory.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Path containing MATLAB helpers to add to the MATLAB path.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing results.")
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

    subjects = list(discover_subjects(args.root))
    if not any(sub.exists() for sub in subjects):
        LOGGER.warning("No valid subject folders (S1-S27) were found in %s", args.root)
        return 0

    cfg = BlinkerRunConfig(
        eeglab_root=args.eeglab_root,
        blinker_plugin=args.blinker_plugin,
        project_root=args.project_root,
    )

    processed = 0
    skipped = 0
    try:
        eng = matlab_start_matlab(
            cfg.eeglab_root,
            project_root=cfg.project_root,
            blinker_plugin=cfg.blinker_plugin,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Unable to start MATLAB engine: %s", exc)
        return 1

    try:
        for subject_dir in subjects:
            subject_id = subject_dir.name
            
            if not subject_dir.is_dir():
                LOGGER.warning("Subject directory %s not found, skipping.", subject_dir)
                skipped += 1
                continue

            fif_path = subject_dir / f"{subject_id}.fif"
            if not fif_path.is_file():
                LOGGER.warning("Missing %s in %s, skipping.", fif_path.name, subject_dir)
                skipped += 1
                continue

            output_dir = subject_dir / "blinker_pyblinker_validation"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            target = output_dir / "blinker_results.pkl"
            if target.exists() and not args.force:
                LOGGER.info(
                    "Skipping %s because Blinker outputs already exist at %s",
                    subject_id,
                    target,
                )
                skipped += 1
                continue

            edf_path = output_dir / f"{subject_id}.edf"
            
            # Convert to EDF if needed
            if not edf_path.exists() or args.force:
                LOGGER.info("Converting %s to EDF format...", fif_path)
                success = convert_fif_to_edf(fif_path, edf_path, overwrite=args.force)
                if not success:
                    LOGGER.error("Skipping %s due to EDF conversion failure.", subject_id)
                    skipped += 1
                    continue
            else:
                LOGGER.info("Using existing EDF file %s", edf_path)

            try:
                LOGGER.info("Running Blinker pipeline for %s", subject_id)
                frames = run_blinker(eng, edf_path)
                persist_results(output_dir, frames, overwrite=True)
                processed += 1
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Blinker failed for %s: %s", subject_id, exc)
    finally:  # pragma: no cover - requires MATLAB engine
        try:
            eng.quit()
        except Exception:  # noqa: BLE001
            LOGGER.warning("Failed to close MATLAB engine cleanly")

    LOGGER.info(
        "Blinker pipeline finished: %s processed, %s skipped, out of %s total subjects",
        processed,
        skipped,
        len(subjects),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
