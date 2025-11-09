"""Run the MATLAB Blinker pipeline for every EDF file in the dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from src.matlab_runner.execute_blinker import (
    BLINKER_KEYS,
    DEFAULT_PROJECT_ROOT,
    run_blinker as matlab_run_blinker,
    start_matlab as matlab_start_matlab,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT_RAW = os.environ.get("MURAT_DATASET_ROOT")
DEFAULT_ROOT = Path(DEFAULT_ROOT_RAW) if DEFAULT_ROOT_RAW else REPO_ROOT / "data" / "murat_2018"
DEFAULT_EEGLAB_ROOT = Path(r"D:\code development\matlab_plugin\eeglab2025.1.0")
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class BlinkerRunConfig:
    eeglab_root: Path = DEFAULT_EEGLAB_ROOT
    blinker_plugin: str = "Blinker1.2.0"
    project_root: Path = DEFAULT_PROJECT_ROOT


def discover_edf_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.edf"))


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.applymap(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )


def run_blinker(eng, edf_path: Path) -> Dict[str, pd.DataFrame]:  # pragma: no cover
    frames: Dict[str, pd.DataFrame] = {}
    output = matlab_run_blinker(eng, edf_path)
    for key in BLINKER_KEYS:
        try:
            frames[key] = _prepare_frame(output.get(key, pd.DataFrame()))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to serialise MATLAB output %s for %s: %s", key, edf_path, exc)
            frames[key] = pd.DataFrame()
    return frames


def persist_results(edf_path: Path, frames: Dict[str, pd.DataFrame], overwrite: bool) -> Path:
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

    target = edf_path.with_name("blinker_results.pkl")
    if target.exists() and not overwrite:
        LOGGER.info("Skipping existing blinker results: %s", target)
        return target

    serialisable = {key: frame.reset_index(drop=True) for key, frame in frames.items()}
    payload["frames"] = serialisable
    with target.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metadata_path = edf_path.with_name("blinker_results.json")
    with metadata_path.open("w", encoding="utf8") as handle:
        json.dump({"keys": list(frames)}, handle, indent=2)

    LOGGER.info("Saved blinker outputs → %s", target)
    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing EDF files.",
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
        default="Blinker1.2.0",
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

    edf_files = list(discover_edf_files(args.root))
    if not edf_files:
        LOGGER.warning("No EDF files were found below %s", args.root)
        return 0

    cfg = BlinkerRunConfig(
        eeglab_root=args.eeglab_root,
        blinker_plugin=args.blinker_plugin,
        project_root=args.project_root,
    )

    processed = 0
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
        for edf_path in edf_files:
            try:
                frames = run_blinker(eng, edf_path)
                persist_results(edf_path, frames, overwrite=args.force)
                processed += 1
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Blinker failed for %s: %s", edf_path, exc)
    finally:  # pragma: no cover - requires MATLAB engine
        try:
            eng.quit()
        except Exception:  # noqa: BLE001
            LOGGER.warning("Failed to close MATLAB engine cleanly")

    LOGGER.info("Blinker pipeline completed for %s/%s file(s)", processed, len(edf_files))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
