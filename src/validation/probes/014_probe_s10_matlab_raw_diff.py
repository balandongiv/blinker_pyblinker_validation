from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

import matlab.engine
import mne
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyblinker.blinker.legacy_eeglab_filter import legacy_blinker_bandpass


def _matlab_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="S10")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\dataset\drowsy_driving_raja_processed"),
    )
    args = parser.parse_args()

    subject_dir = args.dataset_root / args.subject / "blinker_pyblinker_validation"
    edf_path = subject_dir / f"{args.subject}.edf"
    payload = pickle.loads((subject_dir / "blinker_results.pkl").read_bytes())
    matlab_signal = np.asarray(
        next(
            item
            for item in payload["frames"]["blinks"].iloc[0]["signalData"]
            if item["signalLabel"] == "eog_vert_right"
        )["signal"],
        dtype=float,
    ).reshape(-1)

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True, eog=True)
    py_raw = raw.get_data(picks="eog_vert_right")[0]
    py_filtered = legacy_blinker_bandpass(
        py_raw,
        sfreq=float(raw.info["sfreq"]),
        low_cutoff_hz=1.0,
        high_cutoff_hz=20.0,
    )

    eng = matlab.engine.start_matlab("-nojvm")
    try:
        eeglab_root = Path(r"D:\code development\matlab_plugin\eeglab2025.1.0")
        firfilt_dir = Path(
            r"D:\code development\matlab_plugin\eeglab2025.1.0\plugins\firfilt"
        )
        biosig_root = Path(
            r"D:\code development\matlab_plugin\eeglab2025.1.0\plugins\Biosig3.8.4\biosig"
        )
        eeglab_functions_root = eeglab_root / "functions"
        for candidate in (firfilt_dir,):
            eng.addpath(_matlab_path(candidate), nargout=0)
        eng.addpath(eng.genpath(_matlab_path(biosig_root)), nargout=0)
        eng.addpath(eng.genpath(_matlab_path(eeglab_functions_root)), nargout=0)
        eng.addpath(_matlab_path(firfilt_dir), nargout=0)
        print("edfread_exists", eng.eval("exist('edfread','file')", nargout=1))
        print("biosig_exists", eng.eval("exist('pop_biosig','file')", nargout=1))

        if int(eng.eval("exist('pop_biosig','file')", nargout=1)) == 2:
            try:
                script = f"""
                EEG = pop_biosig('{_matlab_path(edf_path)}');
                labels = string({{EEG.chanlocs.labels}});
                disp(labels');
                chIdx = find(strcmpi(labels, "eog_vert_right"), 1);
                signal = double(EEG.data(chIdx, :));
                """
                eng.eval(script, nargout=0)
                matlab_raw = np.asarray(eng.workspace["signal"], dtype=float).reshape(-1)
                raw_diff = py_raw * 1e6 - matlab_raw
                print("raw_max_abs_microvolt", float(np.max(np.abs(raw_diff))))
                print("raw_mean_abs_microvolt", float(np.mean(np.abs(raw_diff))))
            except Exception as exc:  # pragma: no cover - probe-only fallback
                print("matlab_raw_probe_failed", type(exc).__name__, str(exc))
    finally:
        eng.quit()

    diff = py_filtered * 1e6 - matlab_signal
    print("filtered_max_abs_microvolt", float(np.max(np.abs(diff))))
    print("filtered_mean_abs_microvolt", float(np.mean(np.abs(diff))))
    print("filtered_rmse_microvolt", float(np.sqrt(np.mean(diff**2))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
