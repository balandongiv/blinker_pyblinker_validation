from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import matlab
import matlab.engine
import numpy as np
from scipy.io import loadmat

from pyblinker.blinker.get_blink_positions import get_blink_position


DATASET_ROOT = Path(r"D:\dataset\murat_2018")
FIRFILT_PATH = Path(
    r"D:\code development\matlab_plugin\eeglab2025.1.0\plugins\firfilt"
)
SUBJECTS = {
    "9636595": "HaLTSubjectJ1611216StLRHandLegTongue.mat",
    "12400406": "CLA-SubjectJ-170504-3St-LRHand-Inter.mat",
}
CHANNELS = ("CH1", "CH2")
FILTER_LOW = 0.5
FILTER_HIGH = 30.0


def _load_probe002():
    probe_path = Path(__file__).with_name("002_probe_eeglab_filter.py")
    spec = importlib.util.spec_from_file_location("probe002", probe_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import probe module from {probe_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _matlab_firfilt(signal: np.ndarray, coeffs: np.ndarray, eng) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)

    eng.workspace["data_py"] = matlab.double([signal.tolist()])
    eng.workspace["b_py"] = matlab.double([coeffs.tolist()])
    eng.eval(
        "EEG = struct('data', data_py, 'trials', 1, 'nbchan', 1, "
        "'pnts', size(data_py, 2), 'event', struct([]));",
        nargout=0,
    )
    eng.eval("EEG = firfilt(EEG, b_py, [], 1);", nargout=0)
    return np.asarray(eng.eval("EEG.data", nargout=1), dtype=np.float64).reshape(-1)


def _compare_signals(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    length = min(reference.size, candidate.size)
    reference = reference[:length]
    candidate = candidate[:length]

    return {
        "corr": float(np.corrcoef(reference, candidate)[0, 1]),
        "max_abs_diff": float(np.max(np.abs(reference - candidate))),
        "mae": float(np.mean(np.abs(reference - candidate))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Python and MATLAB firfilt outputs against stored subject signals.",
    )
    parser.add_argument("--low", type=float, default=FILTER_LOW)
    parser.add_argument("--high", type=float, default=FILTER_HIGH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    probe002 = _load_probe002()
    eng = matlab.engine.start_matlab()
    eng.addpath(str(FIRFILT_PATH), nargout=0)

    try:
        results: list[dict[str, object]] = []

        for recording_id, mat_name in SUBJECTS.items():
            mat_payload = loadmat(
                DATASET_ROOT / recording_id / mat_name,
                squeeze_me=True,
                struct_as_record=False,
            )["o"]
            raw_data = np.asarray(mat_payload.data, dtype=np.float64)
            blinker = pickle.load(open(DATASET_ROOT / recording_id / "blinker_results.pkl", "rb"))
            signal_data = blinker["frames"]["blinks"].iloc[0]["signalData"]

            sfreq = float(mat_payload.sampFreq)
            coeffs, filter_debug = probe002.eeglab_bandpass_coefficients(
                sfreq,
                args.low,
                args.high,
            )
            params = {
                "min_event_len": 0.05,
                "std_threshold": 1.5,
                "sfreq": sfreq,
            }

            channels: list[dict[str, object]] = []
            for idx, channel in enumerate(CHANNELS):
                raw_signal = raw_data[:, idx]
                stored_signal = np.asarray(
                    signal_data[idx]["signal"],
                    dtype=np.float64,
                ).reshape(-1)

                py_dc_signal = probe002.eeglab_zero_phase_dc_filter(raw_signal, coeffs)
                matlab_firfilt_signal = _matlab_firfilt(raw_signal, coeffs, eng)

                py_positions = get_blink_position(
                    params,
                    blink_component=py_dc_signal,
                    ch=channel,
                    progress_bar=False,
                )
                matlab_firfilt_positions = get_blink_position(
                    params,
                    blink_component=matlab_firfilt_signal,
                    ch=channel,
                    progress_bar=False,
                )
                stored_positions = np.asarray(
                    signal_data[idx]["blinkPositions"],
                    dtype=np.int64,
                )

                channels.append(
                    {
                        "channel": channel,
                        "py_dc_vs_stored": _compare_signals(stored_signal, py_dc_signal),
                        "matlab_firfilt_vs_stored": _compare_signals(
                            stored_signal,
                            matlab_firfilt_signal,
                        ),
                        "py_dc_vs_matlab_firfilt": _compare_signals(
                            matlab_firfilt_signal,
                            py_dc_signal,
                        ),
                        "counts": {
                            "py_dc": int(len(py_positions)),
                            "matlab_firfilt": int(len(matlab_firfilt_positions)),
                            "stored_matlab": int(stored_positions.shape[1]),
                        },
                        "py_head": py_positions.head(5).to_dict("records"),
                        "matlab_firfilt_head": matlab_firfilt_positions.head(5).to_dict(
                            "records"
                        ),
                        "stored_head": [
                            {
                                "start_blink": int(stored_positions[0, j] - 1),
                                "end_blink": int(stored_positions[1, j] - 1),
                            }
                            for j in range(min(5, stored_positions.shape[1]))
                        ],
                    }
                )

            results.append(
                {
                    "recording_id": recording_id,
                    "filter": filter_debug,
                    "channels": channels,
                }
            )
    finally:
        eng.quit()

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
