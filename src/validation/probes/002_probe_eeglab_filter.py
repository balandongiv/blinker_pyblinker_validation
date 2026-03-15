from __future__ import annotations

import json
import pickle
from pathlib import Path

import mne
import numpy as np
from scipy.signal import lfilter

from pyblinker.blinker.get_blink_positions import get_blink_position


DATASET_ROOT = Path(r"D:\dataset\murat_2018")
SUBJECTS = ("9636595", "12400406")
CHANNELS = ("CH1", "CH2")
FILTER_LOW = 0.5
FILTER_HIGH = 30.0


def _fkernel(order: int, normalized_cutoff: float, window: np.ndarray) -> np.ndarray:
    sample = np.arange(-(order // 2), (order // 2) + 1, dtype=np.float64)
    cutoff = float(normalized_cutoff) / 2.0

    kernel = np.empty_like(sample)
    zero_mask = sample == 0
    kernel[zero_mask] = 2.0 * np.pi * cutoff
    kernel[~zero_mask] = (
        np.sin(2.0 * np.pi * cutoff * sample[~zero_mask]) / sample[~zero_mask]
    )
    kernel *= window
    kernel /= np.sum(kernel)
    return kernel


def _spectral_inversion(kernel: np.ndarray) -> np.ndarray:
    out = -kernel.copy()
    out[len(out) // 2] += 1.0
    return out


def eeglab_bandpass_coefficients(
    sfreq: float,
    low_cutoff: float,
    high_cutoff: float,
) -> tuple[np.ndarray, dict[str, float]]:
    f_nyquist = sfreq / 2.0
    edge = np.sort(np.asarray([low_cutoff, high_cutoff], dtype=np.float64))

    max_tbw = np.asarray([edge[0], f_nyquist - edge[1]], dtype=np.float64)
    max_df = float(np.min(max_tbw))

    transition_width = min(max(edge[0] * 0.25, 2.0), max_df)
    order = int(np.ceil((3.3 / (transition_width / sfreq)) / 2.0) * 2.0)

    cutoff = edge + np.asarray([-transition_width, transition_width]) / 2.0
    window = np.hamming(order + 1).astype(np.float64, copy=False)

    lowpass = _fkernel(order, cutoff[0] / f_nyquist, window)
    highpass = _spectral_inversion(_fkernel(order, cutoff[1] / f_nyquist, window))
    bandpass = _spectral_inversion(lowpass + highpass)

    debug = {
        "sfreq": float(sfreq),
        "transition_width": float(transition_width),
        "order": float(order),
        "cutoff_low": float(cutoff[0]),
        "cutoff_high": float(cutoff[1]),
    }
    return bandpass, debug


def eeglab_zero_phase_dc_filter(signal: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)

    if coeffs.size % 2 != 1:
        raise ValueError("Expected an odd filter length (even order).")

    group_delay = (coeffs.size - 1) // 2
    start_pad = np.full(group_delay, signal[0], dtype=np.float64)
    end_pad = np.full(group_delay, signal[-1], dtype=np.float64)
    padded = np.concatenate([start_pad, signal, end_pad])
    filtered = lfilter(coeffs, [1.0], padded)
    return filtered[2 * group_delay :]


def compare_subject(recording_id: str) -> dict[str, object]:
    raw_path = DATASET_ROOT / recording_id / f"{recording_id}.fif"
    blinker_path = DATASET_ROOT / recording_id / "blinker_results.pkl"

    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="ERROR").pick(CHANNELS)
    matlab = pickle.load(open(blinker_path, "rb"))
    signal_data = matlab["frames"]["blinks"].iloc[0]["signalData"]

    coeffs, filter_debug = eeglab_bandpass_coefficients(
        float(raw.info["sfreq"]),
        FILTER_LOW,
        FILTER_HIGH,
    )

    params = {
        "min_event_len": 0.05,
        "std_threshold": 1.5,
        "sfreq": float(raw.info["sfreq"]),
    }

    channel_rows: list[dict[str, object]] = []
    for idx, channel in enumerate(CHANNELS):
        py_signal = raw.get_data(picks=[channel])[0]
        py_filtered = eeglab_zero_phase_dc_filter(py_signal, coeffs)

        matlab_signal = np.asarray(signal_data[idx]["signal"], dtype=np.float64).reshape(-1)
        matlab_positions = np.asarray(
            signal_data[idx]["blinkPositions"],
            dtype=np.float64,
        )

        length = min(py_filtered.size, matlab_signal.size)
        correlation = float(np.corrcoef(py_filtered[:length], matlab_signal[:length])[0, 1])
        scale = float(
            np.dot(matlab_signal[:length], py_filtered[:length])
            / np.dot(py_filtered[:length], py_filtered[:length])
        )
        mae = float(np.mean(np.abs(matlab_signal[:length] - scale * py_filtered[:length])))

        py_positions = get_blink_position(
            params,
            blink_component=py_filtered,
            ch=channel,
            progress_bar=False,
        )

        matlab_positions_zero = np.asarray(matlab_positions, dtype=np.int64) - 1
        matlab_head = [
            {
                "start_blink": int(matlab_positions_zero[0, j]),
                "end_blink": int(matlab_positions_zero[1, j]),
            }
            for j in range(min(5, matlab_positions_zero.shape[1]))
        ]

        channel_rows.append(
            {
                "channel": channel,
                "corr": correlation,
                "scale_mat_over_py": scale,
                "mae_after_scale": mae,
                "py_count": int(len(py_positions)),
                "mat_count": int(matlab_positions_zero.shape[1]),
                "py_head": py_positions.head(5).to_dict("records"),
                "mat_head": matlab_head,
            }
        )

    return {
        "recording_id": recording_id,
        "filter": filter_debug,
        "channels": channel_rows,
    }


def main() -> int:
    results = [compare_subject(recording_id) for recording_id in SUBJECTS]
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
