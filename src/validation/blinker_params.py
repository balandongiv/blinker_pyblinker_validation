from __future__ import annotations

from copy import deepcopy

import numpy as np


EXPLICIT_BLINKER_PARAMS = {
    "std_threshold": 1.50,
    "min_event_len": 0.05,
    "min_event_sep": 0.05,
    "base_fraction": 0.1,
    "correlation_threshold_top": 0.980,
    "correlation_threshold_bottom": 0.90,
    "correlation_threshold_middle": 0.95,
    "shut_amp_fraction": 0.9,
    "blink_amp_range_1": 3,
    "blink_amp_range_2": 50,
    "good_ratio_threshold": 0.7,
    "min_good_blinks": 10,
    "keep_signals": 0,
    "correlation_threshold": 0.98,
    "amplitude_gate_tolerance": 0.0,
    "amplitude_gate_end_window_seconds": 0.0,
    "p_avr_threshold": 3,
    "z_thresholds": np.array([[0.9, 0.98], [2.0, 5.0]]),
}

def build_experiment_blink_params(
    *,
    amplitude_gate_tolerance: float | None = None,
    amplitude_gate_end_window_seconds: float | None = None,
) -> dict[str, object]:
    """Return a fresh copy of the explicit legacy-default experiment settings."""

    params: dict[str, object] = {}
    for key, value in EXPLICIT_BLINKER_PARAMS.items():
        if isinstance(value, np.ndarray):
            params[key] = value.copy()
        else:
            params[key] = deepcopy(value)
    if amplitude_gate_tolerance is not None:
        params["amplitude_gate_tolerance"] = float(amplitude_gate_tolerance)
    if amplitude_gate_end_window_seconds is not None:
        params["amplitude_gate_end_window_seconds"] = float(
            amplitude_gate_end_window_seconds
        )
    return params
