"""
Blink Detection Evaluation with Cross-Sampling Normalization
============================================================

Purpose
-------
Compare blink detections from *blinker* and *pyblinker* against ground-truth
annotations stored in `seg_annotated_raw.fif`, even when the detector outputs
were produced at different sampling rates.

What this script does
---------------------
1) Loads ground truth from the FIF file using MNE, converts annotations into
   (start_sample, end_sample) intervals in **raw-sample space**.

2) Loads detection outputs from:
   - `blinkFits.pkl` (blinker)
   - `blink_details.pkl` (pyblinker)

3) Normalizes detector intervals to the raw's sample space, handling:
   - outputs in **seconds**
   - outputs in **samples** at the detector’s own sampling rate (hard-coded)
   - mixed/unknown formats via `"auto"` inference (falls back to seconds/samples
     heuristics).

4) Computes:
   - **Per-sample** confusion (TP, TN, FP, FN) and **Precision, Recall, F1**.
     (Per-sample yields a meaningful TN.)
   - **Event-level** metrics that respect **start** and **end**:
       * Greedy matching with IoU ≥ `min_iou`, or
       * Both start and end within ± tolerances (in samples).
     Reports event-level TP, FP, FN, Precision, Recall, F1, and MAE of start/end.

How to use (hardcode per-detector format)
-----------------------------------------
In `main()`, set for each detector:
- UNITS: "seconds" if outputs are in seconds; "samples" if outputs are in sample
  indices at the detector's own sampling rate; "auto" to let the script guess.
- FS: detector's sampling rate (Hz) **only if** UNITS == "samples".
  (Ignored if UNITS == "seconds" or "auto" detects seconds.)

Example:
    BLINKER_UNITS = "samples"
    BLINKER_FS    = 200.0       # blinker ran at 200 Hz

    PYBLINKER_UNITS = "seconds"
    PYBLINKER_FS    = None      # ignored because units are seconds

Outputs
-------
- Prints sampling rate of raw, number of ground-truth blinks
- Per-sample metrics for each detector: TP, TN, FP, FN, Precision, Recall, F1
- Event-level metrics for each detector:
    TP, FP, FN, Precision, Recall, F1, MAE of start/end (in samples)

Notes
-----
- If your pickles store only blink centers, ensure a duration is present or
  adjust the `center + duration` branch to supply a default duration.
- Event-level TN is not well-defined, hence only per-sample TN is reported.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Dict, Any

import pickle
import numpy as np
import pandas as pd
import mne


# -------------------------------------------------------------------
# Paths (as per your project layout)
# -------------------------------------------------------------------
BLINKER_PKL   = Path("data_test/S01_20170519_043933_3/blinker/blinkFits.pkl")
PYBLINKER_PKL = Path("data_test/S01_20170519_043933_3/pyblinker/blink_details.pkl")
FIF_PATH      = Path("data_test/S01_20170519_043933_3/seg_annotated_raw.fif")


# -------------------------------------------------------------------
# Format controls (hardcode here as needed)
#   UNITS:  "seconds" | "samples" | "auto"
#   FS:     detector sampling rate (Hz) if UNITS == "samples"
# -------------------------------------------------------------------
BLINKER_UNITS = "auto"     # change to "seconds" or "samples" if you know it
BLINKER_FS    = None       # e.g., 200.0 if blinker indices are at 200 Hz

PYBLINKER_UNITS = "auto"   # change to "seconds" or "samples" if you know it
PYBLINKER_FS    = None     # e.g., 256.0 if pyblinker indices are at 256 Hz


# -------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------
def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_ground_truth_intervals(raw: mne.io.BaseRaw) -> List[Tuple[int, int]]:
    """Extract ground-truth blink intervals from raw.annotations into raw-sample space."""
    anns = raw.annotations
    if anns is None or len(anns) == 0:
        return []
    onset = np.asarray(anns.onset, dtype=float)    # seconds
    dur   = np.asarray(anns.duration, dtype=float) # seconds
    endt  = onset + dur

    starts = raw.time_as_index(onset)
    ends   = raw.time_as_index(endt)

    n = raw.n_times
    out: List[Tuple[int, int]] = []
    for s, e in zip(starts, ends):
        s_i = int(max(0, min(n, s)))
        e_i = int(max(0, min(n, e)))
        if e_i > s_i:
            out.append((s_i, e_i))
    return out


# -------------------------------------------------------------------
# Normalization utilities
# -------------------------------------------------------------------
def _likely_seconds(values: Sequence[float], n_samples: int, sfreq: float) -> bool:
    """Heuristic to guess if numeric values look like seconds rather than samples."""
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return False
    frac = np.mean((vals % 1.0) != 0.0)
    return (np.nanmax(vals) < n_samples / sfreq) or (frac > 0.25)


def _scale_indices_between_fs(idxs: Sequence[float], src_fs: float, dst_fs: float, n_dst: int) -> np.ndarray:
    """Map sample indices from src_fs to dst_fs (raw)."""
    idxs = np.asarray(idxs, dtype=float)
    scaled = np.round(idxs * (dst_fs / src_fs))
    scaled = np.clip(scaled, 0, n_dst - 1)
    return scaled.astype(int)


def _to_raw_indices(
        start_vals: Sequence[float],
        end_vals: Sequence[float],
        units: str,
        raw_sfreq: float,
        raw_n: int,
        detector_fs: Optional[float] = None,
) -> List[Tuple[int, int]]:
    """
    Convert paired start/end values to raw sample indices given the unit spec.

    units == "seconds": start/end are seconds → multiply by raw_sfreq.
    units == "samples": start/end are indices at detector_fs → scale to raw_sfreq.
    units == "auto":    try seconds first (heuristic), else treat as detector samples
                        (requires detector_fs; if absent, fallback to seconds).
    """
    s_vals = np.asarray(start_vals, dtype=float)
    e_vals = np.asarray(end_vals, dtype=float)
    out: List[Tuple[int, int]] = []

    def sec_to_raw(v):
        arr = np.round(v * raw_sfreq)
        return np.clip(arr, 0, raw_n - 1).astype(int)

    if units == "seconds":
        s_idx = sec_to_raw(s_vals)
        e_idx = sec_to_raw(e_vals)

    elif units == "samples":
        if detector_fs is None:
            raise ValueError("units='samples' requires detector_fs to scale indices to raw.")
        s_idx = _scale_indices_between_fs(s_vals, detector_fs, raw_sfreq, raw_n)
        e_idx = _scale_indices_between_fs(e_vals, detector_fs, raw_sfreq, raw_n)

    elif units == "auto":
        # Guess seconds; if unlikely, fall back to sample scaling (needs detector_fs).
        if _likely_seconds(np.r_[s_vals, e_vals], raw_n, raw_sfreq):
            s_idx = sec_to_raw(s_vals)
            e_idx = sec_to_raw(e_vals)
        else:
            if detector_fs is not None:
                s_idx = _scale_indices_between_fs(s_vals, detector_fs, raw_sfreq, raw_n)
                e_idx = _scale_indices_between_fs(e_vals, detector_fs, raw_sfreq, raw_n)
            else:
                # Final fallback: treat as seconds
                s_idx = sec_to_raw(s_vals)
                e_idx = sec_to_raw(e_vals)
    else:
        raise ValueError(f"Unknown units spec: {units}")

    for s, e in zip(s_idx, e_idx):
        s_i = int(max(0, min(raw_n, s)))
        e_i = int(max(0, min(raw_n, e)))
        if e_i > s_i:
            out.append((s_i, e_i))
    return out


def coerce_intervals_from_obj(
        obj: Any,
        raw_sfreq: float,
        raw_n: int,
        units: str,
        detector_fs: Optional[float],
) -> List[Tuple[int, int]]:
    """Coerce common schemas to [(start_sample_raw, end_sample_raw)] using the unit spec."""
    intervals: List[Tuple[int, int]] = []

    # list/tuple of dicts
    if isinstance(obj, (list, tuple)):
        if len(obj) and isinstance(obj[0], dict):
            start_keys = ["start", "start_idx", "start_sample", "blink_start", "s_idx", "start_frame", "start_time"]
            end_keys   = ["end", "end_idx", "end_sample", "blink_end", "e_idx", "end_frame", "end_time"]
            for d in obj:
                if not isinstance(d, dict): continue
                s_val = next((d[k] for k in start_keys if k in d), None)
                e_val = next((d[k] for k in end_keys if k in d), None)
                if s_val is None and "center" in d and ("duration" in d or "width" in d):
                    c = float(d["center"]); dur = float(d.get("duration", d.get("width", 0.0)))
                    s_val, e_val = c - dur/2, c + dur/2
                if s_val is not None and e_val is not None:
                    intervals.extend(_to_raw_indices([s_val], [e_val], units, raw_sfreq, raw_n, detector_fs))
            return intervals

        # list of [start, end]
        if len(obj) and isinstance(obj[0], (list, tuple)) and len(obj[0]) >= 2:
            s_vals = [pair[0] for pair in obj]
            e_vals = [pair[1] for pair in obj]
            return _to_raw_indices(s_vals, e_vals, units, raw_sfreq, raw_n, detector_fs)

    # dict of arrays
    if isinstance(obj, dict):
        candidates = [
            ("start", "end"),
            ("start_idx", "end_idx"),
            ("start_sample", "end_sample"),
            ("blink_start", "blink_end"),
            ("s_idx", "e_idx"),
            ("start_frame", "end_frame"),
            ("start_time", "end_time"),
        ]
        for ks, ke in candidates:
            if ks in obj and ke in obj:
                return _to_raw_indices(obj[ks], obj[ke], units, raw_sfreq, raw_n, detector_fs)
        if "center" in obj and ("duration" in obj or "width" in obj):
            centers = np.asarray(obj["center"], dtype=float)
            durs = np.asarray(obj.get("duration", obj.get("width", 0.0)), dtype=float)
            s_vals = centers - durs/2
            e_vals = centers + durs/2
            return _to_raw_indices(s_vals, e_vals, units, raw_sfreq, raw_n, detector_fs)

    # pandas-like
    if hasattr(obj, "columns"):
        df = pd.DataFrame(obj)
        cols = {c.lower(): c for c in df.columns}
        for sk, ek in [("start","end"), ("start_idx","end_idx"),
                       ("start_sample","end_sample"), ("blink_start","blink_end"),
                       ("start_time","end_time")]:
            if sk in cols and ek in cols:
                return _to_raw_indices(df[cols[sk]].values, df[cols[ek]].values,
                                       units, raw_sfreq, raw_n, detector_fs)
        if "center" in cols and ("duration" in cols or "width" in cols):
            centers = df[cols["center"]].values.astype(float)
            durs = df[cols.get("duration", cols.get("width"))].values.astype(float)
            s_vals = centers - durs/2
            e_vals = centers + durs/2
            return _to_raw_indices(s_vals, e_vals, units, raw_sfreq, raw_n, detector_fs)

    return intervals


# -------------------------------------------------------------------
# Masking + metrics
# -------------------------------------------------------------------
def intervals_to_mask(intervals: List[Tuple[int, int]], n_samples: int) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=bool)
    for s, e in intervals:
        s = max(0, min(n_samples, int(s)))
        e = max(0, min(n_samples, int(e)))
        if e > s:
            mask[s:e] = True
    return mask


def confusion_from_masks(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tp = int(np.sum( y_true &  y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true &  y_pred))
    fn = int(np.sum( y_true & ~y_pred))
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def prf(conf: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn = conf["TP"], conf["FP"], conf["FN"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2*precision*recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# -------------------------------------------------------------------
# Event-level matching (start/end aware)
# -------------------------------------------------------------------
def iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    s1, e1 = a; s2, e2 = b
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def greedy_match(
        preds: List[Tuple[int, int]],
        gts: List[Tuple[int, int]],
        min_iou: float,
        start_tol_samps: int,
        end_tol_samps: int,
):
    used_gt = set()
    matches = []
    for p_idx, p in enumerate(preds):
        best = (-1, -1.0)  # (g_idx, score)
        for g_idx, g in enumerate(gts):
            if g_idx in used_gt:
                continue
            j = iou(p, g)
            s_err = abs(p[0] - g[0]); e_err = abs(p[1] - g[1])
            prox_ok = (s_err <= start_tol_samps) and (e_err <= end_tol_samps)
            score = max(j, 1.0 if prox_ok else 0.0)
            if score > best[1]:
                best = (g_idx, score)
        if best[0] >= 0:
            g_idx = best[0]; g = gts[g_idx]
            s_err = p[0] - g[0]; e_err = p[1] - g[1]
            if (iou(p, g) >= min_iou) or (abs(s_err) <= start_tol_samps and abs(e_err) <= end_tol_samps):
                matches.append((p_idx, g_idx, s_err, e_err))
                used_gt.add(g_idx)

    unpaired_pred = [i for i in range(len(preds)) if i not in [m[0] for m in matches]]
    unpaired_gt   = [i for i in range(len(gts))   if i not in [m[1] for m in matches]]
    return matches, unpaired_pred, unpaired_gt


def event_metrics(
        preds: List[Tuple[int, int]],
        gts: List[Tuple[int, int]],
        start_tol_samps: int,
        end_tol_samps: int,
        min_iou: float = 0.10,
) -> Dict[str, Any]:
    matches, unpaired_pred, unpaired_gt = greedy_match(
        preds, gts, min_iou=min_iou, start_tol_samps=start_tol_samps, end_tol_samps=end_tol_samps
    )
    tp = len(matches); fp = len(unpaired_pred); fn = len(unpaired_gt)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2*precision*recall / (precision + recall)) if (precision + recall) else 0.0

    if tp:
        start_errs = np.array([m[2] for m in matches], dtype=float)
        end_errs   = np.array([m[3] for m in matches], dtype=float)
        mae_start  = float(np.mean(np.abs(start_errs)))
        mae_end    = float(np.mean(np.abs(end_errs)))
    else:
        mae_start = mae_end = float("nan")

    return {
        "TP": tp, "FP": fp, "FN": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "matches": matches,
        "mae_start_samples": mae_start,
        "mae_end_samples": mae_end,
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    # 1) Load raw & ground truth
    raw = mne.io.read_raw_fif(str(FIF_PATH), preload=False, verbose="ERROR")
    raw_sfreq = float(raw.info["sfreq"])
    raw_n = int(raw.n_times)

    print(f"✅ Loaded ground truth: {FIF_PATH}")
    print(f"ℹ️ Raw sampling rate: {raw_sfreq:.3f} Hz | n_samples: {raw_n}")

    gt_intervals = load_ground_truth_intervals(raw)
    print(f"ℹ️ Ground-truth blinks: {len(gt_intervals)}")
    gt_mask = intervals_to_mask(gt_intervals, raw_n)

    # 2) Load detections
    blinker_obj   = load_pickle(BLINKER_PKL)
    pyblinker_obj = load_pickle(PYBLINKER_PKL)

    # 3) Normalize to raw-sample space using hardcoded format controls
    blinker_ints = coerce_intervals_from_obj(
        blinker_obj, raw_sfreq, raw_n, units=BLINKER_UNITS, detector_fs=BLINKER_FS
    )
    pyblinker_ints = coerce_intervals_from_obj(
        pyblinker_obj, raw_sfreq, raw_n, units=PYBLINKER_UNITS, detector_fs=PYBLINKER_FS
    )

    print(f"ℹ️ Parsed blinker intervals:   {len(blinker_ints)}")
    print(f"ℹ️ Parsed pyblinker intervals: {len(pyblinker_ints)}")

    # 4) Per-sample metrics (with TN)
    print("\n===== Per-sample metrics =====")
    for name, pred_ints in [("blinker", blinker_ints), ("pyblinker", pyblinker_ints)]:
        pred_mask = intervals_to_mask(pred_ints, raw_n)
        conf = confusion_from_masks(gt_mask, pred_mask)
        stats = prf(conf)
        print(f"\n[{name}]")
        print(f"TP={conf['TP']}, TN={conf['TN']}, FP={conf['FP']}, FN={conf['FN']}")
        print(f"Precision={stats['precision']:.4f} | Recall={stats['recall']:.4f} | F1={stats['f1']:.4f}")

    # 5) Event-level metrics (start/end aware)
    start_tol_samps = max(1, int(round(0.050 * raw_sfreq)))  # 50 ms
    end_tol_samps   = max(1, int(round(0.050 * raw_sfreq)))
    min_iou = 0.10

    print("\n===== Event-level metrics (start/end aware) =====")
    for name, pred_ints in [("blinker", blinker_ints), ("pyblinker", pyblinker_ints)]:
        em = event_metrics(pred_ints, gt_intervals, start_tol_samps, end_tol_samps, min_iou=min_iou)
        print(f"\n[{name}]")
        print(f"TP={em['TP']}, FP={em['FP']}, FN={em['FN']}")
        print(f"Precision={em['precision']:.4f} | Recall={em['recall']:.4f} | F1={em['f1']:.4f}")
        print(f"MAE start (samples)={em['mae_start_samples']:.2f} | MAE end (samples)={em['mae_end_samples']:.2f}")


if __name__ == "__main__":
    main()
