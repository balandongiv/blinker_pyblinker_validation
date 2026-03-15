# Py Event Index Correction

Date/time: 2026-03-14 00:34:36 +08:00

## Hypothesis
Even after fixing the comparison aligner, subject `9636496` still shows `share_within_tolerance_percent < 100` because the PyBlinker comparison path is likely reading `left_zero` and `right_zero` as if they were already MATLAB-style one-based boundaries. The MATLAB compatibility tests suggest those columns must be shifted by `+1` before direct comparison.

## Files inspected
- `blinker_pyblinker_validation/blink_compare.py`
- `test/blinker_pyblinker_comparison/test_b_fitblink.py`
- `D:/dataset/murat_2018/9636496/exp01_pyblinker_results.pkl`
- `D:/dataset/murat_2018/9636496/blinker_results.pkl`

## Files changed
- `blinker_pyblinker_validation/blink_compare.py`
- `test/blinker_pyblinker_comparison/test_e_prepare_event_tables.py`
- `blinker_pyblinker_validation/finding/008_py_event_index_correction.md`

## Exact change made
Updated `prepare_event_tables(...)` to convert PyBlinker `left_zero` and `right_zero` values from zero-based to one-based sample indices before building the comparison event table. Added a regression test that verifies a zero-based PyBlinker event `[9, 19]` becomes `[10, 20]` to match MATLAB blink fits.

## Why the change was made
The fresh `9636496` rerun showed a consistent `-1` sample offset for both start and end boundaries across PyBlinker events. The existing MATLAB comparison test for `FitBlinks` already documents that these boundary columns require a `+1` correction when compared against MATLAB output.

## MATLAB reference used
- `test/blinker_pyblinker_comparison/test_b_fitblink.py`
- Stored MATLAB output file: `D:/dataset/murat_2018/9636496/blinker_results.pkl`

## Validation scope
- Subjects: `9636595`, `12400406`, `12400412`, `9636607`, `12400409`, `9636496`, `9636577`, `9636487`, `9636484`, `9636580`
- Subject count: 10
- Tests:
  - `test/blinker_pyblinker_comparison/test_a2_stat.py`
  - `test/blinker_pyblinker_comparison/test_a_get_blink_position.py`
  - `test/blinker_pyblinker_comparison/test_b_fitblink.py`
  - `test/blinker_pyblinker_comparison/test_c_BlinkProperties.py`
  - `test/blinker_pyblinker_comparison/test_d_alignment_comparison.py`
  - `test/blinker_pyblinker_comparison/test_e_prepare_event_tables.py`

## Before/after metrics
- Before: subject `9636496` fresh rerun at `99.95359628770302`, with all paired events shifted by `-1` sample in both boundaries
- After:
  - focused tests: `6 passed`
  - fresh top-10 rerun:
    - all 10 subjects: `share_within_tolerance_percent = 100.0`
    - all tracked strict and lenient metrics: `1.0` / `100.0` as applicable
  - overall:
    - `share_within_tolerance_total = 26144.0`
    - `matches_within_tolerance_total = 0.0`
    - `detected_only_total = 0.0`
    - `ground_truth_only_total = 0.0`
    - `precision_strict_micro = 1.0`
    - `recall_strict_micro = 1.0`
    - `f1_strict_micro = 1.0`
    - `accuracy_strict_micro = 1.0`

## Whether the change was kept or reverted
- Kept

## Next recommended step
Scale the same fresh-run experiment runner beyond the top 10 in controlled batches, now that the comparison path and fresh-output path are both validated.
