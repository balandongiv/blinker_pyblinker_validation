# Fresh Top-10 Exp01 Validation

Date/time: 2026-03-14 00:34:36 +08:00

## Hypothesis
The previously reported 100% agreement for the staged rollout is real, but the current `blinker_pyblinker_validation/blink_compare_from_csv.py` path can still read stale `pyblinker_results.pkl` artifacts because it hardcodes that filename and leaves `OVERWRITE = False`. A fresh runner that writes a new prefixed pickle for each subject should confirm whether the refactored pipeline still produces 100% agreement on the top 10 subjects when no old PyBlinker result files are reused.

## Files inspected
- `blinker_pyblinker_validation/blink_compare_from_csv.py`
- `blinker_pyblinker_validation/blink_compare.py`
- `blinker_pyblinker_validation/finding/004_validate_top2_after_fix.py`
- `blinker_pyblinker_validation/summary_metrics.csv`

## Files changed
- `blinker_pyblinker_validation/fresh_compare_from_csv.py`
- `blinker_pyblinker_validation/finding/006_fresh_top10_exp01_validation.md`

## Exact change made
Added a new experiment runner that:
- loads the first `N` subjects from `summary_metrics.csv`
- runs a fresh `BlinkDetector` execution for each subject
- writes a new prefixed pickle file like `exp01_pyblinker_results.pkl` into the subject folder
- compares that fresh artifact against `blinker_results.pkl`
- writes aggregate summary outputs for the experiment

## Why the change was made
We need a validation path that cannot silently reuse old `pyblinker_results.pkl` files. This makes the refactoring claim testable on fresh outputs.

## MATLAB reference used
- `D:/code development/matlab_plugin/eeglab2025.1.0/plugins/Blinker1.2.0/utilities/getBlinkPositions.m`
- `D:/code development/matlab_plugin/eeglab2025.1.0/plugins/Blinker1.2.0/extractBlinks.m`
- Stored MATLAB output files: `blinker_results.pkl` for each subject

## Validation scope
- Subjects: `9636595`, `12400406`, `12400412`, `9636607`, `12400409`, `9636496`, `9636577`, `9636487`, `9636484`, `9636580`
- Subject count: 10
- Tests: fresh top-10 experiment runner execution

## Before/after metrics
- Before:
  - stale comparison path could reuse `pyblinker_results.pkl`
- After first fresh rerun:
  - `9636595`, `12400406`, `12400412`, `9636607`, `12400409`, `9636577`, `9636487`, `9636484`, `9636580`: `share_within_tolerance_percent = 100.0`
  - `9636496`: `share_within_tolerance_percent = 99.95359628770302`
  - aggregate output files written:
    - `blinker_pyblinker_validation/experiment_results/exp01_top10_summary.csv`
    - `blinker_pyblinker_validation/experiment_results/exp01_top10_overall.json`

## Whether the change was kept or reverted
- Kept

## Next recommended step
Investigate subject `9636496` from the fresh rerun, because it is the only top-10 subject below 100% and therefore the first fresh divergence point.
