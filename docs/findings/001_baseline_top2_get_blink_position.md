# Baseline on Top-2 Subjects and Initial `get_blink_position` Review

## Date/time
2026-03-13T23:29:49+08:00

## Hypothesis
The first observable divergence between MATLAB Blinker and `pyblinker` on real subjects may appear before final blink-region export, potentially starting in `get_blink_position(...)` or in how its output is consumed by the downstream six-step pipeline.

## Files inspected
- `blinker_pyblinker_validation/summary_metrics.csv`
- `blinker_pyblinker_validation/blink_compare_from_csv.py`
- `blinker_pyblinker_validation/blink_compare.py`
- `blinker_pyblinker_validation/stat.py`
- `pyblinker/blinker/get_blink_positions.py`
- `pyblinker/pipeline_steps.py`
- `test/blinker_pyblinker_comparison/test_a_get_blink_position.py`
- `test/blinker_pyblinker_comparison/test_a2_stat.py`
- `test/blinker_pyblinker_comparison/matlab_fitblink/getBlinkPositions.m`
- `D:/code development/matlab_plugin/eeglab2025.1.0/plugins/Blinker1.2.0/utilities/getBlinkPositions.m`

## Files changed
- `blinker_pyblinker_validation/finding/001_baseline_top2_get_blink_position.md`

## Exact change made
Created the initial investigation log to capture the requested baseline, evidence trail, and next steps before any behavior-changing code edits.

## Why the change was made
The migration task requires a resumable markdown record for every meaningful investigation or fix attempt, including failed ideas and baseline measurements.

## MATLAB reference used
- `test/blinker_pyblinker_comparison/matlab_fitblink/getBlinkPositions.m`
- `D:/code development/matlab_plugin/eeglab2025.1.0/plugins/Blinker1.2.0/utilities/getBlinkPositions.m`

## Validation scope
- Subjects: `9636595`, `12400406`
- Subject count: 2
- Tests:
  - `test/blinker_pyblinker_comparison/test_a2_stat.py`
  - `test/blinker_pyblinker_comparison/test_a_get_blink_position.py`
  - `test/blinker_pyblinker_comparison/test_b_fitblink.py`
  - `test/blinker_pyblinker_comparison/test_c_BlinkProperties.py`
  - `python -m blinker_pyblinker_validation.blink_compare_from_csv`

## Before/after metrics
Baseline before any behavior change:

| recording_id | share_within_tolerance_percent | precision_strict | recall_strict | f1_strict | accuracy_strict | precision_lenient | recall_lenient | f1_lenient | accuracy_lenient |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `9636595` | 97.9109900091 | 0.9953831948 | 0.9835766423 | 0.9894446994 | 0.9791099001 | 0.9953831948 | 0.9835766423 | 0.9894446994 | 0.9791099001 |
| `12400406` | 97.4825174825 | 0.9907604833 | 0.9837685251 | 0.9872521246 | 0.9748251748 | 0.9907604833 | 0.9837685251 | 0.9872521246 | 0.9748251748 |

Focused comparison tests baseline:
- `4 passed in 3.69s`

Environment note:
- The first baseline attempt failed inside the sandbox because MNE could not open `C:\Users\balan\.mne\mne-python.json`.
- The rerun outside the sandbox succeeded and produced the metrics above.

## Whether the change was kept or reverted
Kept.

## Next recommended step
Compare MATLAB vs Python intermediate outputs starting with `get_blink_position(...)`, then continue downstream only if step 1 still matches on the real-subject artifacts.
