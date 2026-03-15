# Align Detector Preprocessing With Legacy MATLAB Blinker

## Date/time
2026-03-13T23:29:49+08:00

## Hypothesis
Real-subject mismatches come from Python preprocessing drift before `process_channel_data(...)`: the detector uses non-legacy filter cutoffs and leaves `params["sfreq"]` at the static default instead of the processed signal rate.

## Files inspected
- `pyblinker/blinker/pyblinker.py`
- `pyblinker/blinker/default_setting.py`
- `pyblinker/utils/evaluation/blink_detection.py`
- `pyblinker/blinker/legacy_eeglab_filter.py`
- `blinker_pyblinker_validation/finding/002_probe_eeglab_filter.py`
- `blinker_pyblinker_validation/finding/003_probe_matlab_firfilt.py`
- `D:/code development/matlab_plugin/eeglab2025.1.0/plugins/Blinker1.2.0/utilities/getBlinkerDefaults.m`
- `D:/code development/matlab_plugin/eeglab2025.1.0/plugins/firfilt/firfilt.m`

## Files changed
- `pyblinker/blinker/legacy_eeglab_filter.py`
- `pyblinker/blinker/pyblinker.py`
- `pyblinker/utils/evaluation/blink_detection.py`
- `blinker_pyblinker_validation/finding/003_probe_matlab_firfilt.py`
- `blinker_pyblinker_validation/finding/004_legacy_preprocess_fix.md`

## Exact change made
- Added `pyblinker/blinker/legacy_eeglab_filter.py` with an exact Python port of the legacy EEGLAB FIR bandpass design/application used by MATLAB Blinker.
- Updated `pyblinker/blinker/pyblinker.py` to:
  - use legacy default cutoffs `1.0-20.0 Hz`
  - filter selected channels with the legacy FIR helper instead of the MNE FIR helper
  - refresh `channel_list` after channel picking
  - synchronize `self.sfreq` and `self.params["sfreq"]` with the processed raw data
- Updated `pyblinker/utils/evaluation/blink_detection.py` to default to the same legacy `1.0-20.0 Hz` passband.
- Added `blinker_pyblinker_validation/finding/004_validate_top2_after_fix.py` to compare fresh detector output against stored MATLAB results without overwriting dataset pickles.

## Why the change was made
The stored MATLAB subject outputs were produced with legacy `1-20 Hz` preprocessing, and the detector was both using different cutoffs and leaving `params["sfreq"]` at the static default. Those mismatches changed candidate blink detection before downstream fitting and masking could even be compared fairly.

## MATLAB reference used
- `getBlinkerDefaults.m`
- `firfilt.m`
- `getBlinkPositions.m`

## Validation scope
- Subjects: `9636595`, `12400406`
- Subject count: 2
- Tests:
  - `test/blinker_pyblinker_comparison/test_a2_stat.py`
  - `test/blinker_pyblinker_comparison/test_a_get_blink_position.py`
  - `test/blinker_pyblinker_comparison/test_b_fitblink.py`
  - `test/blinker_pyblinker_comparison/test_c_BlinkProperties.py`
  - top-two subject validation via `004_validate_top2_after_fix.py`

## Before/after metrics
Before:
- Baseline top-two metrics from log 001.

After:
- Focused comparison tests: `4 passed`
- Top-two subject validation immediately after this preprocessing fix:
  - `9636595`: `100.0`
  - `12400406`: `99.0990990991`

This fix removed the preprocessing mismatch completely, but a remaining representative-channel selection bug still prevented the second subject from reaching `100.0`.

## Whether the change was kept or reverted
Kept.

## Next recommended step
Fix representative-channel selection so the row explicitly marked as selected is the one returned to downstream code.
