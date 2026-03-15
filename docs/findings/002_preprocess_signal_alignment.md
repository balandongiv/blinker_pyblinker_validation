# First Divergence Is Preprocessing Input, Not `get_blink_position`

## Date/time
2026-03-13T23:29:49+08:00

## Hypothesis
The mismatch on real subjects starts before `process_channel_data(...)` step 1 because Python is feeding a different filtered candidate signal into `get_blink_position(...)` than MATLAB Blinker uses.

## Files inspected
- `pyblinker/blinker/get_blink_positions.py`
- `pyblinker/blinker/pyblinker.py`
- `pyblinker/utils/evaluation/blink_detection.py`
- `D:/code development/matlab_plugin/eeglab2025.1.0/plugins/Blinker1.2.0/utilities/extractBlinks.m`
- `D:/code development/matlab_plugin/eeglab2025.1.0/plugins/Blinker1.2.0/utilities/getCandidateSignals.m`
- `D:/code development/matlab_plugin/eeglab2025.1.0/plugins/firfilt/pop_eegfiltnew.m`
- `D:/code development/matlab_plugin/eeglab2025.1.0/plugins/firfilt/firfilt.m`
- `D:/code development/matlab_plugin/eeglab2025.1.0/plugins/firfilt/fir_filterdcpadded.m`
- `D:/code development/matlab_plugin/eeglab2025.1.0/plugins/firfilt/firws.m`

## Files changed
- `blinker_pyblinker_validation/finding/002_preprocess_signal_alignment.md`

## Exact change made
Created a second investigation log to capture the preprocessing-divergence evidence and the next probe toward an EEGLAB-equivalent filter path.

## Why the change was made
The required debugging history must record the first confirmed divergence point and preserve the investigation trail before any pipeline edits.

## MATLAB reference used
- `extractBlinks.m`
- `getCandidateSignals.m`
- `pop_eegfiltnew.m`
- `firfilt.m`
- `fir_filterdcpadded.m`
- `firws.m`

## Validation scope
- Subjects: `9636595`, `12400406`
- Subject count: 2
- Tests: baseline comparison tests from log 001; focused stage-by-stage shell probes

## Before/after metrics
Stage-isolation results before any preprocessing fix:

- Running Python `get_blink_position(...)` on MATLAB `signalData.signal` reproduced MATLAB `blinkPositions` exactly for both top subjects and both channels.
- Running Python `get_blink_position(...)` on Python-preprocessed raw signals underdetected candidate blink positions:
  - `9636595`: `CH1` Python `764` vs MATLAB `883`; `CH2` Python `669` vs MATLAB `819`
  - `12400406`: `CH1` Python `807` vs MATLAB `849`; `CH2` Python `805` vs MATLAB `852`
- Signal parity against MATLAB `signalData.signal` improved after Python filtering versus raw, but still remained imperfect:
  - `9636595`: filtered correlation `0.9503` / `0.9523` for `CH1` / `CH2`
  - `12400406`: filtered correlation `0.9719` / `0.9748` for `CH1` / `CH2`
- Resampling was not the differentiator in this dataset slice; subjects were already effectively at `200 Hz`, and `filter` vs `filter+resample` produced the same parity metrics in the probe.

## Whether the change was kept or reverted
Kept.

## Next recommended step
Prototype an EEGLAB-style `pop_eegfiltnew` equivalent in Python, compare the resulting signal and `get_blink_position(...)` outputs against MATLAB on the top subjects, and keep the change only if it closes the step-1 gap without breaking the focused comparison tests.
