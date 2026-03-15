# Probe MATLAB `firfilt` Against Stored Subject Signals

## Date/time
2026-03-13T23:29:49+08:00

## Hypothesis
The remaining preprocessing mismatch is due to Python matching MATLAB `fir_filterdcpadded` while the legacy dataset signal was generated through MATLAB `firfilt`, which applies the same FIR coefficients with different state initialization and delay handling.

## Files inspected
- `D:/code development/matlab_plugin/eeglab2025.1.0/plugins/firfilt/firfilt.m`
- `D:/code development/matlab_plugin/eeglab2025.1.0/plugins/firfilt/fir_filterdcpadded.m`
- `D:/code development/matlab_plugin/eeglab2025.1.0/plugins/firfilt/findboundaries.m`
- `blinker_pyblinker_validation/finding/002_probe_eeglab_filter.py`

## Files changed
- `blinker_pyblinker_validation/finding/003_matlab_firfilt_probe.md`
- `blinker_pyblinker_validation/finding/003_probe_matlab_firfilt.py`

## Exact change made
Added a MATLAB-engine probe script that compares:
- Python FIR coefficients vs MATLAB `firws`
- Python DC-padded zero-phase application vs MATLAB `fir_filterdcpadded`
- Python filtered signals vs MATLAB `firfilt`
- all of the above vs stored `signalData.signal`

The probe was then extended with CLI cutoff arguments so the same path could be checked for both the Python-side `0.5-30 Hz` settings and the legacy MATLAB `1-20 Hz` settings.

## Why the change was made
After proving that Python matched MATLAB `firws` coefficient generation and `fir_filterdcpadded`, the remaining uncertainty was isolated to the exact filter-application path and preprocessing parameters used by the stored MATLAB subject outputs.

## MATLAB reference used
- `firfilt.m`
- `fir_filterdcpadded.m`
- `findboundaries.m`

## Validation scope
- Subjects: `9636595`, `12400406`
- Subject count: 2
- Tests: stage-level preprocessing probes only

## Before/after metrics
MATLAB-engine probe results:

- Python FIR coefficients matched MATLAB `firws` exactly:
  - coefficient length `1321`
  - maximum absolute coefficient difference `2.22e-16`
- Python `eeglab_zero_phase_dc_filter(...)` matched MATLAB `fir_filterdcpadded` exactly:
  - maximum absolute signal difference `3.64e-12` on the checked subject/channel
- Python `eeglab_zero_phase_dc_filter(...)` also matched MATLAB `firfilt` exactly for continuous data:
  - `py_dc_vs_matlab_firfilt` correlation `1.0`
  - near-zero max absolute differences on the top two subjects/channels
- However, both Python and MATLAB `firfilt` using the Python-side `0.5-30 Hz` settings still disagreed with the stored MATLAB `signalData.signal`:
  - `9636595` `CH1`: `798` vs stored `883`
  - `9636595` `CH2`: `720` vs stored `819`
  - `12400406` `CH1`: `813` vs stored `849`
  - `12400406` `CH2`: `816` vs stored `852`
- Inspecting stored MATLAB params revealed the legacy subject outputs used `lowCutoffHz=1.0` and `highCutoffHz=20.0`.
- Re-running the exact Python/Matlab `firfilt` path with `1-20 Hz` reproduced the stored `signalData.signal` and `blinkPositions` on both top subjects:
  - `9636595`: `CH1` `883/883`, `CH2` `819/819`
  - `12400406`: `CH1` `849/849`, `CH2` `852/852`

## Whether the change was kept or reverted
Kept.

## Next recommended step
Patch the detector preprocessing to use the exact legacy EEGLAB FIR path with legacy `1-20 Hz` defaults, then rerun the focused tests and staged subject validation.
