# Alignment Amplitude Gate Fix

Date/time: 2026-03-14 00:34:36 +08:00

## Hypothesis
The fresh top-10 rerun exposed one remaining mismatch on subject `9636496`, but the detected blink boundaries themselves may already be within tolerance. The likely bug is in the comparison alignment logic: events with tolerance-expanded overlap are being dropped entirely when their sampled peak amplitudes differ slightly, instead of being paired and classified as `matches_within_tolerance`.

## Files inspected
- `pyblinker/utils/evaluation/similarity.py`
- `pyblinker/utils/evaluation/blink_comparison.py`
- `pyblinker/utils/evaluation/reporting.py`
- `D:/dataset/murat_2018/9636496/exp01_pyblinker_results.pkl`
- `D:/dataset/murat_2018/9636496/blinker_results.pkl`

## Files changed
- `pyblinker/utils/evaluation/similarity.py`
- `test/blinker_pyblinker_comparison/test_d_alignment_comparison.py`
- `blinker_pyblinker_validation/finding/007_alignment_amplitude_gate_fix.md`

## Exact change made
Changed candidate generation in the comparison aligner so that overlap-based candidate pairs are still considered even when amplitude similarity fails. The amplitude condition is now used only to decide whether a paired event is counted as `share_within_tolerance` or downgraded to `matches_within_tolerance`, rather than blocking the pair entirely. Added a regression test covering a one-sample boundary shift whose peak amplitude changes at the extra boundary sample.

## Why the change was made
For subject `9636496`, the fresh detected blink `[227800, 227958]` and the MATLAB blink `[227801, 227959]` overlap and fall within the configured tolerance, but the comparison layer marked them as one detected-only event plus one ground-truth-only event because the peak amplitude changed slightly at the shifted boundary. That is a comparison bug and it understates true blink-region agreement.

## MATLAB reference used
- Stored MATLAB output file: `D:/dataset/murat_2018/9636496/blinker_results.pkl`

## Validation scope
- Subjects: `9636496` first, then fresh top 10 rerun
- Subject count: 10
- Tests:
  - `test/blinker_pyblinker_comparison/test_a2_stat.py`
  - `test/blinker_pyblinker_comparison/test_a_get_blink_position.py`
  - `test/blinker_pyblinker_comparison/test_b_fitblink.py`
  - `test/blinker_pyblinker_comparison/test_c_BlinkProperties.py`
  - `test/blinker_pyblinker_comparison/test_d_alignment_comparison.py`

## Before/after metrics
- Before: subject `9636496` fresh rerun at `99.95359628770302`, with `ground_truth_only=1` and `detected_only=1`
- After:
  - focused tests: `5 passed`
  - subject `9636496`: `ground_truth_only=0`, `detected_only=0`, `matches_within_tolerance=2`, `share_within_tolerance_percent=99.95359628770302`
  - aggregate top 10 still not fully strict-100 because one subject remained in `matches_within_tolerance`

## Whether the change was kept or reverted
- Kept

## Next recommended step
Investigate why subject `9636496` still has `matches_within_tolerance=2`; the most likely source is an indexing mismatch in how PyBlinker event boundaries are converted before comparison.
