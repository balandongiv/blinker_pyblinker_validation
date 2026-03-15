# Return the Selected Representative Channel Row

## Date/time
2026-03-13T23:29:49+08:00

## Hypothesis
After preprocessing is aligned, the remaining mismatch is caused by representative-channel selection returning all candidate rows in their original order instead of the row marked as selected, so downstream code picks `loc[0]` rather than MATLAB's true `usedSignal`.

## Files inspected
- `pyblinker/blinker/get_representative_channel.py`
- `pyblinker/pipeline_steps.py`
- `D:/code development/matlab_plugin/eeglab2025.1.0/plugins/Blinker1.2.0/utilities/extractBlinks.m`

## Files changed
- `pyblinker/blinker/get_representative_channel.py`
- `blinker_pyblinker_validation/finding/005_channel_selection_selected_row_fix.md`

## Exact change made
Updated `channel_selection(...)` so that once MATLAB-style selection marks the chosen row with `select=True`, the function returns only that selected row before dropping helper columns.

## Why the change was made
The top-two validation narrowed the remaining miss to a case where Python and MATLAB channel statistics matched exactly, but Python still chose the wrong channel because selection metadata was computed and then ignored by downstream code.

## MATLAB reference used
- `extractBlinks.m`

## Validation scope
- Subjects: `12400406` primary; `9636595` as regression check
- Subject count: 10
- Tests:
  - `test/blinker_pyblinker_comparison/test_a2_stat.py`
  - `test/blinker_pyblinker_comparison/test_a_get_blink_position.py`
  - `test/blinker_pyblinker_comparison/test_b_fitblink.py`
  - `test/blinker_pyblinker_comparison/test_c_BlinkProperties.py`
  - staged subject validation on top 2, top 5, and top 10 sorted recordings

## Before/after metrics
Before patch:
- `12400406` channel statistics matched MATLAB exactly, but Python selected `CH1` even though MATLAB `usedSignal` corresponded to channel 2.
- Root cause: `channel_selection(...)` left the selected row marked with `select=True` but returned the full DataFrame; downstream code used row 0.

After patch:
- Focused comparison tests: `4 passed`
- Top 2 subjects:
  - `9636595`: `100.0`
  - `12400406`: `100.0`
- Top 5 subjects:
  - `9636595`, `12400406`, `12400412`, `9636607`, `12400409`: all `100.0`
- Top 10 subjects:
  - `9636595`, `12400406`, `12400412`, `9636607`, `12400409`, `9636496`, `9636577`, `9636487`, `9636484`, `9636580`: all `100.0`

## Whether the change was kept or reverted
Kept.

## Next recommended step
If broader dataset coverage is desired, continue the staged rollout beyond the top 10 subjects using the same validation helper; no remaining mismatch was observed in the current validated scope.
