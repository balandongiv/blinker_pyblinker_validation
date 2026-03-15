# 014 FitBlinks Empty Candidate Guard

1. Title
Guard `FitBlinks` against empty candidate frames so driving-dataset channels with no blink positions do not crash the pipeline

2. Date/time
2026-03-14 15:05 Asia/Kuala_Lumpur

3. Hypothesis
The first driving-dataset expansion failure on `S2` is not a MATLAB/Python blink-logic mismatch. It is a Python runtime edge case: when a channel has no candidate blink rows, `FitBlinks.dprocess()` still tries to assign the result of `DataFrame.apply(...)` into multiple columns, which raises `ValueError: Columns must be same length as key`. If this edge case is handled cleanly, the driving run should continue and the existing murat_2018 results should remain unchanged.

4. Files inspected
- `c:\Users\balan\IdeaProjects\pyblinker\pyblinker\blinker\fit_blink.py`
- `c:\Users\balan\IdeaProjects\pyblinker\pyblinker\pipeline_steps.py`
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\fresh_compare_subjects.py`
- `D:\dataset\drowsy_driving_raja_processed\S2\blinker_pyblinker_validation\blinker_results.pkl`

5. Files changed
- `c:\Users\balan\IdeaProjects\pyblinker\pyblinker\blinker\fit_blink.py`
- `c:\Users\balan\IdeaProjects\pyblinker\test\blinker_pyblinker_comparison\test_g_fitblink_empty_candidates.py`

6. Exact change made
Added an `_empty_frame_blinks()` helper in `FitBlinks` and short-circuited both `dprocess()` and `dprocess_segment_raw()` when `self.df` is `None` or empty. The empty return frame includes the downstream-required columns such as `left_zero`, `right_zero`, `max_value`, `leftR2`, and `rightR2`, so `get_blink_statistic()` and `get_good_blink_mask()` can safely treat the channel as having no valid blinks.

7. Why the change was made
`S2` contains at least one channel (`VREF`) with no blink candidates. MATLAB simply leaves that candidate empty and moves on. Python instead crashed before channel selection, which blocked the driving-dataset rollout even though the actual representative blink channel matched MATLAB.

8. MATLAB reference used
- MATLAB `blinker_results.pkl` files from:
  - `D:\dataset\drowsy_driving_raja_processed\S1\blinker_pyblinker_validation`
  - `D:\dataset\drowsy_driving_raja_processed\S2\blinker_pyblinker_validation`
- Existing murat_2018 MATLAB comparison targets under `D:\dataset\murat_2018\*`

9. Validation scope
- Which subjects:
  - driving dataset `S1`
  - driving dataset `S2`
  - murat_2018 full ordered `74` CSV subjects
- How many subjects:
  - `2` driving subjects
  - `74` murat_2018 subjects
- Which tests:
  - `python -m pytest test/blinker_pyblinker_comparison -q`
  - `python -m blinker_pyblinker_validation.fresh_compare_subjects --dataset driving_dataset --prefix drvexp02 --max-subjects 2`
  - `python -m blinker_pyblinker_validation.fresh_compare_from_csv --n 74 --selection top --prefix exp05`

10. Before/after metrics
Before:
- Focused comparison tests: `7 passed`
- `drvexp01`:
  - `S1 = 100.0`
  - `S2` crashed in `FitBlinks.dprocess()` with `ValueError: Columns must be same length as key`
- murat_2018 previous validated scope: `exp04_top74_summary.csv` was `74/74` at `100.0`

After:
- Focused comparison tests: `8 passed`
- `drvexp02`:
  - `S1 = 100.0`
  - `S2 = 100.0`
  - selected channel on `S2`: `eog_vert_left`
- murat_2018 `exp05_top74_summary.csv`: `74/74` at `100.0`
- murat_2018 `exp05_top74_overall.json`:
  - `total_detected_total = 72568`
  - `total_ground_truth_total = 72568`
  - `detected_only_total = 0`
  - `ground_truth_only_total = 0`
  - strict/lenient micro `precision`, `recall`, `f1`, and `accuracy` all `1.0`

11. Whether the change was kept or reverted
Kept. The fix is minimal, reproduces MATLAB’s empty-channel behavior, and preserves the previously validated murat_2018 results.

12. Next recommended step
Continue the driving-dataset rollout with the new prefix `drvexp02` and stop at the first non-100 subject. If a mismatch appears, compare the first divergent subject against MATLAB step-by-step before changing any additional shared logic.
