# 011 Subject 9636514 Terminal Fit Edge Case

1. Title
Subject `9636514`: terminal blink fit row should stay invalid when `rightZero` reaches the final sample

2. Date/time
2026-03-14 09:12 Asia/Kuala_Lumpur

3. Hypothesis
The remaining mismatch is caused by Python keeping the final fitted candidate valid when MATLAB aborts that blink inside `fitBlinks.m`. That extra valid row shifts `bestMedian` and `bestRobustStd`, which in turn allows one borderline blink to pass `getGoodBlinkMask`.

4. Files inspected
- `D:\code development\matlab_plugin\eeglab2025.1.0\plugins\Blinker1.2.0\utilities\fitBlinks.m`
- `D:\code development\matlab_plugin\eeglab2025.1.0\plugins\Blinker1.2.0\utilities\extractBlinks.m`
- `D:\code development\matlab_plugin\eeglab2025.1.0\plugins\Blinker1.2.0\utilities\getGoodBlinkMask.m`
- `c:\Users\balan\IdeaProjects\pyblinker\pyblinker\blinker\fit_blink.py`
- `c:\Users\balan\IdeaProjects\pyblinker\pyblinker\segmentation\geometry.py`
- `c:\Users\balan\IdeaProjects\pyblinker\pyblinker\blinker\stroke_utils.py`

5. Files changed
- `c:\Users\balan\IdeaProjects\pyblinker\pyblinker\blinker\fit_blink.py`
- `c:\Users\balan\IdeaProjects\pyblinker\test\blinker_pyblinker_comparison\test_f_fitblink_terminal_edge_case.py`

6. Exact change made
Added a validity gate in `FitBlinks.fit()` so rows with missing velocity extrema or base landmarks no longer continue into half-height, fit-range, and line-fit calculations. This keeps downstream fit fields as `NaN`, matching MATLAB's all-or-nothing behavior for failed terminal fits. Added a regression test covering a synthetic blink whose `right_zero` reaches the final sample.

7. Why the change was made
For subject `9636514`, MATLAB reports `1287` valid fitted rows after dropping `NaN` rows, while Python reported `1288`. The extra Python-valid row was the final candidate (`number = 1288`) whose `rightZero` landed on the last sample. MATLAB prints `Failed to fit blink 1288` and leaves downstream fit fields `NaN`; Python was previously recovering from the velocity indexing failure and computing `leftR2/rightR2` anyway.

8. MATLAB reference used
- `fitBlinks.m`
- `extractBlinks.m`
- `getGoodBlinkMask.m`
- Direct MATLAB batch run on subject `9636514` using the stored `signalData.signal` and `signalData.blinkPositions`

9. Validation scope
- Subjects:
  - `9636514` targeted first-divergence analysis
  - Revalidation planned for top/bottom staged groups under new experiment prefix after logic change
- How many subjects:
  - Targeted subject analysis first
  - Then top 2, bottom 2, top 5, bottom 5, top 10, bottom 10, and full ordered sweep
- Which tests:
  - `pytest test/blinker_pyblinker_comparison -q`

10. Before/after metrics
Before:
- Subject `9636514` had `share_within_tolerance_percent = 99.93861264579496`
- Ordered full sweep had `1` non-100 subject out of `74`

After:
- Focused comparison tests: `7 passed`
- `exp04_bottom2_summary.csv`: `2/2` at `100.0`
- `exp04_top2_summary.csv`: `2/2` at `100.0`
- `exp04_bottom5_summary.csv`: `5/5` at `100.0`
- `exp04_top5_summary.csv`: `5/5` at `100.0`
- `exp04_bottom10_summary.csv`: `10/10` at `100.0`
- `exp04_top10_summary.csv`: `10/10` at `100.0`
- `exp04_top15_summary.csv`: `15/15` at `100.0`
- `exp04_top20_summary.csv`: `20/20` at `100.0`
- `exp04_top74_summary.csv`: `74/74` at `100.0`
- `exp04_top74_overall.json`:
  - `total_detected_total = 72568`
  - `total_ground_truth_total = 72568`
  - `share_within_tolerance_total = 145136`
  - `detected_only_total = 0`
  - `ground_truth_only_total = 0`
  - strict/lenient micro `precision`, `recall`, `f1`, and `accuracy` all `1.0`

11. Whether the change was kept or reverted
Kept

12. Next recommended step
If the project needs the literal full set of `75` dataset folders, add a supported raw-input path for `12400256` (or a corresponding EDF/FIF export) because it is not listed in `summary_metrics.csv` and does not contain an `.edf` or `.fif` for the fresh runner.
