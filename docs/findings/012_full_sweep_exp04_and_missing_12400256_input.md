# 012 Full Sweep Exp04 And Missing 12400256 Input

1. Title
`exp04` full sweep reached 100% on all ordered CSV subjects; extra dataset folder `12400256` is blocked by missing raw input

2. Date/time
2026-03-14 11:05 Asia/Kuala_Lumpur

3. Hypothesis
After the terminal-fit fix, the ordered validation sweep should remain stable at `100%` across the full `summary_metrics.csv` set. Any remaining gap to "all 75 subjects" would likely be a dataset/input issue rather than a detector mismatch.

4. Files inspected
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\summary_metrics.csv`
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\experiment_results\exp04_top15_summary.csv`
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\experiment_results\exp04_top20_summary.csv`
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\experiment_results\exp04_top74_summary.csv`
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\experiment_results\exp04_top74_overall.json`
- `D:\dataset\murat_2018\12400256`

5. Files changed
- None

6. Exact change made
No code change. Ran validation with experiment prefix `exp04` on the rechecked top groups and the full ordered sweep, then inspected dataset-vs-CSV membership and attempted a fresh run on `12400256`.

7. Why the change was made
The user requested bottom-up plus top-down revalidation after any logic change, followed by scaling to the full dataset. This log records the final sweep result and the exact reason the extra non-CSV subject was not processed by the runner.

8. MATLAB reference used
- `blinker_results.pkl` files in the subject folders for comparison targets

9. Validation scope
- Which subjects:
  - top 15 from `summary_metrics.csv`
  - top 20 from `summary_metrics.csv`
  - all 74 ordered rows from `summary_metrics.csv`
  - direct attempt on extra dataset folder `12400256`
- How many subjects:
  - `15`
  - `20`
  - `74`
  - `1` extra folder probe
- Which tests:
  - `python -m pytest test/blinker_pyblinker_comparison -q`

10. Before/after metrics
Before:
- `exp03_top74_summary.csv` had `1` non-100 subject (`9636514`)
- `exp03_top74_overall.json` had `detected_only_total = 1`

After:
- `exp04_top15_summary.csv`: `15/15` at `100.0`
- `exp04_top20_summary.csv`: `20/20` at `100.0`
- `exp04_top74_summary.csv`: `74/74` at `100.0`
- `exp04_top74_overall.json`:
  - `recording_count = 74`
  - `total_detected_total = 72568`
  - `total_ground_truth_total = 72568`
  - `share_within_tolerance_total = 145136`
  - `detected_only_total = 0`
  - `ground_truth_only_total = 0`
  - strict/lenient micro `precision`, `recall`, `f1`, and `accuracy` all `1.0`
- Extra folder probe:
  - dataset subject folders found: `75`
  - ordered CSV recording IDs found: `74`
  - folder missing from CSV: `12400256`
  - `12400256` fresh run attempt failed because no `.edf` or `.fif` exists in that folder

11. Whether the change was kept or reverted
Retained. No further detector change was needed after the `exp04` sweep.

12. Next recommended step
If the project needs to include `12400256` in the fresh-run workflow, define a supported raw-input route for that folder first (for example an EDF/FIF export or an approved MAT-loader path) and then decide whether to add it to `summary_metrics.csv` for ordered validation.
