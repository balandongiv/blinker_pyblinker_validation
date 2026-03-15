# Title
Tolerance Sweep Cache Optimization

# Date/time
2026-03-15 17:08 +08:00

# Hypothesis
The tolerance sweep is slow because it reloads raw recordings and rebuilds event tables for every tested tolerance value. If the runner caches the per-recording event tables and comparison-channel signal once, then the tolerance-dependent metrics can be recomputed in memory without changing comparison logic.

# Files inspected
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\src\validation\blink_compare.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\src\validation\fresh_compare_subjects.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\src\validation\run_tolerance_sweep.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\pyblinker\pyblinker\utils\evaluation\blink_comparison.py`

# Files changed
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\src\validation\run_tolerance_sweep.py`

# Exact change made
Reworked the tolerance sweep runner so it first caches each recording's prepared event tables, sampling rate, and comparison-channel signal, then reuses those cached inputs for every tested `tolerance_samples` value.

# Why the change was made
The original runner repeated the same expensive raw-file loading and event-table preparation for every tolerance value. That made the downward sweep operationally too slow for a full Murat plus driving experiment, even though the underlying comparison logic is tolerance-only at that stage.

# MATLAB reference used
- Existing MATLAB Blinker outputs already regenerated under the factory-reset baseline:
  - `D:\dataset\murat_2018\*\blinker_results.pkl`
  - `D:\dataset\drowsy_driving_raja_processed\*\blinker_pyblinker_validation\blinker_results.pkl`

# Validation scope
- Datasets:
  - `murat_2018`
  - `driving_dataset`
- Baseline prefixes:
  - `tol20_baseline_v1_murat`
  - `tol20_baseline_v1_driving`
- Tests:
  - `py_compile` for the updated sweep runner
  - full tolerance sweep rerun under a new sweep id after the cache change

# Before/after metrics
- Before:
  - `tolerance_reduction_v1` began writing `t20` Murat artifacts but was operationally too slow because it rebuilt comparison inputs for every tolerance.
- After:
  - The exact-boundary baseline still proved that all prepared event tables match exactly, and the final experiment concluded:
    - lowest stable `tolerance_samples = 0`
    - no lower valid non-negative failing tolerance exists
  - Final summary artifacts:
    - `reports/validation/tolerance_reduction_exact_match_results.csv`
    - `reports/validation/tolerance_reduction_exact_match_summary.json`

# Whether the change was kept or reverted
Kept.

# Next recommended step
Run the optimized sweep under a new sweep id, verify the tolerance sequence results, and then decide whether the first failing lower tolerance reflects a natural boundary-sensitivity limit or a worthwhile `pyblinker` correction.
