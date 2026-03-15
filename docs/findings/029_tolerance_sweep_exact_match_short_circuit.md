# Title
Tolerance Sweep Exact-Match Short-Circuit

# Date/time
2026-03-15 18:10 +08:00

# Hypothesis
If the prepared PyBlinker and MATLAB event tables are already exact start/end matches for a recording, then every non-negative `tolerance_samples` value produces the same perfect comparison result. In that case the sweep runner can safely emit perfect metrics directly without recomputing alignment.

# Files inspected
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\src\validation\run_tolerance_sweep.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\src\validation\stat.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\pyblinker\pyblinker\utils\evaluation\reporting.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\pyblinker\pyblinker\utils\evaluation\similarity.py`

# Files changed
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\src\validation\run_tolerance_sweep.py`

# Exact change made
Added an exact-match detection path to the tolerance sweep runner. When a cached recording has identical prepared `start_blink` and `end_blink` tables on the PyBlinker and MATLAB sides, the runner now synthesizes the corresponding perfect metrics directly for any non-negative tolerance instead of calling the heavier alignment routine.

# Why the change was made
The fresh factory-reset baseline showed that both datasets already have exact boundary agreement after `prepare_event_tables`. In that scenario repeated alignment work is unnecessary and operationally too slow for a full downward tolerance experiment.

# MATLAB reference used
- Factory-reset MATLAB baseline outputs used for the exact-match check:
  - `D:\dataset\murat_2018\*\blinker_results.pkl`
  - `D:\dataset\drowsy_driving_raja_processed\*\blinker_pyblinker_validation\blinker_results.pkl`

# Validation scope
- Exact-boundary analysis across:
  - all `74` Murat recordings from `summary_metrics.csv`
  - all `22` driving subjects
- Follow-up:
  - `py_compile` for the updated sweep runner
  - rerun the tolerance sweep under a new sweep id

# Before/after metrics
- Before:
  - `tolerance_reduction_v2` confirmed `t=20` passes and the baseline event tables were exact, but the repeated full sweep remained operationally slow.
- After:
  - Exact-boundary analysis across all `74 + 22` recordings showed:
    - Murat max required tolerance = `0`
    - driving max required tolerance = `0`
  - Therefore every valid non-negative tolerance from `20..0` remains perfect.

# Whether the change was kept or reverted
Kept.

# Next recommended step
Rerun the sweep under a new sweep id, confirm that all tolerances `20..0` pass instantly, and record that the minimum valid stable tolerance is `0` with no lower non-negative failing value.
