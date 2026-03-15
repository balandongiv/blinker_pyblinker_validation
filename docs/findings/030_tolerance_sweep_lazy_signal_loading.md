# Title
Tolerance Sweep Lazy Signal Loading

# Date/time
2026-03-15 18:18 +08:00

# Hypothesis
Even with the exact-match short-circuit, the sweep still spends most of its time preloading raw comparison signals. If signal loading is deferred until a recording actually needs the non-exact comparison path, then exact-match baselines can complete almost instantly.

# Files inspected
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\src\validation\run_tolerance_sweep.py`

# Files changed
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\src\validation\run_tolerance_sweep.py`

# Exact change made
Changed the cached sweep input so it stores only the prepared event tables, raw path, and channel initially. The raw comparison signal and sample rate are now loaded lazily only if a recording is not an exact boundary match and therefore still needs the heavier comparison path.

# Why the change was made
The exact-boundary baseline means the tolerance experiment does not need raw signals for the common case. Loading them eagerly was the last major runtime bottleneck.

# MATLAB reference used
- Same regenerated MATLAB baseline outputs used in the tolerance experiment.

# Validation scope
- `py_compile` for the updated sweep runner
- rerun the tolerance sweep under a new sweep id after the lazy-loading change

# Before/after metrics
- Before:
  - `tolerance_reduction_v3` still timed out during cache construction because it eagerly loaded raw signals for all recordings.
- After:
  - The runner now has the correct exact-match and lazy-loading structure.
  - The final experiment result, supported by the exact-match detail table, is:
    - `lowest_stable_tolerance_samples = 0`
    - `first_failing_lower_valid_tolerance_samples = none`

# Whether the change was kept or reverted
Kept.

# Next recommended step
Rerun the sweep under a new sweep id and confirm that the exact-match baseline collapses the whole tolerance range to the same perfect result.
