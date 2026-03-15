# 027 Tolerance Sweep Lower Bound

1. Title
   Determine the lowest `tolerance_samples` value that still preserves perfect
   full-sweep results on both `murat_2018` and `driving_dataset`

2. Date/time
   2026-03-15 14:35:00 +08:00

3. Hypothesis
   The current baseline at `tolerance_samples = 20` is known to be perfect for
   both datasets. A systematic downward sweep should identify either:
   - a lower tolerance that still preserves all strict and lenient macro/micro
     metrics at `1.0`, or
   - the first lower value where the current design becomes too sensitive.

4. Files inspected
- `src/validation/run_full_repro_pipeline.py`
- `src/validation/run_murat_full_with_status.py`
- `src/validation/fresh_compare_subjects.py`
- `README.md`
- `good_practice.md`

5. Files changed
- `docs/findings/027_tolerance_sweep_lower_bound.md`
- `src/validation/run_tolerance_sweep.py`
- `run_tolerance_sweep.py`
- `docs/findings/028_tolerance_sweep_cache_optimization.md`
- `docs/findings/029_tolerance_sweep_exact_match_short_circuit.md`
- `docs/findings/030_tolerance_sweep_lazy_signal_loading.md`

6. Exact change made
- Added a dedicated tolerance sweep runner, then optimized it in three steps:
  - cache prepared comparison inputs once per recording
  - short-circuit exact start/end event-table matches
  - load raw comparison signals lazily only for non-exact cases
- Rebuilt both datasets from the factory-reset state and regenerated fresh
  prefixed PyBlinker outputs:
  - `tol20_baseline_v1_murat`
  - `tol20_baseline_v1_driving`
- Computed the exact per-recording boundary deltas after `prepare_event_tables`
  and confirmed they are all zero for both datasets.
- Wrote the final tolerance reduction artifacts:
  - `reports/validation/tolerance_reduction_exact_match_results.csv`
  - `reports/validation/tolerance_reduction_exact_match_results.json`
  - `reports/validation/tolerance_reduction_exact_match_results.md`
  - `reports/validation/tolerance_reduction_exact_match_summary.json`

7. Why the change was made
- We need a reproducible answer for the minimum reliable `tolerance_samples`
  rather than relying only on the current baseline of `20`.

8. MATLAB reference used
- EEGLAB root: `D:\code development\matlab_plugin\eeglab2025.1.0`
- Blinker plugin: `D:\code development\matlab_plugin\eeglab2025.1.0\plugins\Blinker1.2.0`

9. Validation scope
- Factory-reset rebuild:
  - `tutorial/murat_sequence/step1_prepare_dataset.py`
  - `tutorial/murat_sequence/step2_run_blinker.py`
  - `tutorial/raja_sequence/step3_run_blinker.py`
- Fresh PyBlinker baselines:
  - full `murat_2018` ordered sweep: `74` recordings
  - full `driving_dataset` sweep: `22` subjects
- Focused checks:
  - exact boundary-difference analysis over all `96` recordings
  - tolerance pass/fail table for `20..0`
- Tests:
  - `py_compile` for `src/validation/run_tolerance_sweep.py` and `run_tolerance_sweep.py`

10. Before/after metrics
- Before:
  - `tolerance_samples = 20` is fully clean on both datasets
  - no practical dedicated full-dataset tolerance sweep runner exists
- After:
  - Fresh baseline at `tolerance_samples = 20` is again perfect on both datasets
  - Exact prepared-event boundary difference:
    - Murat max required tolerance = `0`
    - driving max required tolerance = `0`
  - Full pass table for all valid non-negative tolerances `20..0`:
    - Murat: pass at every tested value
    - driving_dataset: pass at every tested value
  - Lowest stable `tolerance_samples` = `0`
  - First lower valid failing value = none
  - Lower than `0` is invalid by design (`tolerance_samples` must be non-negative)

11. Whether the change was kept or reverted
- Kept

12. Next recommended step
- Use `tolerance_samples = 0` going forward when exact boundary identity is the
  intended criterion.
- If a future `pyblinker` logic change ever introduces non-zero boundary deltas,
  rerun the same exact-match analysis first, then confirm the new boundary with
  full comparisons at the candidate passing value and the next lower value.
