# 032 PyBlinker Src Refactor Smoke Validation

1. Title
   Remove the nested `pyblinker/src` duplicate tree and rerun 2-recording smoke validation on both datasets

2. Date/time
   2026-03-18 16:55:00 +08:00

3. Hypothesis
   If the nested `pyblinker` checkout stops carrying the stale validation-harness `src` copy, then the
   repository boundary becomes clearer and the canonical validation runners should still reproduce
   `100.0` share-within-tolerance on a small smoke scope for both supported datasets.

4. Files inspected
- `README.md`
- `.gitignore`
- `pyblinker/setup.py`
- `pyblinker/pyproject.toml`
- `pyblinker/tutorial/02b_extract_blink_from_matlab_blinker.py`
- `pyblinker/src/*`
- `src/validation/fresh_compare_from_csv.py`
- `src/validation/fresh_compare_subjects.py`
- `src/validation/run_murat_full_with_status.py`

5. Files changed
- `.gitignore`
- `pyblinker/tutorial/__init__.py`
- `pyblinker/tutorial/02b_extract_blink_from_matlab_blinker.py`
- `pyblinker/docs/01_blink_region_and_candidates.md`
- `pyblinker/docs/06_matlab_migration_and_replication.md`
- deleted nested tracked files under `pyblinker/src/`

6. Exact change made
- Marked `pyblinker/` and `ear_eog_experiment/` as nested-repo paths in the outer repository ignore list.
- Removed the tracked duplicate validation helper files from `pyblinker/src/`.
- Updated the MATLAB tutorial in the nested `pyblinker` repo to resolve
  `src/matlab_runner/execute_blinker.py` from the real validation workspace, either by parent-path
  discovery or `BLINKER_VALIDATION_ROOT`.
- Updated the nested `pyblinker` tutorial/docs text so they no longer describe the removed vendored
  `src` tree as package-owned code.

7. Why the change was made
- The nested `pyblinker/src` tree contained validation-harness leftovers rather than package runtime code.
- The nested repository boundary was noisy in the outer repo status because both nested repos appeared as
  untracked directories.
- The requested safety check was to confirm the cleanup did not change the validation outcome on both
  datasets.

8. MATLAB reference used
- Existing MATLAB-generated `blinker_results.pkl` comparison artifacts already present in:
  - `D:\dataset\murat_2018\<recording_id>\blinker_results.pkl`
  - `D:\dataset\drowsy_driving_raja_processed\<subject_id>\blinker_pyblinker_validation\blinker_results.pkl`

9. Validation scope
- Murat smoke scope:
  - top 2 ordered recordings from `src/validation/summary_metrics.csv`
  - recordings: `9636571`, `9636595`
- Driving smoke scope:
  - first 2 canonical subjects
  - subjects: `S1`, `S2`
- Additional checks:
  - `python tutorial\murat_sequence\step3_validate_pyblinker.py --help`
  - `python tutorial\raja_sequence\step4_validate_pyblinker.py --help`
  - `python -m py_compile pyblinker\tutorial\02b_extract_blink_from_matlab_blinker.py pyblinker\tutorial\__init__.py`

10. Before/after metrics
Before:
- structural cleanup target only; no new smoke summary yet for this refactor step

After:
- Murat smoke summary:
  - `9636571`: `100.0`
  - `9636595`: `100.0`
  - overall micro/macro metrics: all `1.0`
- Driving smoke summary:
  - `S1`: `100.0`
  - `S2`: `100.0`
  - overall micro/macro metrics: all `1.0`

11. Whether the change was kept or reverted
- Kept.

12. Output artifacts
- Murat summary:
  - `reports/validation/src_refactor_smoke_20260318_murat_top2_summary.csv`
  - `reports/validation/src_refactor_smoke_20260318_murat_top2_overall.json`
- Driving summary:
  - `reports/validation/src_refactor_smoke_20260318_driving_driving_dataset_2subjects_summary.csv`
  - `reports/validation/src_refactor_smoke_20260318_driving_driving_dataset_2subjects_overall.json`

13. Conclusion
   The nested `pyblinker/src` duplicate tree can be removed without changing the smoke-validation
   outcome for the checked Murat and driving-dataset scopes. On this 2-item smoke scope for each
   dataset, the repository remains behaviorally clean at `100.0%` share within tolerance.
