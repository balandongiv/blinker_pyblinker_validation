# Title
Refactor-default-setting workspace cleanup and explicit comparison-parameter rerun

# Date/time
2026-03-17 10:40 Asia/Kuala_Lumpur

# Hypothesis
The `refactor_default_blinker_setting_entry` workspace picked up two structural mistakes during refactoring:
1. validation-only tutorial wrappers were copied into the editable `pyblinker` clone under `tutorial/murat_sequence` and `tutorial/raja_sequence`
2. at least one PyBlinker test still pointed back into the validation harness for `prepare_event_tables`

If those are corrected, and the validation runners explicitly inject the legacy default blink parameters instead of relying on implicit defaults, the architecture becomes cleaner without changing behavior. Murat and driving top-2 comparisons should remain at 100%.

# Files inspected
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\pyblinker\tutorial\murat_sequence\step3_validate_pyblinker.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\pyblinker\tutorial\raja_sequence\step4_validate_pyblinker.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\tutorial\murat_sequence\step3_validate_pyblinker.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\tutorial\raja_sequence\step4_validate_pyblinker.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\pyblinker\test\blinker_pyblinker_comparison\test_e_prepare_event_tables.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\src\validation\fresh_compare_from_csv.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\src\validation\fresh_compare_subjects.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\pyblinker\tutorial\01a_basic_usage.py`

# Files changed
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\src\validation\blinker_params.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\src\validation\fresh_compare_from_csv.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\src\validation\fresh_compare_subjects.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\pyblinker\pyblinker\utils\evaluation\event_tables.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\pyblinker\pyblinker\utils\evaluation\__init__.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\pyblinker\test\blinker_pyblinker_comparison\test_e_prepare_event_tables.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\pyblinker\tutorial\01a_basic_usage.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\pyblinker\CHANGELOG.md`
- removed misplaced clone-local wrappers:
  - `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\pyblinker\tutorial\murat_sequence\*`
  - `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\pyblinker\tutorial\raja_sequence\*`

# Exact change made
1. Added `src/validation/blinker_params.py` with an explicit legacy-default `EXPLICIT_BLINKER_PARAMS` mapping and a copy-safe builder.
2. Updated the Murat and driving validation runners to pass `blink_params=build_experiment_blink_params()` into `BlinkDetector(...)`.
3. Stored the explicit blink-parameter profile into the generated PyBlinker payload under `payload["params"]["blink_params"]` for experiment traceability.
4. Added a small PyBlinker-owned helper `pyblinker.utils.evaluation.event_tables.prepare_event_tables(...)`.
5. Replaced the commented reverse-dependency test import with a real self-contained test that imports from PyBlinker instead of the validation harness.
6. Corrected `tutorial/01a_basic_usage.py` in the editable PyBlinker clone so its explicit example uses `min_good_blinks = 10`, matching the experiment profile.
7. Removed the stray `tutorial/murat_sequence` and `tutorial/raja_sequence` directories from the editable PyBlinker clone because the canonical wrappers already exist at the validation-repo top level.
8. Added an `Unreleased` changelog note in the editable PyBlinker clone.

# Why the change was made
- To restore the intended dependency direction: validation depends on PyBlinker, not the other way around.
- To make comparison experiments reproducible and explicit about which blink settings were used.
- To remove confusing duplicate tutorial entry points from the editable PyBlinker clone.
- To keep the workspace academically cleaner without changing detector behavior.

# MATLAB reference used
No new MATLAB logic review was needed in this pass. The explicit parameter profile matches the previously validated legacy-default comparison configuration already used for Murat and driving dataset parity.

# Validation scope
- Subjects:
  - `murat_2018`: top 2 ordered recordings from `summary_metrics.csv` (`9636595`, `9636571`)
  - `driving_dataset`: `S1`, `S2`
- How many subjects:
  - Murat: 2
  - driving_dataset: 2
- Tests:
  - `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\pyblinker\test\run_all_tests.py`
  - `C:\Users\balan\IdeaProjects\pyblinker\test\run_all_tests.py`

# Before/after metrics
- Before:
  - Workspace had duplicated tutorial validation wrappers inside the editable PyBlinker clone.
  - `test_e_prepare_event_tables.py` was commented out and referenced the validation harness path.
  - Comparison runners did not explicitly inject the experiment blink-parameter profile.
- After:
  - Editable-clone tests: `42 passed`
  - Current PyBlinker checkout tests: `42 passed`
  - Murat top 2 overall metrics in `reports/validation/refsetfix_murat02_top2_overall.json`: all strict/lenient macro/micro precision, recall, f1, and accuracy = `1.0`
  - driving top 2 overall metrics in `reports/validation/refsetfix_drv02_driving_dataset_2subjects_overall.json`: all strict/lenient macro/micro precision, recall, f1, and accuracy = `1.0`

# Whether the change was kept or reverted
Kept.

# Next recommended step
If this branch is going to remain the editable validation checkout, the next sensible step is a broader Murat/driving sweep with a fresh experiment prefix to confirm the explicit-parameter profile stays clean beyond the 2-subject smoke scope.
