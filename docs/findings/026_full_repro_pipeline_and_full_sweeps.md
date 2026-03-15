# 026 Full Repro Pipeline And Full Sweeps

1. Title
   Add a one-click end-to-end reproducibility pipeline, document it, and run the
   full Murat and driving-dataset validation sweeps from the restructured repository

2. Date/time
   2026-03-15 13:05:00 +08:00

3. Hypothesis
   The restructured repository already contains the necessary preparation and
   validation building blocks. A single orchestrator script should be able to run
   the complete MATLAB-backed preparation and PyBlinker validation flow for both
   datasets, emit durable logs and a manifest, and reproduce the full published
   comparison results.

4. Files inspected
- `README.md`
- `good_practice.md`
- `src/validation/run_murat_full_with_status.py`
- `src/validation/fresh_compare_subjects.py`
- `src/validation/fresh_compare_from_csv.py`
- `tutorial/murat_sequence/step1_prepare_dataset.py`
- `tutorial/murat_sequence/step2_run_blinker.py`
- `tutorial/murat_sequence/step3_validate_pyblinker.py`
- `tutorial/raja_sequence/step3_run_blinker.py`
- `tutorial/raja_sequence/step4_validate_pyblinker.py`

5. Files changed
- `docs/findings/026_full_repro_pipeline_and_full_sweeps.md`
- `src/validation/run_full_repro_pipeline.py`
- `run_full_repro_pipeline.py`
- `README.md`
- `mock_data/dataset/drowsy_driving_raja_processed/S13/S26_20190108_035218_3/ear_eog.fif`
- `mock_data/CVAT_visual_annotation/cvat_zip_final/S13/from_cvat/default-annotations-human-imagelabels.csv`
- `mock_data/CVAT_visual_annotation/cvat_zip_final/S13/from_cvat/S26_20190108_035218_3.zip`

6. Exact change made
- Created this investigation log before implementing the one-click pipeline and
  before running the full validation sweeps.
- Added `src/validation/run_full_repro_pipeline.py` to orchestrate the full
  end-to-end run for both datasets, stream child-command output into a durable
  pipeline log, and write a combined JSON/Markdown manifest with final metrics.
- Added the top-level clickable wrapper `run_full_repro_pipeline.py`.
- Rewrote `README.md` around the one-click reproducibility flow and the new
  final artifact locations.
- Restored the tiny Raja mock-data fixture expected by the repository-local UI
  tests after the restructure.
- Patched the new pipeline runner once to handle Windows console encoding
  safely while streaming long MATLAB/PyBlinker output.
- Ran the full end-to-end pipeline with:
  `python run_full_repro_pipeline.py --run-id full_repro_v1 --force-validation`

7. Why the change was made
- The repository needs a durable, public-facing, end-to-end entry point that a
  future researcher can run from a clean state to reproduce the full validation
  outputs.

8. MATLAB reference used
- EEGLAB root: `D:\code development\matlab_plugin\eeglab2025.1.0`
- Blinker plugin: `D:\code development\matlab_plugin\eeglab2025.1.0\plugins\Blinker1.2.0`

9. Validation scope
- Implement one-click full pipeline
- Update README for public reproducibility
- Restore the bundled Raja mock-data fixture required by repository-local tests
- Run `pytest tests -q`
- Run full `murat_2018` sweep
- Run full `driving_dataset` sweep

10. Before/after metrics
- Before:
  - smoke scopes after restructure were clean
  - no single public-facing one-click full-pipeline script existed
- After:
  - Murat full sweep:
    - `74/74` recordings at `100.0`
    - `total_detected_total = 72568`
    - `total_ground_truth_total = 72568`
    - strict/lenient macro and micro precision, recall, f1, and accuracy all `1.0`
    - summary: `reports/validation/full_repro_v1_murat_top74_summary.csv`
    - overall: `reports/validation/full_repro_v1_murat_top74_overall.json`
  - driving_dataset full sweep:
    - `22/22` recordings at `100.0`
    - `total_detected_total = 42501`
    - `total_ground_truth_total = 42501`
    - strict/lenient macro and micro precision, recall, f1, and accuracy all `1.0`
    - summary: `reports/validation/full_repro_v1_driving_driving_dataset_22subjects_summary.csv`
    - overall: `reports/validation/full_repro_v1_driving_driving_dataset_22subjects_overall.json`
  - combined reproducibility manifest:
    - `reports/validation/full_repro_v1_pipeline_manifest.json`
    - `reports/validation/full_repro_v1_pipeline_manifest.md`
  - repository-local tests: `14 passed`

11. Whether the change was kept or reverted
- Kept

12. Next recommended step
- Use `run_full_repro_pipeline.py` as the canonical public reproduction entry
  point.
- If shared `pyblinker` logic changes later, rerun the same script with a new
  `--run-id` and record the result in a new finding log.
