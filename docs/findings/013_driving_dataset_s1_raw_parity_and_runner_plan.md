# 013 Driving Dataset S1 Raw Parity And Runner Plan

1. Title
Extend the murat_2018 fresh-validation workflow to the driving dataset and verify whether the first mismatch is detector logic or raw-input parity

2. Date/time
2026-03-14 12:10 Asia/Kuala_Lumpur

3. Hypothesis
The existing PyBlinker-vs-MATLAB comparison logic should carry over to the driving dataset with minimal change if we keep the comparison fair: same subject, same blink-event alignment logic, and the same raw signal that MATLAB used. If `S1` only fails when PyBlinker runs on `S1.fif` but succeeds when it runs on the MATLAB-side `S1.edf`, then the first divergence is likely the source signal rather than the detector pipeline.

4. Files inspected
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\fresh_compare_from_csv.py`
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\blink_compare.py`
- `c:\Users\balan\IdeaProjects\pyblinker\pyblinker\blinker\pyblinker.py`
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\finding\012_full_sweep_exp04_and_missing_12400256_input.md`
- `D:\dataset\drowsy_driving_raja_processed\S1\blinker_pyblinker_validation\blinker_results.pkl`
- `D:\dataset\drowsy_driving_raja_processed\S1\blinker_pyblinker_validation\S1.edf`
- `D:\dataset\drowsy_driving_raja_processed\S1\S1.fif`

5. Files changed
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\fresh_compare_subjects.py`
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\finding\013_driving_dataset_s1_raw_parity_and_runner_plan.md`

6. Exact change made
Added a new reusable subject-list validation runner for dataset presets and explicit subject orders. The new runner is designed to:
- reuse the existing event-table preparation and summary logic
- support configurable raw paths, MATLAB result paths, and output directories
- start with `S1` and continue in order only while subjects stay at the target similarity
- store fresh PyBlinker results under a new experiment prefix

7. Why the change was made
`fresh_compare_from_csv.py` is tightly shaped around the murat_2018 CSV layout. The driving dataset uses a different folder structure and exposes a critical fairness question: MATLAB ran on `S1.edf`, while the normal PyBlinker input is `S1.fif`. A reusable runner is needed so we can compare both datasets without cloning the whole pipeline into dataset-specific scripts.

8. MATLAB reference used
- `D:\dataset\drowsy_driving_raja_processed\S1\blinker_pyblinker_validation\blinker_results.pkl`
- The MATLAB-side raw used to generate it: `D:\dataset\drowsy_driving_raja_processed\S1\blinker_pyblinker_validation\S1.edf`

9. Validation scope
- Which subjects:
  - `S1` manual probe before the runner change
  - full requested driving-dataset list staged after the runner is added
- How many subjects:
  - `1` pre-run probe
  - incremental expansion after script creation
- Which tests:
  - focused `pytest` comparison suite after the script is added

10. Before/after metrics
Before script addition:
- `S1.fif` with default EEG-only picks selected `E8` and reached `85.5188141391106`
- `S1.fif` with `eeg + eog` picks selected `eog_vert_right` and reached `97.70992366412213`
- `S1.edf` with `eeg + eog` picks selected `eog_vert_right` and reached `100.0`

After script addition:
- Pending validation run

11. Whether the change was kept or reverted
Kept so far. No detector-logic change has been justified yet; the current evidence points first to raw-input parity and channel-pick configuration.

12. Next recommended step
Run the new subject-list runner on `S1` with the driving-dataset preset that uses the MATLAB-side EDF path and `eeg,eog` channel picks. If that stays at `100.0`, continue through the requested subject list in order and only investigate detector logic if a matched-raw run still fails.
