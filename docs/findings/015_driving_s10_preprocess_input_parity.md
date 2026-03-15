# 015 Driving S10 Preprocess Input Parity

1. Title
`S10` stops the driving-dataset rollout at `99.94453688297283`; the remaining mismatch is caused by EDF input-path parity before `process_channel_data(...)`

2. Date/time
2026-03-14 16:20 Asia/Kuala_Lumpur

3. Hypothesis
After the empty-frame `FitBlinks` fix, any remaining driving-dataset mismatch should be traced to the first stage where MATLAB and Python actually diverge. If Python reproduces MATLAB exactly when given MATLAB’s own stored filtered signal, then the mismatch is upstream preprocessing or EDF input parity, not blink fitting/statistics/pAVR logic.

4. Files inspected
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\fresh_compare_subjects.py`
- `c:\Users\balan\IdeaProjects\pyblinker\pyblinker\blinker\legacy_eeglab_filter.py`
- `c:\Users\balan\IdeaProjects\pyblinker\pyblinker\blinker\pyblinker.py`
- `c:\Users\balan\IdeaProjects\pyblinker\pyblinker\pipeline_steps.py`
- `D:\dataset\drowsy_driving_raja_processed\S10\blinker_pyblinker_validation\blinker_results.pkl`
- `D:\dataset\drowsy_driving_raja_processed\S10\blinker_pyblinker_validation\S10.edf`

5. Files changed
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\finding\014_probe_s10_matlab_raw_diff.py`

6. Exact change made
No shared detector/comparison logic change was kept in this investigation. Added a probe helper script to compare Python’s filtered EDF signal against MATLAB’s stored filtered signal and to attempt a MATLAB-side raw probe through the local EEGLAB/Biosig stack.

7. Why the change was made
The incremental rollout stayed clean through `S1` to `S7`, then stopped at `S10` with one extra detected blink. The goal of this investigation was to identify the first real divergence rather than guessing at downstream thresholds.

8. MATLAB reference used
- `D:\dataset\drowsy_driving_raja_processed\S10\blinker_pyblinker_validation\blinker_results.pkl`
  - stored `signalData` for `eog_vert_right`
  - final MATLAB `blinkFits`
- EEGLAB FIR reference:
  - `D:\code development\matlab_plugin\eeglab2025.1.0\plugins\firfilt\fir_filterdcpadded.m`
  - `D:\code development\matlab_plugin\eeglab2025.1.0\plugins\firfilt\firws.m`

9. Validation scope
- Which subjects:
  - driving dataset `S1` through `S10` in ordered rollout
  - focused deep dive on `S10`
- How many subjects:
  - `8` processed in the clean/first-failure sweep (`S1, S2, S3, S4, S5, S6, S7, S10`)
  - `1` subject deep dive (`S10`)
- Which tests:
  - `python -m blinker_pyblinker_validation.fresh_compare_subjects --dataset driving_dataset --prefix drvexp02`
  - `python blinker_pyblinker_validation/finding/014_probe_s10_matlab_raw_diff.py`
  - ad hoc step-by-step probes on `S10`

10. Before/after metrics
Driving rollout summary under `drvexp02`:
- `S1 = 100.0`
- `S2 = 100.0`
- `S3 = 100.0`
- `S4 = 100.0`
- `S5 = 100.0`
- `S6 = 100.0`
- `S7 = 100.0`
- `S10 = 99.94453688297283`
  - `total_detected = 902`
  - `total_ground_truth = 901`
  - `detected_only = 1`
  - `ground_truth_only = 0`
  - extra PyBlinker event: `357730-357792` on `eog_vert_right`

Stage-by-stage `S10` evidence:
- On the actual Python EDF path:
  - selected channel matched MATLAB: `eog_vert_right`
  - `number_good_blinks` matched MATLAB summary: `1189`
  - final output still had one extra event
- On MATLAB’s own stored filtered `eog_vert_right` signal from `blinker_results.pkl`:
  - Python reproduced MATLAB exactly:
    - `number_blinks = 2217`
    - `number_good_blinks = 1189`
    - `best_median = 200.52117919921875`
    - `best_robust_std = 70.2523367980957`
    - final `blinkFits = 901`
    - the extra `357730-357792` event disappeared
- FIR implementation check:
  - Python `legacy_blinker_bandpass(...)` and MATLAB `fir_filterdcpadded(...)` matched on the same raw vector up to floating-point noise:
    - `max_abs ~= 4.53e-17`
    - `mean_abs ~= 7.10e-18`
- Remaining filtered-signal difference between Python EDF path and MATLAB stored signal:
  - `filtered_max_abs_microvolt = 0.002586855824590506`
  - `filtered_mean_abs_microvolt = 0.000573308284058922`
  - `filtered_rmse_microvolt = 0.0007082066672924618`

11. Whether the change was kept or reverted
Retained as investigation only. No detector logic change was applied from this step because the evidence does not support a downstream blink-logic bug.

12. Next recommended step
Treat the current blocker as an EDF input-path parity issue. The highest-value next step is to obtain MATLAB’s raw EDF samples for `eog_vert_right` on `S10` through a clean EEGLAB/Biosig environment and compare them directly against MNE’s EDF samples. If those raw vectors differ, the fix belongs in the EDF ingestion path used for validation, not in `get_blink_position`, `FitBlinks`, `get_blink_statistic`, `BlinkProperties`, or pAVR.
