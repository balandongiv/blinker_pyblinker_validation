# 017 S10 Remaining MNE EDF Vs MATLAB Gap

1. Title
`S10` remains the sole driving-dataset blocker after the FIF parity fix; the remaining mismatch starts at the good-blink masking threshold because MNE EDF input still differs slightly from MATLAB’s EDF/Biosig path

2. Date/time
2026-03-14 15:14:33 +08:00

3. Hypothesis
If `S10` still fails after the FIF runner is aligned to the EDF channel grid and EDF tail padding, then the remaining mismatch is not a FIF-channel problem. The next likely source is the previously suspected MATLAB-vs-MNE EDF ingest difference on the same EDF recording.

4. Files inspected
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\fresh_compare_subjects.py`
- `c:\Users\balan\IdeaProjects\pyblinker\pyblinker\utils\statistics_utils.py`
- `c:\Users\balan\IdeaProjects\pyblinker\pyblinker\blinker\fit_blink.py`
- `D:\dataset\drowsy_driving_raja_processed\S10\S10.fif`
- `D:\dataset\drowsy_driving_raja_processed\S10\blinker_pyblinker_validation\S10.edf`
- `D:\dataset\drowsy_driving_raja_processed\S10\blinker_pyblinker_validation\blinker_results.pkl`

5. Files changed
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\finding\017_s10_remaining_mne_edf_vs_matlab_biosig_gap.md`

6. Exact change made
No shared code change was kept in this investigation step. I ran stage-by-stage probes on `S10` after the parity runner fix to identify the first remaining divergence.

7. Why the change was made
The new runner fix made `S11-S27` clean, but `S10` still stopped at `99.94453688297283`. The goal of this step was to determine whether `S10` is still blocked by the validation input path or by a detector-logic bug.

8. MATLAB reference used
- `D:\dataset\drowsy_driving_raja_processed\S10\blinker_pyblinker_validation\blinker_results.pkl`
  - stored MATLAB filtered signal for `eog_vert_right`
  - MATLAB final blink regions
- MATLAB-side raw path: `D:\dataset\drowsy_driving_raja_processed\S10\blinker_pyblinker_validation\S10.edf`

9. Validation scope
- Which subjects:
  - `S10`
- How many subjects:
  - `1`
- Which tests:
  - `C:\Users\balan\anaconda3\envs\pyblinker\python.exe -m blinker_pyblinker_validation.fresh_compare_subjects --dataset driving_dataset --prefix drvexp04 --subjects S10 --py-raw-template '{id}/{id}.fif' --comparison-raw-template '{id}/blinker_pyblinker_validation/{id}.edf' --restrict-py-to-comparison-channels`
  - ad hoc probes comparing the transformed FIF signal, the MNE EDF signal, and the MATLAB stored filtered signal

10. Before/after metrics
Fresh `S10` under `drvexp04`:
- `share_within_tolerance_percent = 99.94453688297283`
- `total_detected = 902`
- `total_ground_truth = 901`
- `detected_only = 1`
- `ground_truth_only = 0`
- extra final PyBlinker event: `357729-357791` (zero-based event bounds in the internal frame; comparison output is the same unmatched blink)

Collected evidence:
- After the new runner parity transform, the transformed FIF EOG channels match the MNE EDF channels exactly:
  - `eog_vert_right max_abs_uV = 0.0`
  - `eog_vert_left max_abs_uV = 0.0`
  - `eog_hor_right max_abs_uV = 0.0`
  - `eog_hor_left max_abs_uV = 0.0`
- Therefore the remaining blocker is not the FIF path. It is the older MATLAB-vs-MNE EDF ingest gap already suspected on `S10`.
- Stage-by-stage comparison on `eog_vert_right`:
  - `get_blink_position`: identical (`2217`)
  - `FitBlinks`: identical (`2217`)
  - `number_blinks`: identical (`2217`)
  - `number_good_blinks`: identical (`1189`)
  - first remaining divergence: `get_good_blink_mask`
    - Python/MNE signal kept `931` rows
    - MATLAB stored signal kept `929` rows
    - the two extra good-mask rows on the Python/MNE signal were:
      - `186124-186147`
      - `341993-342031`
  - final output:
    - Python/MNE signal `902`
    - MATLAB stored signal `901`
- The unmatched final blink `357729-357791` is a threshold-boundary case:
  - Python/MNE stats: `best_median = 200.62784105595682`, `best_robust_std = 70.3601090570245`
  - MATLAB-signal stats: `best_median = 200.52117919921875`, `best_robust_std = 70.2523367980957`
  - Python/MNE lower z=2 cutoff: `59.90762294190782`
  - MATLAB-signal lower z=2 cutoff: `60.016505603027344`
  - extra blink `max_value` on Python/MNE signal: `59.95988720702296`
  - that value passes the Python/MNE cutoff but fails the MATLAB cutoff
- A float32 probe made the filtered-signal gap much smaller, but it did not change the final `S10` event count:
  - filtered max diff against MATLAB stored signal dropped from about `0.0025868558 uV` to about `0.0000911721 uV`
  - final count still stayed `902`

11. Whether the change was kept or reverted
Retained as investigation only. No detector or comparison logic change was justified from this step because the evidence still points upstream to the EDF ingestion path rather than to `get_blink_position`, `FitBlinks`, `get_blink_statistic`, `get_good_blink_mask`, `BlinkProperties`, or pAVR logic.

12. Next recommended step
Treat `S10` as the one remaining EDF-ingestion parity blocker. The highest-value next step is to obtain MATLAB/Biosig raw EDF samples for `S10` in a clean EEGLAB environment and compare them directly against MNE’s EDF samples at the raw-vector level. If those raw vectors differ, the fix belongs in the EDF loading/parity layer used for validation, not in the downstream blink pipeline.
