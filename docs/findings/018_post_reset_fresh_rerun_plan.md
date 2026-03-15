# 018 Post-reset Fresh Rerun Plan

1. Title
Fresh staged rerun after factory reset of the driving_dataset and murat_2018 validation artifacts

2. Date/time
2026-03-14 16:12:00 +08:00

3. Hypothesis
If the previous fixes are valid and the dataset reset did not introduce a new conversion mismatch, then a fresh staged rerun should reproduce the same behavior: driving_dataset should remain clean except for the known `S10` blocker, and murat_2018 should remain fully clean.

4. Files inspected
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\fresh_compare_subjects.py`
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\finding\017_s10_remaining_mne_edf_vs_matlab_biosig_gap.md`

5. Files changed
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\finding\018_post_reset_fresh_rerun_plan.md`

6. Exact change made
Created a new investigation log to track the fresh post-reset rerun, the new experiment prefixes, and any new logic change if one becomes necessary.

7. Why the change was made
The user reset both datasets and regenerated the MATLAB blinker outputs. This rerun needs its own reproducible record so the next steps are not mixed with the earlier artifact set.

8. MATLAB reference used
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\raja_sequence\step3_run_blinker.py`
- `C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\murat_sequence\step3_run_blinker.py`

9. Validation scope
- Which subjects:
  - driving_dataset staged rerun
  - murat_2018 staged rerun
- How many subjects:
  - pending execution
- Which tests:
  - pending execution

10. Before/after metrics
Before rerun:
- pending execution

After rerun:
- pending execution

11. Whether the change was kept or reverted
Kept.

12. Next recommended step
Run fresh staged validation for driving_dataset with a new experiment prefix, then rerun murat_2018 with a new experiment prefix so both datasets are validated against the reset artifacts.
