# 016 Driving Post S10 Subject Validation

1. Title
Validate the remaining listed driving-dataset subjects after `S10` first, then revisit `S10`

2. Date/time
2026-03-14 16:40 Asia/Kuala_Lumpur

3. Hypothesis
The `S10` mismatch may be isolated rather than a general failure across later driving-dataset subjects. If the remaining requested subjects (`S11` onward, excluding `S10`) all stay at `100.0`, then `S10` should remain the prioritized blocker and not be treated as evidence of a broader detector regression.

4. Files inspected
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\fresh_compare_subjects.py`
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\experiment_results\drvexp02_driving_dataset_8subjects_summary.csv`

5. Files changed
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\finding\016_driving_post_s10_subject_validation.md`

6. Exact change made
Started a new validation log for the requested subject order:
- `S11, S12, S13, S16, S17, S18, S19, S20, S21, S22, S23, S24, S26, S27`
- revisit `S10` afterward

7. Why the change was made
The user requested continuing the driving-dataset validation on the later subjects first, and only returning to `S10` if that block stays clean.

8. MATLAB reference used
- MATLAB `blinker_results.pkl` files in each driving-dataset subject folder

9. Validation scope
- Which subjects:
  - pending run on `S11, S12, S13, S16, S17, S18, S19, S20, S21, S22, S23, S24, S26, S27`
  - then `S10`
- How many subjects:
  - `14` later subjects first
  - `1` revisit subject
- Which tests:
  - validation runner only unless logic changes

10. Before/after metrics
Before:
- `drvexp02` clean through `S7`
- `S10 = 99.94453688297283`

After:
- Pending

11. Whether the change was kept or reverted
Pending

12. Next recommended step
Run the later subject block under the current driving-dataset experiment prefix, inspect whether any subject besides `S10` fails, and then rerun `S10` explicitly.
