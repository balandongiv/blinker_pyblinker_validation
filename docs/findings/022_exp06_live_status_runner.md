# 022 Exp06 Live Status Runner

1. Title
Add a dedicated full-sweep runner with live status files for the post-reset `murat_2018` `exp06` rerun

2. Date/time
2026-03-15 09:20:00 +08:00

3. Hypothesis
If the full ordered `exp06` murat sweep is run from a dedicated wrapper that rewrites status files on each heartbeat and each completed subject, then the user can monitor progress live even if the terminal session is interrupted.

4. Files inspected
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\fresh_compare_from_csv.py`
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\summary_metrics.csv`

5. Files changed
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\run_murat_exp06_full_with_status.py`
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\finding\022_exp06_live_status_runner.md`

6. Exact change made
Added a dedicated runner that force-reruns the full ordered `murat_2018` `exp06` sweep, writes a rolling text log, and rewrites live status JSON/Markdown files while the run is in progress.

7. Why the change was made
The user needs a way to confirm that the full rerun is still active after repeated power interruptions, rather than guessing from a silent terminal.

8. MATLAB reference used
- `D:\dataset\murat_2018\*\blinker_results.pkl`

9. Validation scope
- Which subjects:
  - full ordered `murat_2018` CSV list
- How many subjects:
  - `74`
- Which tests:
  - pending execution

10. Before/after metrics
Before:
- no live status files existed for the full `exp06` murat rerun

After:
- background runner started successfully
- live files created:
  - `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\experiment_results\exp06_top74_live_status.json`
  - `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\experiment_results\exp06_top74_live_status.md`
  - `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\experiment_results\exp06_top74_live_log.txt`
- initial verification after launch:
  - process `python` running in background
  - `completed_count = 5`
  - `failed_count = 0`
  - completed recordings observed in the live status:
    - `12400409`
    - `12400412`
    - `9636595`
    - `12400406`
    - `9636607`

11. Whether the change was kept or reverted
Kept.

12. Next recommended step
Let the background process finish, then inspect the final `exp06_top74` summary artifacts and confirm that the requested subject folders now contain `exp06_pyblinker_results.pkl`.
