# 021 Murat Exp06 Full Ordered Sweep

1. Title
Extend the fresh post-reset `exp06` rerun to the full ordered `murat_2018` validation list

2. Date/time
2026-03-15 09:05:00 +08:00

3. Hypothesis
If the fresh post-reset top-10 `murat_2018` smoke scope is representative and no shared logic changed after that run, then the full ordered `summary_metrics.csv` sweep under the same `exp06` prefix should remain clean at `100.0`.

4. Files inspected
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\summary_metrics.csv`
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\fresh_compare_from_csv.py`
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\fresh_compare_subjects.py`

5. Files changed
- `c:\Users\balan\IdeaProjects\pyblinker\blinker_pyblinker_validation\finding\021_murat_exp06_full_ordered_sweep.md`

6. Exact change made
Created a new investigation log for the full fresh `exp06` murat sweep before running the ordered validation.

7. Why the change was made
The user requested the full murat rerun so that all ordered recording IDs receive fresh `exp06_pyblinker_results.pkl` outputs rather than only the earlier top-10 smoke scope.

8. MATLAB reference used
- `D:\dataset\murat_2018\*\blinker_results.pkl`

9. Validation scope
- Which subjects:
  - all ordered recording IDs from `summary_metrics.csv`
- How many subjects:
  - pending execution
- Which tests:
  - pending execution

10. Before/after metrics
Before:
- `exp06` exists only for the earlier top-10 smoke scope

After:
- pending execution

11. Whether the change was kept or reverted
Kept.

12. Next recommended step
Run the full ordered `exp06` sweep, inspect the aggregate metrics, and confirm which subject folders now contain `exp06_pyblinker_results.pkl`.
