"""

This is similar to step4_compare_pyblinker_vs_blinker.py but compares Blinker vs the visualization-based annotations.

The ground truth is the CSV files created during in the step4_compare_pyblinker_vs_blinker.
For example 12400406.csv, 12400409.csv, 12400412.csv
The annotation description may contain different label such as 'B', 'BD', 'BG', 'MANUAL', etc.
We consider all these labels as blink events.

The pyblinker input is the filename 'blinker_results.pkl' located in the same folder as the CSV files.

Same as step4, we compute various statistics and generate a report.

Edit also for the murat_sequence/complete_workflow_all_steps.py to include this step.
But in the murat_sequence/complete_workflow_all_steps.py
Give two options,the first option is to process all the recordings in the dataset
The second option is to process only a few selected recordings since this experiment only consider the top 10 best result and bottom 10 poor result.

Below are the bottom 10 poor result recording ids from step4_compare_pyblinker_vs_blinker.py
12400382
12400388
12400394
12400349
12400343
12400397
12400376
12400346

Below are the top 10 best result recording ids from step4_compare_pyblinker_vs_blinker.py
9636571
9636595
12400406
12400412
9636607
12400409
9636496
9636577
12400256

Output only the performance for top 10, bottom 10, and the combination of these 20 recordings.
"""