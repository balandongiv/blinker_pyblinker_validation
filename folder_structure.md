# 📂 Folder Structure

```plaintext
# The complete path is as describe in  config/config.yaml
dataset/
├── drowsy_driving_raja/                      # RAW INPUTS (unaltered)
│   ├── S1/
│   │   └── MD.mff/                 # This is a folder
│   │       ├── info.xml
│   │       ├── eeg1.mff
│   │       ├── ...
│   │       ├── S01_20170519_043933.mov
│   │       ├── S01_20170519_043933_2.mov
│   │       ├── S01_20170519_043933_3.mov
│   ├── S2/           
│   ├── ear_pkl_data/                        # RAW INPUTS for pkl, this is the output from calculated EAR
│   │   └── S1/                 # This is a folder
│   │       ├── S01_20170519_043933.pkl
│   │       ├── S01_20170519_043933_2.pkl
│   │       ├── S01_20170519_043933_3.pkl
│   ├── human_label_annotation/                        # label from 2 human annotator
│   │   └── S1/                 # This is a folder
│   │       └──  S01_20170519_043933/
│   │       │   └── ear_eog.csv
│   │       └──  S01_20170519_043933_2/
│   │       │   └── ear_eog.csv
│   │       └──  S01_20170519_043933_3/
│   │       │   └── ear_eog.csv


├── drowsy_driving_raja_processed/                      # CLEANED + SYNC OUTPUTS
│   ├── S1/
│   │   S1.fif
│   │   ├── S01_20170519_043933/
│   │   │   ├── pyblinker_blinker_validation/                       # All data from pyblinker valiidaton will come here
    │   │       ├── pyblinker_output.pkl   # Output from pyblinker
│   │   │   ├── seg_data_raw/                       # All aligned data, croped to same length, and for viz will be stored in this folder. Will be used for feature extraction and modeling.
    │   │       ├── ear_raw.fif   # Output from EAR; Previously known as seg_EAR_annotated_raw.fif
        │   │   ├── eeg_eog_raw.fif   # Previously known as raw_seg_annotated.fif or seg_annotated_raw.fif
│   │   │   ├── seg_data/                       #  From combine_ear_eeg_eog.py. All aligned data, croped to same length, and for viz will be stored in this folder. Will be used for feature extraction and modeling.
    │   │       ├── ear_eog_eeg_raw.fif   # This combine all modalities, but we downsample the EEG/EOG to 30 Hz. We usually use this for vizual confirmation only, not for feature extraction or modeling.
    │   │       ├── ear_raw.fif      
    │   │       ├── eeg_eog_raw.fif
    │   │       ├── eog_eeg_clean_raw.fif # This has been undergo ica, and this will be use as an input for epoching      
│   │   │   ├── ep30/
    │   │       ├── epoch/
    │   │       │   ├──  eeg_eog_epo.fif # This is epochs from eog_eeg_clean_raw.fif, it has been epoch and subject to autoreject, and this will be used for feature extraction and modeling.

[//]: # (    │   │       │   ├──  ear_epo.fif # this is EAR in fif format  , not sure kita perlu make it as epoch x skrang ni )
    │   │       │   ├──   
    │   │       ├── gt/
│   │   │       │   ├── delta/
    │   │       │   │ ├── lb2.parquet           # binary
    │   │       │   │ ├── lb3.parquet           # ternary
    │   │       │   │ └── label_sanity.html     # per-segment sanity plots
│   │   │       │   ├── ta/                     # theta/alpha
    │   │       │   │ ├── lb2.parquet           # binary
    │   │       │   │ ├── lb3.parquet           # ternary
    │   │       │   │ └── label_sanity.html     # per-segment sanity plots
    │   │       ├── feat/
    │   │       │   ├──  pyblinker.parquet    
    │   │       │   ├──  mne_features.parquet
    │   │       │   ├──  combined.parquet
    │   │       │   ├── feature_schema.json  
    │   │       ├── qc/
    │   │       │   ├── sync_report.html
    │   │       │   ├── signal_quality.json
                ├── manifest.json
│   │   ├── S01_20170519_043933_2/
│   │   ├── S01_20170519_043933_3/
│   ├── S2/
│   └── ...
└── _experiments/  
    └── runs/
        └── <hash8>/
            ├── manifest.json
            ├── STAGE_04_MERGE.DONE
            ├── STAGE_05_TRAIN.DONE
            ├── STAGE_06_XAI.DONE
            ├── DONE.ok
            │
            ├── 00_config/
            │   ├── run_key.json              # canonical dict that generated hash8
            │   ├── config_resolved.yaml      # frozen resolved config for reproducibility
            │   └── env.txt
            │
            ├── 01_data_index/
            │   ├── segments_used.txt
            │   ├── splits.json
            │   └── join_report.html
            │
            ├── 02_labels/
            │   ├── y.parquet
            │   └── label_summary.json
            │
            ├── 03_features/
            │   ├── X.parquet
            │   ├── feature_list.txt
            │   └── missingness_summary.json
            │
            ├── 04_stats/
            │   ├── corr_table.parquet
            │   ├── stability_table.parquet
            │   └── stats_report.html
            │
            ├── 05_models/
            │   ├── model.pkl
            │   ├── metrics.json
            │   ├── cv_results.json
            │   └── predictions.parquet
            │
            ├── 06_xai/
            │   ├── shap_values.npz
            │   ├── shap_summary.png
            │   ├── pdp_ice/
            │   └── xai_stability.json
            │
            ├── 07_plots/
            └── 08_logs/
                ├── run.log
                └── errors.log
    └── hash_mapping/
        ├── README.md
        ├── runs_index.parquet          # primary database (fast, typed, scalable)
        ├── runs_index.csv              # optional human-friendly export
        ├── runs_index.jsonl            # optional for streaming/log-style append
        ├── hash_to_path.json           # minimal {hash8: "runs/<hash8>"} mapping
        └── schema.json                 # field definitions + allowed codes
```
