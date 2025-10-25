# 📂 Folder Structure

```plaintext
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
│   │   └── MD.mff/
│   │       ├── ...
│   │       ├── TEST_20170601_042544.mov
│   │       ├── TEST_20170601_042544_2.mov
│   │       ├── TEST_20170601_042544_3.mov
│   └── ...
│
├── cvat_zip_final/                          # ZIP files from CVAT
│   ├── S1/
│   │   └── from_cvat/
│   │       ├── S01_20170519_043933.zip
│   │       ├── S01_20170519_043933_2.zip
│   │       ├── S01_20170519_043933_3.zip
│   ├── S2/
│   │   └── from_cvat/
│   │       ├── ...
│   └── ...
├── drowsy_driving_raja_processed/                      # CLEANED + SYNC OUTPUTS
│   ├── S1/
│   │   S1.fif
│   │   ├── S01_20170519_043933/
│   │   │   ├── seg_annotated_raw.fif
│   │   │   ├── eeg_clean_epo.fif
│   │   │   ├── seg_ear.fif # this is EAR in fif format
│   │   │   ├── ear_eog.fif
│   │   │   ├── /blinker/
│   │   │   │  ├── blinkFits.pkl
│   │   │   │  ├── blinkProps.pkl
│   │   │   │  ├── blinkStats.pkl
│   │   │   │  ├── blinks.pkl
│   │   │   │  ├── params.pkl
│   │   │   ├── /pyblinker/
│   │   │   │  ├── selected_ch.pkl
│   │   │   │  ├── blink_details.pkl
│   │   ├── S01_20170519_043933_2/
│   │   ├── S01_20170519_043933_3/
│   ├── S2/
│   └── ...
│
├── drowsy_driving_raja_features/                       # Optional ML-friendly outputs
│   ├── S1/
│   │   ├── S01_20170519_043933_bandpower.npy
│   │   ├── S01_20170519_043933_2_bandpower.npy
│   │   └── ...
│   └── ...

```

### 📈 Pipeline Flowchart

Stage 1 preprocessing is orchestrated by `pipeline/stage1_preprocessing.py`,
which executes these steps sequentially:

1. `preprocess_mff_to_fif`
2. `annotate_segments_with_blinks`
3. `eeg_ica_autoreject_clean`
4. `build_ear_annotated_raw`
5. `align_ear_eeg_eog_flash`

```plaintext
                   +------------------------------+
                   |    Raw EEG/EOG (.mff folder) |
                   |        MD.mff (EGI format)   |
                   +------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        |                   Step 1: preprocess_mff_to_fif.py           |
        |                      [Status: Partially Done]                |
        |                                                              |
        |   ✓  Load .mff                                               |
        |   ✓  Set EEG montage (GSN-HydroCel-129)                      |
        |   ✓  Set reference (VREF)                                    |
        |   ✓  Create bipolar EOG channels                             |
        |   ✓  Select EEG region(s) (e.g., 'frontal')                  |
        |   🔧 Apply EEG average reference                             |
        |   ✓  High-pass filter (e.g., 0.5 Hz)                         |
        |   🔧 Low-pass filter (e.g., 30–40 Hz)                        |
        |   ✓  Resample to 100 Hz                                      |
        |   ✓  Save S*/ S*.fif per subject                              |
        |      (e.g., S1/S1.fif, S2/S2.fif — where S* = subject ID)    |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        |              Step 2: annotate_segments_with_blinks.py        |
        |                      [Status: Done]                          |
        |                                                              |
        |   ✓ Load preprocessed EEG: S*/S*.fif (~90 min)               |
        |   ✓ Split into 3 segments based on driving video durations   |
        |     using config (videoEndTime, shift offsets, etc.)         |
        |   ✓ Load EAR (.pkl) and CVAT (.zip) blink annotations        |
        |   ✓ Align annotations using frame-to-time mapping            |
        |   ✓ Annotate each segment with blink events                  |
        |   ✓ Save per-segment FIF:                                    |
        |       S*/S*_<session_id>/seg_annotated_raw.fif              |
        |       (where <session_id> = video name like S01_20170519_043933[_2][_3]) |
        |                                                              |
        |   Example:                                                   |
        |     drowsy_driving_raja_processed/                           |
        |     └── S1/S01_20170519_043933/seg_annotated_raw.fif         |
        |                                                              |
        |   Naming Convention(e.g.,):                                  |
        |     S01_20170519_043933    → Segment 1                       |
        |     S01_20170519_043933_2  → Segment 2                       |
        |     S01_20170519_043933_3  → Segment 3                       |
        +--------------------------------------------------------------+
                     |                |                |
           ┌─────────┘                |                └─────────┐
           v                          v                          v
     Segment 1                 Segment 2                   Segment 3
  (first driving video)   (second driving video)     (third driving video)
                                      |
                   (process each segment separately)
                                      v
            +--------------------------------------------------------------+
            |             Step 3: eeg_ica_autoreject_clean.py              |
            |                      [Status: In Progress]                   |
            |                                                              |
            |   ✓ Load seg_annotated_raw.fif per segment                   |
            |   ✓ Pick EEG+EOG channels                                    |
            |   ✓ High-pass filter at 1 Hz (for ICA stability)            |
            |   ✓ Fit ICA (n_components=15)                                |
            |   ✓ Detect EOG components using find_bads_eog                |
            |   ✓ Apply ICA to remove blink artifacts                      |
            |   ✓ Save ICA-cleaned raw as eeg_clean_raw.fif               |
            |   ✓ Epoch into 30-sec fixed-length chunks                    |
            |   ✓ Apply AutoReject (EEG only) to clean epochs              |
            |   ✓ Save final cleaned epochs as eeg_clean_epo.fif          |
            |                                                              |
            |   Output Example:                                            |
            |     drowsy_driving_raja_processed/                           |
            |     └── S1/S01_20170519_043933/                              |
            |         ├── eeg_clean_raw.fif                                |
            |         └── eeg_clean_epo.fif                                |
            +--------------------------------------------------------------+

In parallel:

            +--------------------------------------------------------------+
            |               Step 4: build_ear_annotated_raw.py             |
            |                      [Status: Done]                          |
            |                                                              |
            |   ✓ Load EAR signal (.pkl) from MOV (30 Hz)                  |
            |   ✓ Load blink annotations from CVAT ZIP                     |
            |   ✓ Align CVAT blinks to EAR using video frame metadata      |
            |   ✓ Construct RawArray with 3 misc channels:                 |
            |       - left_ear, right_ear, avg_ear                         |
            |   ✓ Add annotations to Raw (blink events as `BAD_blink`)     |
            |   ✓ Save output per segment:                                 |
            |       S*/S*_<session_id>/seg_ear.fif                         |
            |       (EAR signal with blink annotations @ 30 Hz)            |
            |                                                              |
            |   Example:                                                   |
            |     drowsy_driving_raja_processed/                           |
            |     └── S1/S01_20170519_043933/seg_ear.fif                   |
            +--------------------------------------------------------------+
                                      |
                                      v
        +--------------------------------------------------------------+
        |         Step 5: align_ear_eeg_eog_flash.py                   |
        |                      [Status: Done]                          |
        |                                                              |
        |   ✓ Load segmented EEG + EOG: seg_annotated_raw.fif (100 Hz) |
        |   ✓ Load EAR with blink events: seg_ear.fif (30 Hz)          |
        |   ✓ Detect flash events in both modalities                   |
        |   ✓ Align EAR to EEG using flash timestamps                  |
        |   ✓ Rename & retype channels (e.g., 'EEG-E1', 'EAR-left_ear')|
        |   ✓ Resample EEG down to 30 Hz (if needed)                   |
        |   ✓ Time-trim both streams to equal duration                 |
        |   ✓ Merge EEG + EOG + EAR into one Raw object                |
        |   ✓ Save per-segment combined file:                          |
        |       S*/S*_<session_id>/ear_eog.fif                         |
        |       (Final aligned + annotated multimodal signal @ 30 Hz) |
        |                                                              |
        |   Example:                                                   |
        |     drowsy_driving_raja_processed/                           |
        |     └── S1/S01_20170519_043933/ear_eog.fif                   |
        +--------------------------------------------------------------+
```

