## 🧭 Overall Goal

Evaluate how closely **pyblinker** replicates **BLINKER**’s blink-position detection using **public EEG datasets** that contain clear blink components.
Here, **BLINKER** (MATLAB) serves as the **ground-truth reference**, unlike earlier work on the *Drowsy Driving Raja* dataset, which relied on **human-labeled blink regions**.

---

## 📂 Dataset Sources (Ground Truth Benchmarks)

| Dataset                                                              | Citation                                                                                 | Host                 | Notes                                                                           |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------- | ------------------------------------------------------------------------------- |
| **Kaya et al. (2018)** – *A large EEG motor imagery dataset for BCI* | DOI [10.6084/m9.figshare.c.3917698.v1](https://doi.org/10.6084/m9.figshare.c.3917698.v1) | Figshare Collection  | Multiple subjects; .mat format; contains frontal EEG with blink activity        |
| **Cho et al. (2017)** – *EEG datasets for motor imagery BCI*         | DOI [10.5524/100295](https://doi.org/10.5524/100295)                                     | GigaScience Database | 52 subjects; includes resting and MI runs; blinks present in non-task intervals |

Both datasets provide **publicly available .mat EEG recordings** suitable for algorithmic comparison.

---

## ⚙️ Processing Concept (High-Level Flow)

1. **Data Ingestion**

    * Download each dataset into `dataset/public_raw/…`.
    * Unpack and catalog `.mat` files per subject/session.

2. **Standardization**

    * Convert `.mat` → **MNE Raw** format.
    * Normalize sampling rate, channel names, montage, and reference.
    * Store standardized files under `dataset/public_mne/…`.

3. **BLINKER Ground-Truth Generation**

    * Use the **MATLAB BLINKER** package via the MATLAB Engine interface.
    * Feed it the standardized signal.
    * Save the full BLINKER output and extract only:

        * `leftZero`
        * `rightZero`
    * Store in `blinker_zeroes.csv`.

4. **pyblinker Prediction**

    * Run **pyblinker** directly on the same `MNE Raw` input.
    * Export its detected `leftZero` / `rightZero` to `pyblinker_zeroes.csv`.

5. **Comparison**

    * Feed both CSVs into the **pyblinker comparison API** (`compare_zeroes`).
    * Compute overlap, temporal deviation, and detection accuracy.
    * Export results (`comparison_report.json`, `comparison_table.csv`).

---

## 🗂️ Output Hierarchy (Example)

```
dataset/
└── blinker_pyblinker_eval/
    ├── kaya2018/
    │   └── subject01/
    │       ├── blinker_out.mat
    │       ├── blinker_zeroes.csv
    │       ├── pyblinker_zeroes.csv
    │       ├── comparison_report.json
    │       └── comparison_table.csv
    └── cho2017/
        └── subject01/
            ├── ...
```

---

## 🧩 Key Principles

* **Ground truth assumption:** BLINKER output defines the reference; pyblinker is compared against it.
* **Minimal metric scope:** compare only **`leftZero`** and **`rightZero`** events.
* **Uniform input:** both tools operate on the same standardized MNE Raw signal.
* **Traceable artifacts:** every run produces auditable outputs (.mat, .csv, .json).
* **Dataset-agnostic framework:** easily extendable to other public EEG sets with blink components.

---

## 🧠 Purpose Summary

| Aspect            | Description                                                              |
| ----------------- | ------------------------------------------------------------------------ |
| **Objective**     | Validate pyblinker fidelity to BLINKER on public data.                   |
| **Baseline**      | BLINKER (MATLAB) detections = ground truth.                              |
| **Focus Metrics** | Blink onset/offset alignment via leftZero / rightZero.                   |
| **Outcome**       | Quantitative and visual reports of match rate and deviations.            |
| **Benefit**       | Confirms equivalence before deploying pyblinker across broader datasets. |

---

