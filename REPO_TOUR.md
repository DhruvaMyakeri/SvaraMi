# Repository Tour: Speech-Disease-Observation

A research platform for detecting and profiling neurodegenerative diseases (Alzheimer's and Parkinson's) through acoustic and lexical analysis of speech recordings. The methodology follows Botelho et al. (2024) — no machine learning, only signal processing and statistical normalization to produce clinically interpretable biomarker profiles.

---

## Table of Contents

1. [Repository at a Glance](#1-repository-at-a-glance)
2. [Directory Structure](#2-directory-structure)
3. [Datasets](#3-datasets)
4. [Alzheimer's Pipeline (Primary)](#4-alzheimers-pipeline-primary)
   - [Architecture & Data Flow](#architecture--data-flow)
   - [Stage-by-Stage Breakdown](#stage-by-stage-breakdown)
   - [Features Extracted](#features-extracted)
   - [Outputs](#outputs)
5. [Parkinson's Pipeline (src/)](#5-parkinsons-pipeline-src)
   - [Architecture & Data Flow](#architecture--data-flow-1)
   - [Module Breakdown](#module-breakdown)
   - [Outputs](#outputs-1)
6. [Technology Stack](#6-technology-stack)
7. [Key Design Decisions](#7-key-design-decisions)
8. [Running the Pipelines](#8-running-the-pipelines)

---

## 1. Repository at a Glance

| Property | Detail |
|---|---|
| **Purpose** | Speech biomarker extraction & deviation profiling for AD and PD |
| **Methodology** | Acoustic signal processing (Praat) + z-score normalization — no ML |
| **Primary Dataset (AD)** | DementiaBank Pitt Corpus (~1,200 AD + ~600 HC recordings) |
| **Primary Dataset (PD)** | UCI Oxford PD, UCI Telemonitoring PD, CLAC (normative reference) |
| **Features** | 27 acoustic + 5 lexical (optional, if transcripts available) |
| **Outputs** | CSV tables, radar plots, heatmaps, boxplots, violin plots, summary reports |
| **Language** | Python 3.8+ |
| **Core Library** | Parselmouth (Python binding for Praat) |

---

## 2. Directory Structure

```
Speech-Disease-Observation/
│
├── alzheimers_pipeline/           # Complete AD analysis pipeline (10 stages)
│   ├── run_pipeline.py            # CLI entry point & orchestrator
│   ├── data_loader.py             # File discovery & manifest construction
│   ├── feature_extractor.py       # Acoustic feature extraction (Parselmouth/Praat)
│   ├── transcript_feature_extractor.py  # Lexical features from CHAT .cha transcripts
│   ├── healthy_reference.py       # HC normative reference interval builder
│   ├── deviation_scoring.py       # Z-score deviation computation
│   ├── radar_plots.py             # Radar (spider) plot visualization
│   ├── heatmap_visualization.py   # Heatmap & per-feature distribution plots
│   ├── regenerate_radar_plots.py  # Utility: rebuild radar plots from saved CSVs
│   ├── requirements.txt           # Python dependencies
│   ├── README.md                  # Pipeline documentation
│   ├── pipeline.log               # Execution log from last run
│   └── outputs/                   # All generated results (see §4 Outputs)
│
├── src/                           # Parkinson's analysis modules
│   ├── feature_extractor.py       # CLAC acoustic feature extraction
│   ├── csv_loader_oxford_pd.py    # UCI Oxford PD loader & feature mapping
│   ├── csv_loader_telemonitoring.py  # UCI Telemonitoring PD loader
│   ├── deviation_scoring.py       # In-corpus PD deviation scoring
│   ├── confound_gate.py           # Cross-corpus reference contamination check
│   ├── reference_intervals.py     # Bootstrap CI on RI limits (Botelho Step 5)
│   └── merge_metadata.py          # Metadata integration utility
│
├── outputs/                       # PD analysis results (see §5 Outputs)
│   ├── features/
│   ├── reference_intervals/
│   ├── deviation_scores/
│   ├── confound_gate/
│   └── figures/
│
├── CLAC-Dataset/                  # Colombian Linguistic Atlas (~18,600 WAV files)
├── DEMENTIAbANK/                  # Pitt Corpus (AD study, loaded by AD pipeline)
├── parkinsons/                    # UCI Oxford Parkinson's Dataset
├── parkinsons+telemonitoring/     # UCI Parkinson's Telemonitoring Dataset
└── .gitignore                     # Excludes datasets, venv, __pycache__
```

---

## 3. Datasets

### DementiaBank — Pitt Corpus
Used by the Alzheimer's pipeline. Contains Cookie Theft picture description recordings, with speakers divided into AD patients and healthy controls.

- **Format:** `.wav` audio + `.cha` CHAT-format transcripts
- **Scale:** ~1,200 AD recordings, ~600 HC recordings (1,102 processed in last run)
- **Gitignored:** Data must be obtained separately

### CLAC — Colombian Linguistic Atlas of Speech
Used as the normative reference population for the Parkinson's pipeline.

- **Tasks:** 12 types — Cookie Theft, Counting (1–20), Days of the Week, Grandfather passage, Max Phonation, Picnic scene, Rainbow passage, Repeat 5× (with variants: *artillery*, *catastrophe*, *impossibility*), Spontaneous Monologue
- **Format:** `.wav` + `.txt` transcript pairs per speaker
- **Scale:** 18,609 WAV files, 18,613 transcript files, 1,000+ Colombian speakers
- **Metadata:** Age, gender, education, location, health symptoms (Excel)

### UCI Oxford Parkinson's Dataset
195 PD subjects + healthy controls. Pre-extracted sustained vowel features from MDVP voice analysis software.

- **Features:** ~22 columns including MDVP pitch, jitter, shimmer, HNR, RPDE, DFA
- **File:** `parkinsons/parkinsons.data`

### UCI Parkinson's Telemonitoring Dataset
42 PD patients, multiple longitudinal recordings each, UPDRS severity scores included.

- **Note:** All subjects are PD patients — no healthy controls. CLAC used as normative reference.
- **File:** `parkinsons+telemonitoring/parkinsons_updrs.data`

---

## 4. Alzheimer's Pipeline (Primary)

### Architecture & Data Flow

```
AD directory (WAV)  ──┐
                       ├──► [1] data_loader.py           → manifest.csv
HC directory (WAV)  ──┘
                                │
                                ▼
                       [2] feature_extractor.py           → extracted_features.csv
                                │
                                ▼
                       [3] transcript_feature_extractor.py → (5 lexical features,
                                │                             if .cha files present)
                                ▼
                       [4] Merge acoustic + lexical        → merged feature table
                                │
                                ▼
                       [5] healthy_reference.py            → healthy_reference.csv
                                │
                                ▼
                       [6] deviation_scoring.py            → deviation_scores.csv
                                │
                    ┌───────────┼────────────────┐
                    ▼           ▼                ▼
             [7] radar_plots  [8] heatmap  [9] distributions   [10] report
                 (3 PNGs)      (PNG)        (54 PNGs)           (TXT + CSV)
```

### Stage-by-Stage Breakdown

#### Stage 1 — `data_loader.py`
Recursively scans AD and HC directories for `.wav` files. For each file:
- Derives a `speaker_id` from the filename (e.g. `001-0.wav` → `001`)
- Searches up to 4 nearby locations for a paired `.cha` transcript
- Validates the file is non-zero in size

Output: `manifest.csv` — one row per recording with columns `speaker_id`, `file_name`, `file_path`, `group` (AD/HC), `cha_path`.

#### Stage 2 — `feature_extractor.py`
Loads each WAV via Parselmouth (Python Praat binding) and extracts 27 acoustic features across 6 groups (see [Features Extracted](#features-extracted)). Evidence-based Praat parameters:

| Parameter | Value | Rationale |
|---|---|---|
| Pitch floor | 75 Hz | Captures low male voices |
| Pitch ceiling | 500 Hz | Avoids octave errors |
| Silence threshold | −25 dB | Praat default |
| Min pause duration | 150 ms | Psycholinguistic standard |
| Formant max | 5.5 kHz | Standard for speech analysis |

Failed extractions return a row of `NaN` values and are logged — the pipeline never crashes.

#### Stage 3 — `transcript_feature_extractor.py`
Parses `.cha` CHAT-format transcripts (DementiaBank format). Only `*PAR:` (participant) lines are used; investigator lines and header metadata are ignored. Extracts 5 lexical features after stripping CHAT special markers (`[//]`, `[/]`, `<>`, `&um`, etc.).

> In the last run, 0 of 1,102 recordings had paired `.cha` files, so transcript features were skipped. They activate automatically if `.cha` files are co-located with WAV files.

#### Stage 4 — Merge
Acoustic and lexical DataFrames are joined on `file_path`. The active feature list is the union of available columns.

#### Stage 5 — `healthy_reference.py`
Computes normative statistics from HC recordings only, per feature:
- mean, std, median, Q25, Q75, IQR
- lower_95_bound = mean − 1.96 × std
- upper_95_bound = mean + 1.96 × std
- n_valid (sample count)

Requires ≥ 3 HC observations per feature. Features with std = 0 produce `NaN` z-scores downstream.

#### Stage 6 — `deviation_scoring.py`
Computes individual z-scores:

```
z = (feature_value − HC_mean) / HC_std
```

- z ≈ 0 → within healthy range
- z > +2 → abnormally HIGH
- z < −2 → abnormally LOW

Also computes per-participant summary stats: `abs_z_mean` (mean |z| across all features, global severity proxy) and `max_abs_z` (single most deviant feature).

#### Stage 7 — `radar_plots.py`
Generates 3 publication-quality radar (spider) plots. Each axis is one feature; the radial coordinate is `z_score + MAX_Z` (converting signed z to non-negative for display). A green shaded band marks the 95% healthy range.

| Plot | Content |
|---|---|
| `radar_hc.png` | HC individual traces (faint blue) + HC mean (thick blue) |
| `radar_ad.png` | AD individual traces (faint red) + AD mean (thick red) |
| `radar_comparison.png` | HC and AD means side-by-side |

Individual traces are capped at 30 per group to keep plots readable.

#### Stage 8 — `heatmap_visualization.py` (Heatmap)
Participants × features z-score matrix using diverging coolwarm colormap centered at z = 0, clipped to ±4. Rows are sorted: HC first (by abs_z_mean ascending), then AD. A separator line marks the boundary between groups.

#### Stage 9 — `heatmap_visualization.py` (Distribution Plots)
One boxplot + stripplot and one violin plot per feature (54 files total), saved to `boxplots/` and `violinplots/`. Falls back gracefully to boxplot if n < 4 per group (insufficient data for KDE).

#### Stage 10 — Summary Report
Writes `summary_report.txt` (human-readable) and `group_feature_stats.csv` (detailed per-group descriptive statistics).

---

### Features Extracted

#### Acoustic Features (27)

| Group | Feature(s) | Count |
|---|---|---|
| **F0 (Pitch)** | mean, std, min, max | 4 |
| **HNR** | harmonics-to-noise ratio | 1 |
| **Jitter** | local, absolute, RAP, PPQ5 | 4 |
| **Shimmer** | local, dB, APQ3, APQ5, APQ11 | 5 |
| **Formants** | F1 mean, F2 mean, F3 mean, F4 mean | 4 |
| **Timing** | recording duration, speaking duration, n_pauses, mean pause, max pause, total pause duration, silence ratio, speech rate, articulation rate, phonation time ratio | 9 |

#### Lexical Features (5, optional)

| Feature | Description |
|---|---|
| `total_words` | Token count |
| `unique_words` | Vocabulary size |
| `type_token_ratio` | Lexical diversity (unique / total) |
| `mean_utterance_len` | Words per utterance |
| `avg_word_length` | Mean character length per word |

---

### Outputs

```
alzheimers_pipeline/outputs/
│
├── manifest.csv                   # File discovery log (1,102 rows)
├── extracted_features.csv         # All acoustic features (1,102 × 27)
├── healthy_reference.csv          # HC normative stats (27 features × 10 columns)
├── deviation_scores.csv           # Z-scores per recording (1,102 × 30 columns)
│
├── deviation_heatmap.png          # Participants × features heatmap (coolwarm, ±4)
├── radar_hc.png                   # HC profile radar (99 HC participants)
├── radar_ad.png                   # AD profile radar (194 AD participants)
├── radar_comparison.png           # HC vs AD side-by-side radar
│
├── boxplots/                      # 27 per-feature boxplots (HC vs AD)
├── violinplots/                   # 27 per-feature violin plots (HC vs AD)
│
└── reports/
    ├── summary_report.txt         # Human-readable statistics
    └── group_feature_stats.csv    # Detailed group descriptive stats
```

**Last run stats (from `pipeline.log`):**
- 1,102 recordings (618 AD, 484 HC)
- 100% feature extraction success rate
- 27 active features
- 99 unique HC participants, 194 unique AD participants
- Total runtime: ~2.5 hours

---

## 5. Parkinson's Pipeline (src/)

### Architecture & Data Flow

```
CLAC WAV files ─────────────────────────────────────────────────────────────┐
                                                                             │
UCI Oxford PD CSV ──────┐                                                    │
                         ├──► standardized feature tables                    │
UCI Telemonitoring CSV ─┘                                                    │
                                                                             ▼
                                                          [reference_intervals.py]
                                                          Bootstrap 95% RIs from
                                                          CLAC HC (per task, gender)
                                                                             │
                                 [confound_gate.py] ◄────────────────────────┘
                                 Check if CLAC reference separates
                                 from Oxford HC before disease scoring
                                                                             │
                                                 ┌───────────────────────────┘
                                                 ▼
                              [deviation_scoring.py]
                              In-corpus z-scores (Oxford HC as reference
                              for both Oxford PD + Telemonitoring PD)
                                                 │
                                                 ▼
                                          Radar plots (3 PNGs)
```

### Module Breakdown

#### `src/feature_extractor.py`
Extracts 28 acoustic features from CLAC WAV files using Parselmouth, following Botelho et al. (2024) Table I exactly. Includes formant means and medians (F1–F4) and rhythm features specific to spontaneous speech tasks (speech rate, articulation rate, syllable duration, pause metrics).

#### `src/csv_loader_oxford_pd.py`
Loads `parkinsons.data` and maps 11 MDVP columns to standardized Botelho feature names:

| Oxford Column | Standardized Name |
|---|---|
| MDVP:Fo(Hz) | meanF0 |
| HNR | HNR |
| MDVP:Jitter(%) | localJitter |
| MDVP:Jitter(Abs) | localabsoluteJitter |
| MDVP:RAP | rapJitter |
| MDVP:PPQ | ppq5Jitter |
| MDVP:Shimmer | localShimmer |
| MDVP:Shimmer(dB) | localdbShimmer |
| Shimmer:APQ3 | apq3Shimmer |
| Shimmer:APQ5 | aqpq5Shimmer |
| MDVP:APQ | apq11Shimmer |

Speaker ID is derived from recording name (e.g. `phon_R01_S01_1` → `R01_S01`). Condition: 0 → HC, 1 → PD.

#### `src/csv_loader_telemonitoring.py`
Loads `parkinsons_updrs.data`. All subjects are PD patients. Multiple recordings per subject are aggregated to per-subject means. UPDRS severity scores (motor and total) are retained.

#### `src/reference_intervals.py`
Computes reference intervals (2.5th–97.5th percentiles) with bootstrap confidence intervals from CLAC HC speakers, stratified by task and gender. Follows Botelho et al. (2024) Step 5.

#### `src/confound_gate.py`
Validates that cross-corpus CLAC normative reference is not contaminated by recording condition differences. Tests whether CLAC HC and Oxford PD HC already separate on deviation scores before adding clinical PD patients. If they do, confound is present and in-corpus normalization is required.

#### `src/deviation_scoring.py`
Uses Oxford PD healthy controls (n = 8) as the reference for computing z-scores for **both** Oxford PD and Telemonitoring PD datasets. This is the in-corpus approach adopted after confound gate analysis revealed cross-corpus (CLAC-based) normalization was unreliable.

#### `src/merge_metadata.py`
Utility that joins CLAC metadata (age, gender, education from Excel) with extracted acoustic feature tables.

---

### Outputs

```
outputs/
│
├── features/
│   ├── clac_cookie_theft_acoustic_features.csv
│   ├── clac_max_phonation_acoustic_features.csv
│   ├── clac_picnic_acoustic_features.csv
│   ├── clac_*_merged.csv          # Acoustic + metadata per task
│   ├── clac_all_merged.csv        # All CLAC tasks pooled
│   ├── oxford_pd_features.csv     # Oxford PD (standardized)
│   └── telemonitoring_pd_features.csv
│
├── reference_intervals/
│   ├── reference_intervals.csv    # 2.5th–97.5th percentiles + bootstrap CIs
│   ├── ri_width_*.png             # RI width distributions
│   └── reference_intervals_summary.txt
│
├── deviation_scores/
│   ├── oxford_pd_deviation.csv
│   ├── telemonitoring_pd_deviation.csv
│   └── pd_combined_deviation.csv
│
├── confound_gate/
│   ├── confound_gate_results.csv
│   ├── confound_gate_summary.txt
│   └── confound_gate_distributions.png
│
└── figures/
    ├── radar_oxford_pd.png
    ├── radar_telemonitoring_pd.png
    └── radar_pd_combined.png
```

---

## 6. Technology Stack

| Library | Version | Role |
|---|---|---|
| **praat-parselmouth** | ≥ 0.4.3 | Python binding for Praat; all acoustic feature extraction |
| **pandas** | ≥ 2.0.0 | Data manipulation and CSV I/O |
| **numpy** | ≥ 1.26.0 | Numerical computing |
| **scipy** | ≥ 1.11.0 | Bootstrap CIs, statistical tests |
| **matplotlib** | ≥ 3.8.0 | Radar plots, heatmaps, all publication figures |
| **seaborn** | ≥ 0.13.0 | Boxplots, violin plots, distribution plots |
| **tqdm** | ≥ 4.66.0 | Progress bars during long feature extraction runs |

Install via:
```bash
pip install -r alzheimers_pipeline/requirements.txt
```

---

## 7. Key Design Decisions

**No machine learning.** All analysis is signal processing + z-score normalization. Results are directly interpretable as "how many standard deviations from healthy average" per feature.

**Graceful degradation.** Any failed WAV load or feature extraction returns a `NaN` row rather than crashing. Missing transcript files simply omit lexical features. The pipeline always completes.

**In-corpus normalization for PD.** The confound gate analysis revealed that CLAC HC and Oxford PD HC recordings already differ due to equipment and session differences — not disease. The PD pipeline uses Oxford HC (n = 8) as reference instead.

**Parametric 95% bounds.** Healthy reference intervals use mean ± 1.96 × std (Gaussian assumption), which is standard in clinical speech normative studies.

**Speaker-level aggregation for radar plots.** Multiple recordings per speaker are averaged to one deviation profile before radar visualization, so each line represents a person rather than a session.

**Botelho et al. (2024) compliance.** Feature selection, Praat parameters, and reference interval methodology are all traceable to this published protocol.

---

## 8. Running the Pipelines

### Alzheimer's Pipeline

```bash
cd alzheimers_pipeline

# Basic run
python run_pipeline.py \
    --ad_dir /path/to/dementiabank/AD \
    --hc_dir /path/to/dementiabank/HC \
    --output_dir outputs/

# Skip per-feature distribution plots (faster)
python run_pipeline.py \
    --ad_dir /path/to/AD \
    --hc_dir /path/to/HC \
    --skip_dist

# Verbose logging
python run_pipeline.py \
    --ad_dir /path/to/AD \
    --hc_dir /path/to/HC \
    --log_level DEBUG

# Rebuild only radar plots from existing CSVs
python regenerate_radar_plots.py
```

### Parkinson's Pipeline

Individual modules are run directly. Typical order:

```bash
# 1. Extract CLAC features (produces clac_*_acoustic_features.csv)
python src/feature_extractor.py

# 2. Load UCI datasets
python src/csv_loader_oxford_pd.py
python src/csv_loader_telemonitoring.py

# 3. Compute reference intervals from CLAC HC
python src/reference_intervals.py

# 4. Run confound gate check
python src/confound_gate.py

# 5. Compute in-corpus deviation scores
python src/deviation_scoring.py
```

---

*Generated from repository state as of 2026-07-01.*
