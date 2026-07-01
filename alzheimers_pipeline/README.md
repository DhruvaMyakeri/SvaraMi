# Alzheimer's Disease Speech Biomarker Analysis Pipeline

A complete, production-ready pipeline for extracting clinically interpretable
speech biomarkers and visualising deviation profiles relative to healthy
control reference intervals — inspired by published clinical speech deviation
profiling studies.

---

## Pipeline Overview

```
WAV Files (AD + HC)
       │
       ▼
[data_loader.py]           Discover files, build manifest
       │
       ▼
[feature_extractor.py]     Acoustic features via Parselmouth/Praat
       │
[transcript_extractor.py]  Lexical features from .cha transcripts (optional)
       │
       ▼
[healthy_reference.py]     Build normative HC reference intervals
       │
       ▼
[deviation_scoring.py]     Compute z-scores relative to HC
       │
       ▼
[radar_plots.py]           Radar plots (HC profile / AD profile / comparison)
[heatmap_visualization.py] Deviation heatmap + distribution plots
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Organise your data

```
data/
├── pitt_corpus/          ← Alzheimer's Disease recordings
│   ├── 001-0.wav
│   ├── 001-0.cha         ← optional CHAT transcript
│   └── ...
└── healthy_controls/     ← Healthy Control recordings
    ├── HC001.wav
    └── ...
```

Folder layout within each directory can be **flat or nested** — the loader
walks recursively.

### 3. Run the pipeline

```bash
python run_pipeline.py \
    --ad_dir data/pitt_corpus \
    --hc_dir data/healthy_controls \
    --output_dir outputs
```

#### Optional flags

| Flag | Effect |
|------|--------|
| `--skip_dist` | Skip per-feature boxplot/violin plots (faster for large datasets) |
| `--log_level DEBUG` | Verbose logging including Praat warnings |

---

## Output Files

```
outputs/
├── manifest.csv                  File discovery log
├── extracted_features.csv        All features for all recordings
├── healthy_reference.csv         HC reference: mean, std, 95% bounds
├── deviation_scores.csv          z-scores per participant per feature
│
├── radar_hc.png                  HC speech profile radar
├── radar_ad.png                  AD deviation radar
├── radar_comparison.png          HC vs AD comparison radar
├── deviation_heatmap.png         Participant × feature heatmap
│
├── boxplots/
│   └── <feature>_boxplot.png
├── violinplots/
│   └── <feature>_violin.png
└── reports/
    ├── summary_report.txt
    └── group_feature_stats.csv
```

---

## Features Extracted

### Acoustic (Parselmouth/Praat)

| Category | Features |
|----------|----------|
| **F0** | mean, std, min, max |
| **HNR** | harmonics-to-noise ratio |
| **Jitter** | local, absolute, RAP, PPQ5 |
| **Shimmer** | local, dB, APQ3, APQ5, APQ11 |
| **Formants** | F1–F4 mean |
| **Timing** | recording duration, speaking duration, pause count, mean/max/total pause, silence ratio, speech rate, articulation rate, phonation time ratio |

### Lexical (CHAT transcripts, optional)

| Feature | Description |
|---------|-------------|
| `total_words` | Word token count |
| `unique_words` | Vocabulary size |
| `type_token_ratio` | Lexical diversity (unique / total) |
| `mean_utterance_len` | Mean words per utterance |
| `avg_word_length` | Mean character count per word |

---

## Key Design Decisions

### No Machine Learning
All biomarkers are computed from signal processing and interpretable
statistics. The deviation score is a standard z-score against the HC
reference population.

### Healthy Centre = z = 0 on Radar Plots
Radar axes show **absolute** z-scores so all deviations point outward.
The green band at |z| ≤ 2 represents the healthy reference range.

### Graceful Degradation
- Corrupted or zero-byte WAV files: logged and skipped
- Missing .cha transcripts: transcript columns are NaN; acoustic features unaffected
- Features where HC std = 0: z-scores set to NaN, excluded from mean deviation

### Praat Parameters (evidence-based defaults)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Pitch floor | 75 Hz | Captures low male voices |
| Pitch ceiling | 500 Hz | Avoids octave errors |
| Silence threshold | −25 dB | Standard Praat default |
| Min pause | 150 ms | Psycholinguistic pause definition |
| Formant max | 5500 Hz | Standard for speech analysis |

---

## Using as a Library

```python
from data_loader import discover_dataset
from feature_extractor import extract_all_features
from healthy_reference import build_reference
from deviation_scoring import compute_deviations
from radar_plots import make_all_radar_plots
from feature_extractor import ACOUSTIC_FEATURE_COLUMNS

manifest = discover_dataset("data/pitt", "data/hc")
features = extract_all_features(manifest)
ref = build_reference(features, ACOUSTIC_FEATURE_COLUMNS)
deviations = compute_deviations(features, ref, ACOUSTIC_FEATURE_COLUMNS)
make_all_radar_plots(deviations, ACOUSTIC_FEATURE_COLUMNS, "outputs/")
```

---

## Code Files

| File | Purpose |
|------|---------|
| `data_loader.py` | File discovery, manifest construction |
| `feature_extractor.py` | Parselmouth/Praat acoustic feature extraction |
| `transcript_feature_extractor.py` | CHAT transcript lexical features |
| `healthy_reference.py` | HC normative reference intervals |
| `deviation_scoring.py` | z-score deviation computation |
| `radar_plots.py` | Radar/spider plot visualisations |
| `heatmap_visualization.py` | Heatmap + distribution plots |
| `run_pipeline.py` | CLI orchestrator, full pipeline runner |
| `requirements.txt` | Python dependency list |
