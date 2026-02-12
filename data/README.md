# Data Directory

This directory contains all dataset files and derived artifacts used in the DNLP project on Figurative Language Understanding.

The repository follows a separation between:
- `raw/`: original provided data files (not edited)
- `processed/`: canonical, model-agnostic CSVs used by all training/evaluation notebooks
- `splits/`: split definitions and reproducibility helpers
- `reports/`: lightweight summaries produced during preprocessing and exploration

## Directory Structure

### `data/raw/`
Stores the original dataset files exactly as provided (no manual edits). Expected contents typically include:
- `besstie_train.*`
- `besstie_validation.*`

These files are the input to the preprocessing pipeline.

### `data/processed/`
Stores cleaned, canonical CSVs used by all models (BERT, RoBERTa, Mistral) to ensure consistent comparisons across baselines and extensions.

Common conventions:
- Task-level folders use lowercase names: `sentiment/`, `sarcasm/`
- Files are model-agnostic and include at minimum:
  - `text`, `label`, `task`, `variety_name`, `source_name`
  - normalized text column (e.g., `text_norm`) when applicable
  - stable IDs (e.g., `row_id`) for traceability

Typical structure per task:
- `data/processed/<task>/trainsets/`
  - training pools defined by evaluation setting (e.g., source- or variety-specific)
- `data/processed/<task>/testsets/`
  - test sets for evaluation (e.g., FULL, per-source, per-variety)
  - `index_testsets.csv` mapping test settings to CSV files
- `data/processed/<task>/index_settings.csv`
  - maps each training setting to its corresponding train/val CSV files
- `data/processed/<task>/splits/`
  - split metadata or saved indices used for reproducibility (if exported)

Notes:
- Sentiment typically supports both source-based and variety-based settings.
- Sarcasm is treated according to the benchmark protocol (commonly Reddit-only), and settings focus on FULL and variety-based training.

### `data/splits/`
Stores global split artifacts and bookkeeping used across notebooks (if applicable), such as:
- saved random seeds / split indices
- JSON files describing split composition

### `data/reports/`
Stores small diagnostic outputs produced during preprocessing, such as:
- dataset composition summaries
- sanity checks and audit tables
- label/source/variety distributions

## Reproducibility
All experiments assume:
- consistent usage of `data/processed/` across baselines and extensions
- fixed seeds recorded in the notebooks/configs
- training/test settings resolved via `index_settings.csv` and `index_testsets.csv`
