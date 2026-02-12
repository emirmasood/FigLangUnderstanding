# Source Directory

This directory contains the reusable Python modules used across all project notebooks. The project follows a hybrid workflow:
- Notebooks orchestrate experiments (choose task/model/setting, call runners, export artifacts)
- `src/` provides stable, reusable implementations for preprocessing, dataset loading, training/evaluation, metrics, plotting, and utilities

The goal is to ensure that baselines and extensions:
- use identical processed inputs and split/index definitions
- compute metrics consistently
- save predictions/metrics/plots in a standardized layout
- remain reproducible across runs (fixed seeds and logged configs)

## Typical Responsibilities

### Data I/O and Schema
Modules in this group handle reading raw/processed files and enforcing a canonical schema used by all models.
Common functionality includes:
- loading CSV/JSON (and optionally Excel/Parquet if used)
- safe filename/path helpers
- canonicalizing column names and value normalization (e.g., task/source/variety naming)

Examples (names may vary by repo):
- `io_utils.py` — `load_any`, `save_any`, safe path/name helpers
- `schema.py` — `canonicalize()` to enforce required columns and consistent naming

### Text Normalization
Text normalization is kept model-agnostic so the same processed CSVs work for RoBERTa/BERT/Mistral.
Typical operations:
- whitespace cleanup
- URL/user mention normalization (if used)
- lightweight punctuation normalization
- optional lowercasing rules (while keeping raw text available)

Examples:
- `text_norm.py` — `normalize_text()`

### Split and Index Utilities
This group defines how train/validation splits are generated and how training/testing settings are indexed.
Typical outputs:
- per-task training settings index (e.g., `index_settings.csv`)
- per-task test sets index (e.g., `index_testsets.csv`)
- optional JSON split metadata for reproducibility

Examples:
- `splits.py` — stratified splitting helpers, JSON export utilities

### Dataset Loaders
Dataset loaders convert processed CSVs into the format expected by the training code.
Typical functionality:
- reading train/val/test CSVs based on index files
- tokenization and batching (for encoder models)
- label mapping and sanity checks

Examples:
- `dataset.py` or `data_loader.py`

### Model Training and Evaluation Runners
Runners implement the standardized experiment loop used by notebooks:
- load setting from index (train/val)
- train model
- evaluate on all required test sets
- save artifacts in a consistent structure

Typical outputs per run:
- `metrics/*.csv`
- `predictions/*.csv`
- `plots/*.png`
- `checkpoints/*`
- `analysis/*.csv`

Examples:
- `run_roberta.py`, `run_bert.py`, `run_mistral.py`
- or a single generic runner (e.g., `runner.py`) with model-specific wrappers

### Metrics and Reporting
Standardized computation across tasks and settings:
- accuracy, macro F1, precision, recall
- per-variety breakdowns
- per-source breakdowns (when applicable)
- pivoted tables for heatmaps and summary plots

Examples:
- `metrics.py` — metric computation and aggregation
- `reporting.py` — writing tables in a consistent format

### Plotting (Paper-style Figures)
Plotting utilities produce the final figures in a consistent style across notebooks and models:
- cross-variety matrix heatmaps (annotated cells)
- grouped bar charts for cross-domain/in-domain comparisons
- saved automatically to `plots/`

Examples:
- `plotting.py` — functions to generate and export all “paper-style” figures
