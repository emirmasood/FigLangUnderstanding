# Source Directory

This directory contains the reusable Python modules used across all notebooks in the DNLP project on Figurative Language Understanding.

The repository follows a hybrid workflow:
- notebooks: orchestration (select model/task/setting, run experiments, export artifacts)
- `src/`: shared implementations to keep baselines and extensions consistent
- `data/processed/`: model-agnostic inputs consumed by all runners

## Directory Structure

### `src/io_utils.py` and schema utilities
Utilities for safe file I/O and enforcing a canonical dataset schema.
Typical responsibilities include:
- loading CSV/JSON files used by the pipeline
- safe naming/path helpers for consistent artifact naming
- canonicalizing required columns and standardizing values (task/source/variety naming)

### `src/text_norm.py`
Text normalization utilities used during preprocessing to produce model-agnostic inputs.
Typical operations include:
- whitespace cleanup and normalization
- lightweight normalization rules that do not depend on a specific model tokenizer
- generating a normalized column (e.g., `text_norm`) while preserving raw `text`

### `src/splits.py`
Split and reproducibility helpers used in preprocessing.
Typical responsibilities include:
- stratified splitting utilities
- saving split metadata (e.g., JSON indices) when needed
- helpers used to build train/validation pools for different settings

### `src/data_loading/` (or dataset loader module)
Dataset loading utilities that convert processed CSVs into the format required by model trainers.
Typical responsibilities include:
- reading train/val/test CSVs based on index files
- label mapping and sanity checks
- tokenization/batching for encoder models (BERT/RoBERTa/XLM-R)

### `src/runners/` (or model runner modules)
Standardized training and evaluation entry points used by the notebooks.
Typical responsibilities include:
- loading a training setting from `index_settings.csv`
- training a model and saving checkpoints
- evaluating on all required test sets from `index_testsets.csv`
- exporting standardized artifacts (metrics, predictions, plots, analysis)

### `src/metrics.py`
Metric computation and aggregation used across all models and runs.
Typical outputs include:
- accuracy, macro F1, precision, recall
- per-variety breakdown tables
- per-source/domain breakdown tables (when applicable)
- pivot tables used for heatmaps and summary figures

### `src/plotting.py`
Utilities for producing paper-style figures used in the final notebook cells.
Typical figures include:
- cross-variety matrix heatmaps (annotated cells)
- grouped bar charts for in-domain and cross-domain comparisons
- consistent color palette and automatic export to `plots/`

### `src/utils.py` (optional)
Small shared helpers used across modules and notebooks, such as:
- seeding and determinism helpers
- logging helpers
- common constants (label names, variety/source mappings)

## How the notebooks use `src/`
All experiment notebooks follow the same pattern:
1. Read processed indices for a task:
   - `data/processed/<task>/index_settings.csv`
   - `data/processed/<task>/testsets/index_testsets.csv`
2. Call a runner from `src/` to train and evaluate under a selected setting
3. Save standardized artifacts per run:
   - `metrics/*.csv`, `predictions/*.csv`, `plots/*.png`, `checkpoints/*`, `analysis/*.csv`

This structure ensures fair comparisons between baselines and extensions.

## Reproducibility
All experiments assume:
- consistent usage of `data/processed/` across baselines and extensions
- fixed seeds recorded in the notebooks/configs
- training/test settings resolved via `index_settings.csv` and `index_testsets.csv`
- standardized metrics and plotting functions shared through `src/`
