# Notebooks Directory

This directory contains the phase-based Colab notebooks used to run all experiments for the DNLP project on Figurative Language Understanding. Each notebook is designed to be runnable end-to-end and to save standardized artifacts (metrics, predictions, plots, checkpoints, and analysis tables) for fair comparisons between baselines and extensions.

The notebooks are organized by project phase:
- `P0_*`: data collection and dataset understanding
- `P1_*`: preprocessing and generation of model-agnostic processed datasets
- `P2_*`: baseline models (standardized training/evaluation protocol)
- `P3_*`: extensions (two higher-level improvements beyond the original paper)

## Directory Structure

### `notebooks/P0_*` — Data
Data-focused notebooks used to understand the dataset and verify assumptions.
Typical outputs include dataset summaries and lightweight plots used to guide later experimentation.

Examples:
- `P0_01_data_collection.ipynb`
- `P0_02_data_exploration.ipynb`

### `notebooks/P1_*` — Preprocessing
Preprocessing notebook(s) that generate the canonical `data/processed/` structure used by all models.
This phase produces:
- canonical CSVs with consistent columns
- training settings indices (`index_settings.csv`)
- test set indices (`index_testsets.csv`)
- optional split metadata and audit reports

Example:
- `P1_01_data_preprocessing.ipynb`

### `notebooks/P2_*` — Baselines
Baseline notebooks that train and evaluate models under the same protocol:
- in-domain, cross-domain (sentiment), FULL, and cross-variety matrix
- standardized metric computation and saving
- standardized prediction export for later error analysis
- paper-style plotting at the end of the notebook

Examples:
- `P2_01_baseline_bert.ipynb`
- `P2_02_baseline_roberta.ipynb`
- `P2_03_baseline_mistral.ipynb`

### `notebooks/P3_*` — Extensions
Extension notebooks implementing two higher-level improvements beyond the original paper.
Each extension notebook:
- reuses the same processed datasets and index files as baselines
- saves artifacts using the same folder structure and naming conventions
- includes comparisons against the strongest baseline and a brief error analysis

Examples:
- `P3_01_extension_roberta_mixture_of_adapters.ipynb`
- `P3_02_extension_mistral_variety_classifier.ipynb` (prerequisite component)
- `P3_03_extension_mistral_hyperbolic_toc.ipynb`

## Outputs (standard across notebooks)
Each experiment notebook is expected to write the following artifacts per run:
- `metrics/` — CSV metrics tables (overall + breakdowns + pivots)
- `predictions/` — CSV predictions per test set (including probabilities where available)
- `plots/` — saved figures (paper-style heatmaps and grouped bar charts)
- `checkpoints/` — trained model weights
- `analysis/` — error analysis tables (wins/regressions, representative examples)

## Reproducibility
All notebooks assume:
- processed inputs come from `data/processed/` and are model-agnostic
- training and test sets are resolved via:
  - `data/processed/<task>/index_settings.csv`
  - `data/processed/<task>/testsets/index_testsets.csv`
- fixed seeds and logged configuration parameters per run
- consistent metric computation and plotting style across baselines and extensions
