# Figurative Language Understanding via Mixture-of-Adapters and Tensor-of-Cues

## Project overview
This repository contains our end-to-end experimental pipeline for the Deep Natural Language Processing course project on Figurative Language Understanding, using a BESSTIE-style dataset. We replicate paper-aligned baselines with a standardized protocol and implement two extensions. All notebooks are designed to be runnable end-to-end and to export comparable artifacts (metrics, predictions, paper-style plots, and error analysis) so that baselines and extensions can be compared fairly.

## Team
- Ashkan Shafiei (Politecnico di Torino) — s342583@studenti.polito.it  
- Amir Masoud Almasi (Politecnico di Torino) — s337006@studenti.polito.it  
- Mehdi Nickzamir (Politecnico di Torino) — s323959@studenti.polito.it  
- Balzhan Dosmukhametova (Politecnico di Torino) — s343931@studenti.polito.it  

## Tasks and evaluation protocol
Each example contains at least:
- `text`
- `label`
- `task` (sentiment or sarcasm)
- `variety` (en-AU, en-IN, en-UK)
- `source/domain` (Google vs Reddit)

We evaluate both tasks and report:
- In-domain (train Google → test Google; train Reddit → test Reddit)
- Cross-domain (train Google → test Reddit; train Reddit → test Google)  
  Applies to sentiment. Sarcasm follows the benchmark protocol (commonly Reddit-only).
- FULL (train on combined pool; test on full validation)
- Cross-variety matrix (train on one variety → test on each variety)

All results are reported with:
- macro F1, accuracy, precision, recall
- per-variety breakdowns (and per-source breakdowns where applicable)
- saved predictions for error analysis
- paper-style figures (annotated heatmaps and grouped bar charts) exported to disk

## Repository structure
- `data/`  
  Raw dataset files and standardized processed outputs used by all notebooks.
- `src/`  
  Reusable utilities used across the pipeline (preprocessing helpers, loaders, metrics, plotting).
- `notebooks/`  
  Phase-based notebooks (P0–P3) that run the full pipeline.
- `models/`  
  Run artifacts produced by experiments (metrics, predictions, plots, analysis; checkpoints may be excluded from Git).
- `docs/`  
  Reference paper (PDF).

## Notebook pipeline (phase-based)
Run notebooks in this order:

### Phase P0 — Data
- `P0_01_data_collection.ipynb`  
  Dataset setup and expected raw file placement.
- `P0_02_data_exploration.ipynb`  
  Exploratory analysis (distributions by task/variety/source, imbalance checks, length statistics).

### Phase P1 — Preprocessing
- `P1_01_data_preprocessing.ipynb`  
  Builds the canonical `data/processed/` structure and creates index files for standardized training/evaluation.

Outputs include:
- model-agnostic processed CSVs (consistent columns)
- `index_settings.csv` (training settings) and `testsets/index_testsets.csv` (evaluation test sets)
- optional audit/report tables under `data/reports/`

### Phase P2 — Baselines
- `P2_01_baseline_bert.ipynb`  
  BERT baseline runs using the standardized protocol and processed indices.
- `P2_02_baseline_roberta.ipynb`  
  RoBERTa baseline runs (strong encoder baseline; main comparison point for extensions).
- `P2_03_baseline_mistral.ipynb`  
  Mistral baseline runs (decoder-only model), writing outputs to a dedicated run directory structure.

### Phase P3 — Extensions
- `P3_01_extension_roberta_mixture_of_adapters.ipynb`  
  Extension 1: RoBERTa Mixture of Adapters, compared directly against the RoBERTa baseline with standardized reporting and error analysis.
- `P3_02_extension_mistral_variety_classifier.ipynb`  
  Prerequisite for the Mistral extension: trains an XLM-R variety classifier (en-AU/en-IN/en-UK) used downstream.
- `P3_03_extension_mistral_hyperbolic_toc.ipynb`  
  Extension 2: Mistral Hyperbolic-ToC fine-tuning, using the prerequisite classifier and compared against the Mistral baseline.

## Outputs and artifacts
Across baselines and extensions, each run is expected to export:
- `metrics/`  
  CSV metrics tables (overall + breakdowns + pivot tables used for plots)
- `predictions/`  
  CSV predictions per test set (including probabilities when available)
- `plots/` (or `figures/`)  
  Paper-style grouped bar charts and annotated heatmaps saved as PNG
- `analysis/`  
  Error analysis tables (wins/regressions; representative examples)
- `checkpoints/` (optional)  
  Model weights (may be excluded from Git due to size)

## Running the project
Notebooks assume a configurable project root (commonly used in Colab):
- `BASE = /content/drive/MyDrive/DNLP`

If running locally or using a different Drive location, update the `BASE` path in the configuration cell near the top of each notebook. The preprocessing notebook creates required directories if they do not exist.

Suggested run sequence:
1. Place raw files in `data/raw/` as expected by the preprocessing notebook.
2. Run `P1_01_data_preprocessing.ipynb` to generate `data/processed/` and indices.
3. Run baselines in `P2_*`.
4. Run extensions in `P3_*`.
5. Use saved metrics and exported plots for the final report.

## Reproducibility
- The pipeline is index-driven: training and testing sets are resolved via processed index files created in preprocessing.
- Seeds and run configurations are defined in each notebook’s config cell.
- Comparisons are fair by construction: all baselines and extensions rely on the same processed inputs and standardized reporting.
