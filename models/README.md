# Models Directory

This directory stores experiment artifacts produced by training and evaluation notebooks (baselines and extensions). It is used as the central location for run outputs such as metrics tables, predictions, plots, and (optionally) model checkpoints.

The repository follows a separation between:
- run artifacts (recommended to keep): metrics, predictions, plots, analysis tables
- checkpoints/weights (large files): typically not tracked in Git due to size

## Directory Structure

### `models/` (general convention across runs)
Across baselines and extensions, each run directory is expected to contain some or all of:
- `metrics/` CSV tables (overall + breakdowns + pivots)
- `predictions/` CSV predictions per test set (including probabilities when available)
- `figures/` exported figures (paper-style heatmaps and grouped bar charts)
- `analysis/` error analysis tables (wins/regressions, representative examples)
- `checkpoints/` trained weights (often excluded from Git due to volume)

## Notes
- Metrics, predictions, and plots are sufficient to reproduce tables and figures.
- In the public GitHub repository, checkpoint folders may exist but remain empty to keep the repo lightweight.
