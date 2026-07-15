# Decisions

## 2026-07-15

- **Initial setup**: TF-IDF + LogisticRegression pipeline on AG News dataset
- **Config**: Single config.py with paths and label names
- **Training**: `--force` flag to retrain; skip if model exists
- **Prediction**: Interactive CLI with per-category scores