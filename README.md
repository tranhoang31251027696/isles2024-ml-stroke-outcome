# ISLES 2024 Stroke Outcome Prediction — V3

Reproducible analysis code and publication-ready aggregate results for predicting 90-day functional outcome after mechanical thrombectomy using pre-procedural clinical variables from ISLES 2024.

The primary analysis uses a fixed `GradientBoostingClassifier` with 150 trees, maximum depth 2, learning rate 0.05, subsample 0.8, and no class weighting. The configuration was retained after exploratory work on this cohort; its repeated cross-validation estimates are therefore not selection-adjusted. The repository does not claim confirmed superiority over equally optimized algorithms or clinical readiness.

## What is included

- `src/`: the numerical V3 pipeline and verification scripts.
- `results/figures/`: six manuscript figures and one supplementary figure.
- `results/tables/`: aggregate manuscript and supplementary result tables, plus revision notes.
- `data/README.md`: instructions for obtaining the public source data.

This public package deliberately excludes patient-level clinical tables, raw workbooks, out-of-fold patient predictions, fold/checkpoint files, trained model artefacts, development caches, and manuscript submission files.

## Data access

The ISLES 2024 training data are publicly available through Zenodo version 7: [10.5281/zenodo.17652035](https://doi.org/10.5281/zenodo.17652035). Download the permitted source data yourself and place them under `data/source_v7/` as described in [data/README.md](data/README.md). The data are not redistributed here.

## Environment and analysis

The analysis was developed with Python and scikit-learn 1.7.2. Create an isolated environment and install the pinned packages:

```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows PowerShell
pip install -r requirements.txt
```

After placing the source data in `data/source_v7/`, run the numerical workflow from the repository root:

```bash
python src/run_all.py
```

The run reconstructs the release table, applies fold-specific preprocessing, evaluates the fixed primary GBM and matched comparators, generates aggregate figures and tables, and verifies saved results. It can take substantial time because it uses repeated nested cross-validation.

`src/stage_ac_revision_checks.py` reproduces the explicitly labelled post-review auxiliary checks. Those checks do not replace the primary model or select a new model.

## Interpretation and limitations

The primary estimate is the mean of five within-seed out-of-fold AUCs. Patient-mean ensemble AUC is a distinct secondary estimand used for ROC, calibration and probability-based clinical comparisons. Bootstrap intervals condition on the retained configuration and fitted predictions. The cohort is small, covariate missingness is substantial and centre-specific, and no independent external validation was performed.

See `results/tables/V3_revision_notes.txt` for the reconciled reviewer-response corrections and current auxiliary diagnostics.

## Citation

If you use this code, cite the accompanying manuscript and the ISLES 2024 data release. A permanent code archive DOI will be added here after the public repository release is finalized.
