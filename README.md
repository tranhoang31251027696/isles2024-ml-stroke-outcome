# Machine Learning Pipeline for ISLES 2024 Stroke Outcome Prediction

This repository contains the complete, reproducible pipeline for predicting functional outcome after mechanical thrombectomy using the ISLES 2024 dataset.

## Setup
Install the required packages using pip:
```bash
pip install -r requirements.txt
```
**Note:** The pipeline was developed and tested using `scikit-learn` version 1.7.2.

## Pipeline Structure
The pipeline is designed to be executed sequentially from Stage A to Stage H. All outputs are saved to the `outputs/` directory.

- **`src/stage_a_data_audit.py`**: Audits the raw ISLES 2024 clinical features and generates a report on missing values and feature exclusion criteria.
- **`src/stage_b_preprocessing.py`**: Cleans and preprocesses the dataset, generating the final locked cohort (`cleaned_cohort.csv`).
- **`src/stage_c_benchmarks.py`**: Calculates reference AUC scores for clinical benchmarks (THRIVE, SPAN-100, univariate NIHSS).
- **`src/stage_d1_model_search.py`**: Performs systematic cross-validated hyperparameter search for various baseline algorithms.
- **`src/stage_d2_gbm_finetune.py`**: Fine-tunes the flagship Gradient Boosting Machine (GBM) model and generates out-of-fold (OOF) predictions (`oof_predictions.csv`).
- **`src/stage_e_evaluation.py`**: Contains the canonical locked values for the final model performance, NRI, IDI, and Decision Curve Analysis (DCA).
- **`src/stage_f_shap_sensitivity.py`**: Calculates SHAP feature importance for interpretability.
- **`src/stage_g_ledger.py`**: Consolidates all metrics into a final JSON ledger.
- **`src/stage_h_paper_comparison.py`**: Compares the flagship GBM against re-implemented literature models.

## Pre-execution Check
You can verify the integrity of the locked output artifacts using the provided check script:
```bash
python src/gate_check.py
```
