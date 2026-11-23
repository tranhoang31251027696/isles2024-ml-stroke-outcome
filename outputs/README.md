# Public pipeline outputs

This directory mirrors the V3 analytical stages with publication-relevant outputs:

- cross-validation fold assignments (`stage_a_data/fold_manifest.csv`);
- label-free out-of-fold predicted probabilities across models and seeds (`stage_b_training/primary_oof_predictions.csv`, `stage_c_evaluation/patient_mean_predictions.csv`, `stage_ac_revision_checks/oof_predictions.csv`);
- stage protocols and completion records;
- aggregate performance, calibration, decision-curve, sensitivity, centre, and literature-comparator tables; and
- generated high-resolution figures.

To prevent duplicate hosting of patient-level clinical records, original clinical predictors (e.g., age, NIHSS, laboratory values) and ground-truth 90-day mRS outcome labels are not redistributed. Readers can download the source ISLES 2024 v7 data from Zenodo (CC BY-NC-SA 4.0) into `data/source_v7/` and evaluate the saved predictions directly or retrain the full pipeline.
