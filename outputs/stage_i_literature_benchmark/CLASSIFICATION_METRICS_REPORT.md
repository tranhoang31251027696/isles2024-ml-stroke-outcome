# Classification metrics — literature benchmark V3

Positive class: poor outcome (90-day mRS 3–6). Threshold fixed at 0.5. Metrics are calculated on 129 OOF predictions per seed and averaged over five seeds. No threshold tuning.

| Model | Precision (%) | Recall (%) | F1 (%) | Specificity (%) | Accuracy (%) |
|---|---:|---:|---:|---:|---:|
| Fixed Cand4 GBM (this study) | 62.7 | 48.6 | 54.7 | 84.9 | 72.6 |
| Bogey et al. - RF (d=5) | 67.7 | 42.3 | 52.0 | 89.4 | 73.3 |
| Bogey et al. - GBM (d=3) | 61.5 | 51.4 | 55.9 | 83.3 | 72.4 |
| Otieno et al. - XGBoost (d=6) | 61.3 | 46.8 | 53.0 | 84.7 | 71.8 |
| Liu et al. - XGBoost (d=4) | 64.6 | 45.5 | 53.3 | 87.1 | 72.9 |
| Li et al. - ElasticNet LR | 72.4 | 34.5 | 46.8 | 93.2 | 73.2 |
| Otieno et al. - SVM RBF | 70.5 | 40.5 | 51.3 | 91.3 | 74.0 |
| Otieno et al. - mLR (L2) | 66.7 | 45.5 | 54.0 | 88.2 | 73.6 |
| Otieno et al. - ANN (100-50) | 50.8 | 45.0 | 47.7 | 77.4 | 66.4 |

These metrics describe a fixed threshold; their ordering may differ from AUC. They are not metrics for favorable outcome as the positive class.
