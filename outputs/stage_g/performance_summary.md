# Branch 2 — Performance Summary

Generated: 2026-08-25T03:32  |  Cohort n=129  |  Poor=44  Good=85

## Primary Model: GBM + SelectKBest  (5×10-fold, poor=1)

| Metric | Value |
|---|---|
| Mean OOF AUC (Stage H) | **0.7495 ± 0.0122** |
| Bootstrap 95% CI       | [0.682 – 0.858] |
| Accuracy               | 74.26% |
| Brier Score (poor=1)   | 0.1872 |
| Calibration slope      | 0.5274 (< 1 = overconfident) |
| ΔAUC vs THRIVE         | +0.154 [0.041–0.271] p=0.0060 |
| NRI vs THRIVE          | 0.9166 [0.5933–1.2528] |
| IDI vs THRIVE          | 0.2825 [0.1794–0.3931] |

## Literature Benchmark

| Model | Mean AUC | AUC Std | Acc (%) | F1 |
|---|---|---|---|---|
| [OUR FLAGSHIP] GBM + Feature Selection (Repeated 10-Fold CV) ⬅ **OUR MODEL** | 0.7495 | 0.0122 | 74.26 | 0.8126 |
| [Otieno 2024] XGBoost (n=100 d=6 lr=0.1, MinMax) | 0.7388 | 0.0199 | 71.63 | 0.7927 |
| [Bogey 2025] GBM (n=100 d=3 lr=0.1, StdScaler) | 0.7368 | 0.0169 | 72.71 | 0.8005 |
| [Liu 2026] XGBoost (n=100 d=4 lr=0.05, StdScaler) | 0.7312 | 0.0086 | 72.25 | 0.7986 |
| [Bogey 2025] RF  (n=100 d=5, StdScaler) | 0.7261 | 0.0055 | 72.87 | 0.7974 |
| [Li 2024]   Nomogram LR (ElasticNet, StdScaler) | 0.7079 | 0.0152 | 71.47 | 0.784 |
| [Otieno 2024] SVM  (RBF C=10 g=0.01, MinMax) | 0.7019 | 0.0201 | 72.25 | 0.8145 |
| [Otieno 2024] mLR  (L2 C=1, MinMax) | 0.7 | 0.0146 | 68.53 | 0.7509 |
| [Otieno 2024] ANN  (100-50 relu, MinMax) | 0.6289 | 0.0197 | 66.2 | 0.7475 |