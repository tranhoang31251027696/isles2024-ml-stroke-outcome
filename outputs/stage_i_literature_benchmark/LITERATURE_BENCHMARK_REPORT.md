# Literature benchmark

Eight fixed literature configurations were refitted on 129 patients with the same 22 predictors and 5 seeds × 10 outer folds used for the primary GBM (400 fits). Official v7 source missingness was used; binary variables were mode-imputed, other variables median-imputed, and scaling was fitted within training folds. Both XGBoost models used XGBClassifier.

| Rank by AUC point estimate | Model | Mean seed OOF AUC ± SD | 95% CI for mean seed AUC |
|---:|---|---:|---|
| 1 | Fixed GBM (this study) | 0.7599 ± 0.0184 | 0.6729–0.8394 |
| 2 | Bogey et al. - RF (d=5) | 0.7438 ± 0.0054 | 0.6498–0.8295 |
| 3 | Bogey et al. - GBM (d=3) | 0.7434 ± 0.0162 | 0.6522–0.8288 |
| 4 | Otieno et al. - XGBoost (d=6) | 0.7315 ± 0.0172 | 0.6403–0.8181 |
| 5 | Liu et al. - XGBoost (d=4) | 0.7298 ± 0.0110 | 0.6360–0.8189 |
| 6 | Li et al. - ElasticNet LR | 0.7144 ± 0.0128 | 0.6130–0.8111 |
| 7 | Otieno et al. - SVM RBF | 0.7110 ± 0.0114 | 0.6097–0.8104 |
| 8 | Otieno et al. - mLR (L2) | 0.7080 ± 0.0127 | 0.6049–0.8088 |
| 9 | Otieno et al. - ANN (100-50) | 0.6278 ± 0.0111 | 0.5278–0.7263 |

Literature configurations were held fixed. The primary GBM was retained after exploratory work on this cohort, whereas the other seven families used inner CV. This auxiliary benchmark does not establish algorithm superiority or equal tuning.

Ranking is based on point estimates and does not itself establish a statistical difference. CIs and p values use a conditional paired patient bootstrap with 2,000 replicates; p values are Holm-adjusted across eight comparisons with GBM.

These are reimplementations on the present cohort, not the published performance of the source studies and not external validation.
