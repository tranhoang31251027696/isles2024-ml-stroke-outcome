# Independent V3 Cand4 audit

Cand4 was independently refitted on all 50 outer training sets using the original 22-variable preprocessing. Inputs and missingness match V2; source XLSX CRCs were checked. No active checkpoint, scientific source configuration, manuscript, or V2 output was modified.

                model      auc  precision   recall       f1  specificity  accuracy    brier   auc_sd  auc_ci_low  auc_ci_high  ensemble_auc  ensemble_ci_low  ensemble_ci_high  ensemble_brier
            GBM_Cand4 0.759920   0.626513 0.486364 0.546910     0.849412  0.725581 0.188235 0.018426    0.672890     0.839419      0.780749         0.693386          0.861710        0.182269
           V2_SVM_RBF 0.750241   0.622941 0.472727 0.537368     0.851765  0.722481 0.187383 0.022399    0.663026     0.836199      0.762032         0.671592          0.851846        0.184421
      V2_RandomForest 0.744706   0.628730 0.368182 0.463688     0.887059  0.710078 0.186681 0.007205    0.655288     0.830529      0.754545         0.660690          0.843744        0.184145
               V2_GBM 0.732032   0.606045 0.509091 0.552943     0.828235  0.719380 0.208548 0.023232    0.646682     0.812014      0.752406         0.660327          0.836932        0.195788
           V2_XGBoost 0.707594   0.590311 0.477273 0.527571     0.828235  0.708527 0.200447 0.011589    0.618375     0.794982      0.721925         0.623357          0.814063        0.192870
               V2_MLP 0.701979   0.597341 0.540909 0.566452     0.809412  0.717829 0.211962 0.011043    0.600616     0.795257      0.727273         0.623052          0.825069        0.196493
        V2_SVM_Linear 0.697299   0.692019 0.395455 0.502726     0.908235  0.733333 0.199582 0.015352    0.598160     0.794958      0.700267         0.599473          0.801159        0.196756
V2_ElasticNetLogistic 0.687861   0.562742 0.490909 0.523124     0.802353  0.696124 0.213680 0.018725    0.596209     0.781804      0.700535         0.598209          0.802603        0.201151
        V2_L2Logistic 0.628824   0.578409 0.231818 0.329773     0.908235  0.677519 0.222755 0.017007    0.551188     0.706142      0.647059         0.544090          0.751864        0.213845

           comparator  delta_auc    ci_low  ci_high    p_two  p_holm_8
               V2_GBM   0.027888  0.012538 0.044173 0.001000  0.007996
      V2_RandomForest   0.015214 -0.026321 0.060159 0.488756  0.977511
           V2_XGBoost   0.052326  0.014351 0.092456 0.008996  0.053973
        V2_L2Logistic   0.131096  0.056978 0.204030 0.001000  0.007996
V2_ElasticNetLogistic   0.072059 -0.007839 0.150507 0.074963  0.374813
           V2_SVM_RBF   0.009679 -0.055338 0.069825 0.756622  0.977511
        V2_SVM_Linear   0.062620 -0.019183 0.144144 0.138931  0.555722
               V2_MLP   0.057941 -0.021887 0.141803 0.157921  0.555722

## Findings

- major: Cand4 is fixed, not a 12-candidate inner-selected GBM; reporting all eight families as equally tuned or GBM selected from 12 candidates is inaccurate.

- major: The claimed 0.7599/0.7807 values are reproducible but already present in a multi-variant V2 outcome comparison. Calling this choice pre-specified is not supported by that analysis history; selecting it using this cohort creates selection optimism not removed by merely fixing it in V3.

- major: Conditional bootstrap p-values/intervals assume these predictions/configuration are fixed and do not correct the historical configuration search or quantify full training uncertainty. p=0.001 must not be presented as selection-adjusted confirmation.

- moderate: The manuscript builder changes ranking claims but retains the obsolete 12-candidate GBM methods, equal-tuning figure caption and hard-coded SHAP top features. It is not synchronized with fixed Cand4.

- moderate: The variant comparison mixes mean-seed AUC with ensemble-threshold accuracy/recall/F1/Brier. These are valid separate estimands but must be labeled; use the audit mean-seed classification metrics for like-for-like comparison.

- moderate: Legacy experimental scripts include unused HGB variants and some baseline labels now reference V3 ROOT rather than V2; copying these scripts does not preserve their original V2 baseline interpretation.

- status: Active V3 training output is being written during audit; this audit does not certify that the entire downstream figure/DCA/sensitivity/manuscript pipeline has finished or is synchronized.

## Practical interpretation

The numerical improvement is reproducible, with original V2 input missingness/preprocessing unchanged. It is a fixed-Cand4 result, not a new nested-search result. Mean-seed precision/recall/F1 are 0.626513 / 0.486364 / 0.546910; mean-seed Brier is 0.188235. The earlier variant table instead used patient-mean ensemble classification metrics and Brier 0.182269. Do not mix these columns.

The 0.7599 point estimate is higher than SVM 0.7502 and RF 0.7447, but paired intervals versus SVM and RF cross zero. Fixing a shallow tree structure can regularize the learner, but does not eliminate overfitting or retroactively pre-specify the cohort-informed choice. Results can be reported transparently as an exploratory fixed-configuration analysis; confirmatory superiority is not established.

Recommended edits before generating the V3 manuscript: describe GBM as fixed Cand4 with one candidate and the others as 12-candidate searches; remove claims of eight equally tuned models; disclose how Cand4 was chosen; remove the claim that depth 2 eliminates overfitting; distinguish conditional p-values from selection-adjusted evidence; regenerate all downstream results before treating the full output set as synchronized. The manuscript verifier currently checks many numeric entries but does not catch these methodological wording inconsistencies.
