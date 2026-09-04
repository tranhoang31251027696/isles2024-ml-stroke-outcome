import os
import json
import pandas as pd
import numpy as np


import pathlib as _pathlib
_HERE = _pathlib.Path(__file__).resolve().parent          # src/
BASE_DIR = str(_HERE.parent)                               # project root (FINAL/)

def run_stage_b():
    tabular_path = os.path.join(BASE_DIR, "processed", "tabular", "clinical_features.csv")
    feature_names_path = os.path.join(BASE_DIR, "processed", "tabular", "feature_names.txt")
    labels_path = os.path.join(BASE_DIR, "processed", "labels", "outcome_labels.csv")
    output_dir = os.path.join(BASE_DIR, "outputs", "stage_b")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== STAGE B: COHORT SELECTION & PREPROCESSING ===")
    
    df_tabular = pd.read_csv(tabular_path)
    df_labels = pd.read_csv(labels_path)
    
    with open(feature_names_path, 'r', encoding='utf-8') as f:
        feature_names = [line.strip() for line in f if line.strip()]
        
    if 'subject_id' not in df_tabular.columns:
        df_tabular['subject_id'] = df_labels['subject_id']
        
    df_merged = pd.merge(df_tabular, df_labels, on='subject_id', how='left')
    
    # 1. Filter cohort (n=129 valid cases with non-null mRS_3_months_bin)
    valid_mask = df_merged['mRS_3_months_bin'].notna()
    df_cohort = df_merged[valid_mask].copy().reset_index(drop=True)
    df_excluded = df_merged[~valid_mask].copy().reset_index(drop=True)
    
    print(f"Total Cohort Size: {len(df_cohort)} (Excluded: {len(df_excluded)})")
    
    # 2. Check Selection Bias (Demographic comparison: Age, NIHSS, Sex)
    bias_check = {}
    for col in ['Age', 'NIHSS at admission', 'Sex']:
        inc_mean = float(df_cohort[col].mean())
        exc_mean = float(df_excluded[col].mean())
        bias_check[col] = {
            'included_mean': round(inc_mean, 4),
            'excluded_mean': round(exc_mean, 4),
            'diff': round(abs(inc_mean - exc_mean), 4)
        }
    print("Selection Bias Check (Included vs Excluded):", bias_check)
    
    # 3. Explicitly define Predictor Sets based ONLY on feature_names.txt
    leakage_cols = [
        'Door to groin_min',
        'Door to first series_min',
        'Door to recanalization_min',
        'Time of intervention_min'
    ]
    
    all_predictors = [f for f in feature_names if f in df_cohort.columns]
    baseline_predictors = [f for f in all_predictors if f not in leakage_cols]
    
    print(f"Total Predictors in feature_names.txt: {len(all_predictors)}")
    print(f"Baseline Predictors (strictly pre-treatment, 22 features): {len(baseline_predictors)}")
    print(f"Intra-procedural Leakage Risk Features (excluded from baseline): {leakage_cols}")
    
    # 4. Save Cleaned Cohort Dataset
    df_cohort.to_csv(os.path.join(output_dir, "cleaned_cohort.csv"), index=False)
    
    metadata = {
        'n_total': len(df_merged),
        'n_included': len(df_cohort),
        'n_excluded': len(df_excluded),
        'target_binary': 'mRS_3_months_bin',
        'class_distribution': df_cohort['mRS_3_months_bin'].value_counts().to_dict(),
        'all_predictors': all_predictors,
        'baseline_predictors': baseline_predictors,
        'leakage_predictors': leakage_cols,
        'selection_bias_check': bias_check
    }
    
    with open(os.path.join(output_dir, "cohort_metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Stage B complete! Cohort saved to: {output_dir}")

if __name__ == "__main__":
    run_stage_b()
