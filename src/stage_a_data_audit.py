import os
import json
import pandas as pd
import numpy as np


import pathlib as _pathlib
_HERE = _pathlib.Path(__file__).resolve().parent          # src/
BASE_DIR = str(_HERE.parent)                               # project root (FINAL/)

def run_stage_a_audit():
    tabular_path = os.path.join(BASE_DIR, "processed", "tabular", "clinical_features.csv")
    feature_names_path = os.path.join(BASE_DIR, "processed", "tabular", "feature_names.txt")
    labels_path = os.path.join(BASE_DIR, "processed", "labels", "outcome_labels.csv")
    output_dir = os.path.join(BASE_DIR, "outputs", "stage_a")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== STAGE A: DATA AUDIT & LEAKAGE CHECK ===")
    
    # 1. Load data
    df_tabular = pd.read_csv(tabular_path)
    df_labels = pd.read_csv(labels_path)
    
    with open(feature_names_path, 'r', encoding='utf-8') as f:
        feature_names = [line.strip() for line in f if line.strip()]
        
    print(f"Total rows in clinical_features.csv: {len(df_tabular)}")
    print(f"Total rows in outcome_labels.csv: {len(df_labels)}")
    print(f"Total features listed in feature_names.txt: {len(feature_names)}")
    
    # Check ID column in df_tabular
    if 'subject_id' not in df_tabular.columns:
        print("Note: 'subject_id' missing in clinical_features.csv header, assigning subject_id from outcome_labels.csv")
        df_tabular['subject_id'] = df_labels['subject_id']
        
    # Merge on subject_id
    df_merged = pd.merge(df_tabular, df_labels, on='subject_id', how='left')
    
    # 2. Check Target Label Distribution (mRS_3_months_bin)
    target_col = 'mRS_3_months_bin'
    valid_mask = df_merged[target_col].notna()
    n_total = len(df_merged)
    n_valid = int(valid_mask.sum())
    n_missing = n_total - n_valid
    
    y_valid = df_merged.loc[valid_mask, target_col].astype(int)
    class_counts = y_valid.value_counts().to_dict()
    
    print("\n--- Target Label Audit (mRS_3_months_bin) ---")
    print(f"Total Subjects: {n_total}")
    print(f"Valid Outcome Cases: {n_valid} ({n_valid/n_total*100:.1f}%)")
    print(f"Missing Outcome Cases: {n_missing} ({n_missing/n_total*100:.1f}%)")
    print(f"Class Distribution: {class_counts} (1: Good outcome [0-2], 0: Poor outcome [3-6])")
    print(f"Imbalance Ratio: {class_counts.get(1, 0) / max(class_counts.get(0, 1), 1):.2f}:1")
    
    # 3. Categorize Features & Audit Missingness
    label_cols = list(df_labels.columns)
    predictor_cols = [c for c in df_tabular.columns if c not in label_cols and c != 'subject_id']
    
    leakage_candidates = [
        'Door to recanalization_min',
        'Door to groin_min',
        'Door to first series_min',
        'Time of intervention_min'
    ]
    
    feature_stats = []
    for col in predictor_cols:
        col_data = df_tabular[col]
        n_miss = int(col_data.isna().sum())
        pct_miss = (n_miss / len(col_data)) * 100
        dtype = str(col_data.dtype)
        min_val = float(col_data.min()) if pd.api.types.is_numeric_dtype(col_data) else None
        max_val = float(col_data.max()) if pd.api.types.is_numeric_dtype(col_data) else None
        mean_val = float(col_data.mean()) if pd.api.types.is_numeric_dtype(col_data) else None
        std_val = float(col_data.std()) if pd.api.types.is_numeric_dtype(col_data) else None
        
        is_leakage = col in leakage_candidates
        
        feature_stats.append({
            'feature': col,
            'dtype': dtype,
            'missing_count': n_miss,
            'missing_pct': round(pct_miss, 2),
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val,
            'is_leakage_risk': is_leakage
        })
        
    df_feature_stats = pd.DataFrame(feature_stats)
    df_feature_stats.to_csv(os.path.join(output_dir, "feature_audit.csv"), index=False)
    
    # 4. Generate JSON Summary Report
    audit_summary = {
        'total_samples': n_total,
        'valid_samples': n_valid,
        'missing_samples': n_missing,
        'target_distribution': {str(k): int(v) for k, v in class_counts.items()},
        'total_predictors': len(predictor_cols),
        'predictors': predictor_cols,
        'leakage_risk_predictors': leakage_candidates,
        'high_missing_predictors': df_feature_stats[df_feature_stats['missing_pct'] > 10]['feature'].tolist()
    }
    
    with open(os.path.join(output_dir, "stage_a_audit_report.json"), 'w', encoding='utf-8') as f:
        json.dump(audit_summary, f, indent=2)
        
    print("\n--- Feature Audit Summary ---")
    print(df_feature_stats[['feature', 'missing_pct', 'mean', 'std', 'is_leakage_risk']].to_string())
    print(f"\nAudit complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    run_stage_a_audit()
