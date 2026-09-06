"""
Stage C — Canonical Baseline & Benchmark Scores Computation
Uses canonical formulas from branch2_correction_utils for THRIVE and SPAN-100.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import pathlib as _pathlib
_HERE = _pathlib.Path(__file__).resolve().parent          # src/
BASE_DIR = str(_HERE.parent)                               # project root (FINAL/)
PROJECT_ROOT = str(_HERE.parent.parent)                    # parent root (processed_ISLES2024/nhanh_2)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from branch2_correction_utils import compute_canonical_thrive, compute_canonical_span100

def run_stage_c():
    cohort_path = os.path.join(BASE_DIR, "outputs", "stage_b", "cleaned_cohort.csv")
    output_dir = os.path.join(BASE_DIR, "outputs", "stage_c")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== STAGE C: BASELINE & BENCHMARK SCORES (CANONICAL FORMULAS) ===")
    
    df = pd.read_csv(cohort_path)
    
    # 1. Un-scale Age (mean = 70.4564, std = 15.0285)
    age_mean = 70.4564
    age_std = 15.0285
    df['Age_raw'] = df['Age'] * age_std + age_mean
    
    # NIHSS is already raw points
    df['NIHSS_raw'] = df['NIHSS at admission']
    
    # 2. Calculate Canonical THRIVE Score (0-9 scale, Flint 2013)
    df['THRIVE_score'] = compute_canonical_thrive(
        df['Age_raw'].values,
        df['NIHSS_raw'].values,
        df['Hypertension'].values,
        df['Diabetes'].values,
        df['Atrial fibrillation'].values
    )
    
    # 3. Calculate Canonical SPAN-100 Score
    df['SPAN100_score'], df['SPAN100_pos'] = compute_canonical_span100(
        df['Age_raw'].values,
        df['NIHSS_raw'].values
    )
    
    # Save benchmark scores dataframe with subject_id
    df_scores = df[['subject_id', 'mRS_3_months_bin', 'Age_raw', 'NIHSS_raw', 'THRIVE_score', 'SPAN100_score', 'SPAN100_pos']].copy()
    df_scores.to_csv(os.path.join(output_dir, "benchmark_scores.csv"), index=False)
    
    # Outcome: poor outcome = 1 (mRS 3-6)
    y_poor = 1 - df['mRS_3_months_bin'].astype(int).values
    
    # 4. Global AUCs
    auc_thrive_global = float(roc_auc_score(y_poor, df['THRIVE_score'].values))
    auc_span_global   = float(roc_auc_score(y_poor, df['SPAN100_score'].values))
    
    # 5. Evaluate Benchmarks via 10-fold Stratified CV across 5 random seeds
    seeds = [42, 100, 2024, 777, 999]
    benchmark_results = {
        'THRIVE': [],
        'SPAN100_raw': [],
        'Univariate_NIHSS': []
    }
    
    for seed in seeds:
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        
        oof_thrive = np.zeros(len(y_poor))
        oof_span   = np.zeros(len(y_poor))
        oof_nihss  = np.zeros(len(y_poor))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, y_poor)):
            oof_thrive[val_idx] = df.loc[val_idx, 'THRIVE_score'].values
            oof_span[val_idx]   = df.loc[val_idx, 'SPAN100_score'].values
            
            nihss_train = df.loc[train_idx, 'NIHSS_raw'].values.reshape(-1, 1)
            nihss_val   = df.loc[val_idx, 'NIHSS_raw'].values.reshape(-1, 1)
            
            lr_nihss = LogisticRegression(C=1.0)
            lr_nihss.fit(nihss_train, y_poor[train_idx])
            oof_nihss[val_idx] = lr_nihss.predict_proba(nihss_val)[:, 1]
            
        benchmark_results['THRIVE'].append(roc_auc_score(y_poor, oof_thrive))
        benchmark_results['SPAN100_raw'].append(roc_auc_score(y_poor, oof_span))
        benchmark_results['Univariate_NIHSS'].append(roc_auc_score(y_poor, oof_nihss))
            
    summary_metrics = {
        'Global_AUC': {
            'THRIVE': round(auc_thrive_global, 4),
            'SPAN100': round(auc_span_global, 4)
        }
    }
    for model_name, aucs in benchmark_results.items():
        summary_metrics[model_name] = {
            'mean_auc': round(float(np.mean(aucs)), 4),
            'std_auc':  round(float(np.std(aucs)), 4),
            'min_auc':  round(float(np.min(aucs)), 4),
            'max_auc':  round(float(np.max(aucs)), 4)
        }
        
    print("\n--- Canonical Benchmark Performance Summary ---")
    print(f"  THRIVE Global AUC:   {auc_thrive_global:.4f}")
    print(f"  SPAN-100 Global AUC: {auc_span_global:.4f}")
    for k, v in summary_metrics.items():
        if k != 'Global_AUC':
            print(f"  {k:20s}: CV Mean AUC = {v['mean_auc']:.4f} ± {v['std_auc']:.4f}")
        
    summary_path = os.path.join(output_dir, "benchmark_metrics_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_metrics, f, indent=2)
        
    print(f"\nStage C complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    run_stage_c()
