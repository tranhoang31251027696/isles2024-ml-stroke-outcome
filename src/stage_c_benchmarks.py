import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, brier_score_loss
from sklearn.linear_model import LogisticRegression


import pathlib as _pathlib
_HERE = _pathlib.Path(__file__).resolve().parent          # src/
BASE_DIR = str(_HERE.parent)                               # project root (FINAL/)

def run_stage_c():
    cohort_path = os.path.join(BASE_DIR, "outputs", "stage_b", "cleaned_cohort.csv")
    output_dir = os.path.join(BASE_DIR, "outputs", "stage_c")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== STAGE C: BASELINE & BENCHMARK SCORES ===")
    
    df = pd.read_csv(cohort_path)
    
    # 1. Un-scale Age (mean = 70.4564, std = 15.0285)
    age_mean = 70.4564
    age_std = 15.0285
    df['Age_raw'] = df['Age'] * age_std + age_mean
    
    # NIHSS is already raw points
    df['NIHSS_raw'] = df['NIHSS at admission']
    
    # 2. Calculate THRIVE Score (0-9)
    thrive_age_pts = np.where(df['Age_raw'] >= 80, 2, np.where(df['Age_raw'] >= 60, 1, 0))
    thrive_nihss_pts = np.where(df['NIHSS_raw'] >= 21, 4, np.where(df['NIHSS_raw'] >= 11, 2, 0))
    thrive_htn_pts = (df['Hypertension'] > 0).astype(int)
    thrive_dm_pts = (df['Diabetes'] > 0).astype(int)
    thrive_af_pts = (df['Atrial fibrillation'] > 0).astype(int)
    
    df['THRIVE_score'] = thrive_age_pts + thrive_nihss_pts + thrive_htn_pts + thrive_dm_pts + thrive_af_pts
    
    # 3. Calculate SPAN-100 Score
    df['SPAN100_score'] = df['Age_raw'] + df['NIHSS_raw']
    df['SPAN100_pos'] = (df['SPAN100_score'] >= 100).astype(int)
    
    # Save scores with subject_id
    df_scores = df[['subject_id', 'mRS_3_months_bin', 'Age_raw', 'NIHSS_raw', 'THRIVE_score', 'SPAN100_score', 'SPAN100_pos']].copy()
    df_scores.to_csv(os.path.join(output_dir, "benchmark_scores.csv"), index=False)
    
    # 4. Evaluate Benchmarks via 10-fold Stratified CV across 5 random seeds
    y = df['mRS_3_months_bin'].astype(int).values
    
    seeds = [42, 100, 2024, 777, 999]
    benchmark_results = {
        'THRIVE': [],
        'SPAN100_raw': [],
        'Univariate_NIHSS': []
    }
    
    for seed in seeds:
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        
        oof_thrive = np.zeros(len(y))
        oof_span = np.zeros(len(y))
        oof_nihss = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, y)):
            y_val = y[val_idx]
            
            # THRIVE AUC
            thrive_val = df.loc[val_idx, 'THRIVE_score'].values
            oof_thrive[val_idx] = -thrive_val
            
            # SPAN-100 AUC
            span_val = df.loc[val_idx, 'SPAN100_score'].values
            oof_span[val_idx] = -span_val
            
            # Univariate Logistic Regression on NIHSS
            nihss_train = df.loc[train_idx, 'NIHSS_raw'].values.reshape(-1, 1)
            nihss_val = df.loc[val_idx, 'NIHSS_raw'].values.reshape(-1, 1)
            
            lr_nihss = LogisticRegression()
            lr_nihss.fit(nihss_train, y[train_idx])
            pred_nihss_prob = lr_nihss.predict_proba(nihss_val)[:, 1]
            oof_nihss[val_idx] = pred_nihss_prob
            
        benchmark_results['THRIVE'].append(roc_auc_score(y, oof_thrive))
        benchmark_results['SPAN100_raw'].append(roc_auc_score(y, oof_span))
        benchmark_results['Univariate_NIHSS'].append(roc_auc_score(y, oof_nihss))
            
    summary_metrics = {}
    for model_name, aucs in benchmark_results.items():
        summary_metrics[model_name] = {
            'mean_auc': round(float(np.mean(aucs)), 4),
            'std_auc': round(float(np.std(aucs)), 4),
            'min_auc': round(float(np.min(aucs)), 4),
            'max_auc': round(float(np.max(aucs)), 4)
        }
        
    print("\n--- Benchmark Performance Summary (AUC across 5 seeds x 10 folds = 50 evaluations) ---")
    for k, v in summary_metrics.items():
        print(f"{k:20s}: Mean AUC = {v['mean_auc']:.4f} ± {v['std_auc']:.4f}")
        
    with open(os.path.join(output_dir, "benchmark_metrics_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary_metrics, f, indent=2)
        
    print(f"Stage C complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    run_stage_c()
