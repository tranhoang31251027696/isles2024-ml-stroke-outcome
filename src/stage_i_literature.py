"""Literature-specified fixed configurations on the same cohort, predictors and outer folds.

Restores the benchmark of the submitted manuscript with the reviewers' corrections applied:
  * genuine XGBClassifier instead of a GradientBoostingClassifier proxy          (R3#2)
  * identical patients, predictors, outer folds and fold-nested imputation       (R1#3, R1#4)
  * the model of interest is fixed Cand4 GBM in V3, whose out-of-fold
    estimate does not adjust for historical cohort-informed selection                                     (R1#5, R3#1)
Hyper-parameters stated in the submitted manuscript are used; anything not stated there
is the library default. These are re-implementations, not the published models.

  python src/round2_literature.py --primary-run primary
"""
import os
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_k, '1')
import argparse
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from round2_common import (OUT, RUNS, SEEDS, FEATURES, dump, environment, load_cohort, auc_boot,
                           bootstrap_counts, ci, p_two, holm, seed_matrix, ROOT, preprocessing, digest)


def _xgb(seed, **kw):
    return XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=seed, n_jobs=1,
                         tree_method='hist', **kw)


def models(seed):
    mm, ss = MinMaxScaler, StandardScaler
    spec = {
        'Otieno et al. - XGBoost (d=6)': (mm, _xgb(seed, n_estimators=100, max_depth=6, learning_rate=.1)),
        'Otieno et al. - SVM RBF': (mm, SVC(kernel='rbf', C=10, gamma=.01, probability=True, random_state=seed)),
        'Otieno et al. - mLR (L2)': (mm, LogisticRegression(penalty='l2', C=1, max_iter=5000, random_state=seed)),
        'Otieno et al. - ANN (100-50)': (mm, MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu',
                                                           max_iter=1000, random_state=seed)),
        'Li et al. - ElasticNet LR': (ss, LogisticRegression(penalty='elasticnet', l1_ratio=.5, C=.1, solver='saga',
                                                             max_iter=8000, random_state=seed)),
        'Liu et al. - XGBoost (d=4)': (ss, _xgb(seed, n_estimators=100, max_depth=4, learning_rate=.05)),
        'Bogey et al. - RF (d=5)': (ss, RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed, n_jobs=1)),
        'Bogey et al. - GBM (d=3)': (ss, GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=.1,
                                                                    random_state=seed)),
    }
    return {n: Pipeline([('imputer', preprocessing()),
                         ('scaler', sc()), ('model', m)]) for n, (sc, m) in spec.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--primary-run', default='primary')
    ap.add_argument('--imputation', choices=['nested', 'upstream'], default='nested')
    a = ap.parse_args()
    warnings.simplefilter('ignore')
    out = ROOT / 'outputs' / 'stage_i_literature_benchmark'
    out.mkdir(parents=True, exist_ok=True)
    raw, y, _, _ = load_cohort(a.imputation)
    X = raw[FEATURES].to_numpy()
    rows = []
    fit_checks = []
    for seed in SEEDS:
        for fold, (tr, va) in enumerate(StratifiedKFold(10, shuffle=True, random_state=seed).split(X, y)):
            for name, pipe in models(seed).items():
                pipe.fit(X[tr], y[tr])
                prep = pipe.named_steps['imputer']
                for j, f in enumerate(FEATURES):
                    values = raw[f].iloc[tr].dropna()
                    expected = (values.mode().iloc[0] if j < 11 else values.median()) if len(values) else 0
                    assert np.isclose(prep.named_transformers_[f'v{j}'].statistics_[0], expected)
                fit_checks.append({'seed':seed,'fold':fold,'model':name,'training_n':len(tr),'validation_n':len(va)})
                p = pipe.predict_proba(X[va])[:, list(pipe.classes_).index(1)]
                rows.extend({'subject_id': raw.subject_id.iloc[i], 'seed': seed, 'fold': fold, 'model': name,
                             'y_poor': int(y[i]), 'p_poor': float(q)} for i, q in zip(va, p))
    lit = pd.DataFrame(rows)
    lit.to_csv(out / 'oof_predictions_long.csv', index=False)
    manifest = pd.read_csv(ROOT/'outputs/stage_a_data/fold_manifest.csv')
    manifest = manifest[(manifest.level=='outer_validation') & manifest.seed.isin(SEEDS)]
    checked = lit.merge(manifest[['subject_id','seed','outer_fold']],on=['subject_id','seed'],validate='many_to_one')
    assert len(checked)==len(lit) and (checked.fold==checked.outer_fold).all()
    assert len(lit)==129*5*8 and not lit.duplicated(['subject_id','seed','model']).any()
    assert np.isfinite(lit.p_poor).all() and lit.p_poor.between(0,1).all()
    dump(out/'verification.json',{'all_passed':True,'rows':len(lit),'models':8,'seeds':5,'outer_folds':10,
         'outer_imputer_statistics_verified_fits':len(fit_checks),'fold_manifest_matches':True,
         'source_cohort_sha256':digest(ROOT/'outputs/stage_a_data/cohort_source_missingness.csv')})

    primary = pd.read_csv(ROOT / 'outputs/stage_b_training/primary_oof_predictions.csv').rename(columns={'value':'p_poor'})
    ids = raw.subject_id
    counts = bootstrap_counts(y)
    G = seed_matrix(primary, 'GBM', ids)
    bg = np.mean([auc_boot(y, G[:, j], counts) for j in range(5)], axis=0)
    g_auc = [roc_auc_score(y, G[:, j]) for j in range(5)]
    table = [{'model': 'Fixed Cand4 GBM (this study)', 'mean_seed_auc': np.mean(g_auc), 'sd_seed_auc': np.std(g_auc),
              'mean_seed_auc_CI95_low': ci(bg)[0], 'mean_seed_auc_CI95_high': ci(bg)[1],
              'accuracy_pct': 100 * np.mean([accuracy_score(y, (G[:, j] >= .5).astype(int)) for j in range(5)]),
              'brier': np.mean([brier_score_loss(y, G[:, j]) for j in range(5)]),
              'ensemble_auc': roc_auc_score(y, G.mean(1))}]
    for name in sorted(lit.model.unique()):
        M = seed_matrix(lit, name, ids)
        b = np.mean([auc_boot(y, M[:, j], counts) for j in range(5)], axis=0)
        aucs = [roc_auc_score(y, M[:, j]) for j in range(5)]
        d = bg - b
        table.append({'model': name, 'mean_seed_auc': np.mean(aucs), 'sd_seed_auc': np.std(aucs),
                      'mean_seed_auc_CI95_low': ci(b)[0], 'mean_seed_auc_CI95_high': ci(b)[1],
                      'accuracy_pct': 100 * np.mean([accuracy_score(y, (M[:, j] >= .5).astype(int)) for j in range(5)]),
                      'brier': np.mean([brier_score_loss(y, M[:, j]) for j in range(5)]),
                      'ensemble_auc': roc_auc_score(y, M.mean(1)),
                      'delta_GBM_minus_model': np.mean(g_auc) - np.mean(aucs), 'delta_CI95_low': ci(d)[0],
                      'delta_CI95_high': ci(d)[1], 'p_two_sided_conditional': p_two(d)})
    comp = [r for r in table if 'p_two_sided_conditional' in r]
    for r, adj in zip(comp, holm([r['p_two_sided_conditional'] for r in comp])):
        r['p_Holm_eight_comparisons'] = float(adj)
    t = pd.DataFrame(table).sort_values('mean_seed_auc', ascending=False)
    t.to_csv(out / 'literature_benchmark.csv', index=False)
    print('Completed 400 fits on matched V3 folds.',flush=True)
    dump(out / 'protocol.json', {'imputation': a.imputation, 'primary_run': a.primary_run, 'environment': environment(),
                                 'note': 'fixed literature-specified configurations; genuine XGBoost; no tuning. Comparator weights are library defaults (unweighted); this is an auxiliary configuration benchmark; primary GBM is fixed Cand4 after cohort exploration, not a nested-tuned model.', 'source': 'official v7 source missingness', 'imputation_detail': 'binary training-mode, other training-median, scaler training-only', 'bootstrap_scope': 'conditional patient bootstrap; not full model retraining bootstrap', 'configurations': {n: str(p) for n,p in models(42).items()}})
    print(t.to_string(index=False, float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
