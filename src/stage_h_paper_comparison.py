Stage H — Head-to-Head Benchmark: Re-implement published model architectures
on ISLES 2024 data (n=129, 22 pre-treatment features) using 10-fold CV.

import os, json, warnings
import numpy as np
import pandas as pd

import pathlib as _pathlib
_HERE = _pathlib.Path(__file__).resolve().parent          # src/
BASE_DIR = str(_HERE.parent)                               # project root (FINAL/)

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, f1_score


def run_stage_h_paper_comparison(n_splits=10):
    cohort_csv = os.path.join(BASE_DIR, "outputs", "stage_b", "cleaned_cohort.csv")
    meta_json  = os.path.join(BASE_DIR, "outputs", "stage_b", "cohort_metadata.json")

    df = pd.read_csv(cohort_csv)
    with open(meta_json) as f:
        meta = json.load(f)

    X = df[meta['baseline_predictors']].values
    y = df[meta['target_binary']].astype(int).values

    flagship_json = os.path.join(BASE_DIR, "outputs", "stage_d2", "flagship_results.json")
    with open(flagship_json) as f:
        d2_res = json.load(f)
    best_params = d2_res["best_gbm_params"]

    n_estimators = int(best_params.get("n_estimators", 80))
    max_depth = int(best_params.get("max_depth", 3))
    learning_rate = float(best_params.get("learning_rate", 0.1))
    subsample = float(best_params.get("subsample", 0.7))
    min_samples_leaf = int(best_params.get("min_samples_leaf", 3)) if "min_samples_leaf" in best_params else 1

    registry = {
        '[Otieno 2024] mLR  (L2 C=1, MinMax)': Pipeline([
            ('sc', MinMaxScaler()),
            ('clf', LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                                        max_iter=2000, class_weight='balanced',
                                        random_state=42)),
        ]),
        '[Otieno 2024] SVM  (RBF C=10 g=0.01, MinMax)': Pipeline([
            ('sc', MinMaxScaler()),
            ('clf', SVC(kernel='rbf', C=10.0, gamma=0.01, probability=True,
                        class_weight='balanced', random_state=42)),
        ]),
        '[Otieno 2024] ANN  (100-50 relu, MinMax)': Pipeline([
            ('sc', MinMaxScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu',
                                   solver='adam', alpha=0.0001, max_iter=500,
                                   random_state=42)),
        ]),
        '[Otieno 2024] XGBoost (n=100 d=6 lr=0.1, MinMax)': Pipeline([
            ('sc', MinMaxScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=100, max_depth=6,
                                                learning_rate=0.1, subsample=0.8,
                                                random_state=42)),
        ]),
        '[Li 2024]   Nomogram LR (ElasticNet, StdScaler)': Pipeline([
            ('sc', StandardScaler()),
            ('clf', LogisticRegression(penalty='elasticnet', l1_ratio=0.5, C=0.1,
                                        solver='saga', max_iter=3000,
                                        class_weight='balanced', random_state=42)),
        ]),
        '[Liu 2026] XGBoost (n=100 d=4 lr=0.05, StdScaler)': Pipeline([
            ('sc', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                                learning_rate=0.05, subsample=0.8,
                                                min_samples_leaf=5, random_state=42)),
        ]),
        '[Bogey 2025] RF  (n=100 d=5, StdScaler)': Pipeline([
            ('sc', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=5,
                                            min_samples_leaf=3,
                                            class_weight='balanced', random_state=42)),
        ]),
        '[Bogey 2025] GBM (n=100 d=3 lr=0.1, StdScaler)': Pipeline([
            ('sc', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                learning_rate=0.1, random_state=42)),
        ]),
        '[OUR FLAGSHIP] Proposed GBM (Repeated 10-Fold CV)': Pipeline([
            ('sc', StandardScaler()),
            ('fs', SelectKBest(score_func=f_classif, k='all')),
            ('clf', GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                learning_rate=learning_rate, subsample=subsample,
                                                min_samples_leaf=min_samples_leaf,
                                                random_state=42)),
        ]),
    }

    seeds = [42, 100, 2024, 777, 999]
    res = {k: dict(seed_aucs=[], seed_accs=[], seed_briers=[], seed_f1s=[]) for k in registry}

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for label, pipe_tmpl in registry.items():
            prob_oof = np.zeros(len(y)); class_oof = np.zeros(len(y))
            for tr_idx, va_idx in skf.split(X, y):
                Xtr, Xva = X[tr_idx], X[va_idx]
                ytr, yva = y[tr_idx], y[va_idx]

                pipe = clone(pipe_tmpl)
                # KEY FIX: vary the classifier's internal random_state with the
                # current seed, for every model (comparators AND flagship),
                # so all 9 models are evaluated under a genuinely identical
                # repeated-CV protocol, as stated in the manuscript.
                if hasattr(pipe.named_steps['clf'], 'random_state'):
                    pipe.set_params(clf__random_state=seed)

                pipe.fit(Xtr, ytr)
                prob = pipe.predict_proba(Xva)[:, 1]

                prob_oof[va_idx] = prob
                class_oof[va_idx] = (prob >= 0.5).astype(int)
            res[label]['seed_aucs'].append(roc_auc_score(y, prob_oof))
            res[label]['seed_accs'].append(accuracy_score(y, class_oof))
            res[label]['seed_briers'].append(brier_score_loss(y, prob_oof))
            res[label]['seed_f1s'].append(f1_score(y, class_oof, zero_division=0))

    rows = []
    for label, metrics in res.items():
        rows.append({
            'Model': label,
            'Accuracy (%)': round(np.mean(metrics['seed_accs']) * 100, 2),
            'Mean OOF AUC': round(np.mean(metrics['seed_aucs']), 4),
            'AUC Std': round(np.std(metrics['seed_aucs']), 4),
            'Brier Score': round(np.mean(metrics['seed_briers']), 4),
            'Brier Std': round(np.std(metrics['seed_briers']), 4),
            'F1 Score': round(np.mean(metrics['seed_f1s']), 4),
            'F1 Std': round(np.std(metrics['seed_f1s']), 4),
        })
    df_out = pd.DataFrame(rows).sort_values('Mean OOF AUC', ascending=False)
    print(df_out.to_string(index=False))
    return df_out


if __name__ == "__main__":
    run_stage_h_paper_comparison(n_splits=10)
