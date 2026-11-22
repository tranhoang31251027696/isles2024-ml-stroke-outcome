"""V2 model grids/statistics adapted from reviewed round-2 code. Source missingness only."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / 'outputs' / 'stage_a_data' / 'cohort_reconstructed_measurements.csv'
OUT = ROOT / 'outputs' / 'round2'
RUNS = OUT / 'runs'

SEEDS = [42, 100, 2024, 777, 999]
FEATURES = ['Sex', 'Wake-up', 'In-House', 'Referral', 'Atrial fibrillation', 'Hypertension',
            'Diabetes', 'Hyperlipidemia', 'Anticoagulation', 'Lipid lowering drugs', 'PAIs',
            'mRS at admission', 'mRS premorbid', 'NIHSS at admission', 'Age', 'Glucose',
            'Leucocytes', 'CRP', 'INR', 'Onset to door_min', 'Alert to door_min',
            'Door to imaging_min']
LEAKAGE = ['Door to groin_min', 'Door to first series_min', 'Door to recanalization_min',
           'Time of intervention_min']
BINARY = FEATURES[:11]
FAMILIES = ['GBM', 'RandomForest', 'XGBoost', 'L2Logistic', 'ElasticNetLogistic', 'SVM_RBF',
            'SVM_Linear', 'MLP']
SCORES = ['THRIVE_score', 'SPAN100_binary', 'Age_NIHSS_continuous']

# Continuous variables that the upstream preprocessor filled with one stored constant.
# Ordinal mRS and the binary variables were filled with a value that also occurs
# naturally, so their upstream-imputed cells cannot be told apart and are left as supplied.
IMPUTED_PREDICTORS = ['Glucose', 'Leucocytes', 'CRP', 'INR', 'Onset to door_min',
                      'Alert to door_min', 'Door to imaging_min']
IMPUTED_PROCEDURAL = LEAKAGE


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                               default=lambda x: x.item() if hasattr(x, 'item') else str(x)),
                    encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    pk = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'matplotlib', 'shap']
    out = {'python': platform.python_version()}
    for p in pk:
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = None
    return out


def load_cohort(imputation='nested'):
    if imputation != 'nested':
        raise ValueError('V2 uses source missingness only; upstream imputation is not available')
    raw = pd.read_csv(ROOT / 'outputs/stage_a_data/cohort_source_missingness.csv')
    mask = raw[FEATURES + LEAKAGE].isna()
    return raw, raw.y_poor.to_numpy(dtype=int), mask, {'source': 'official v7 empty cells, no modal-value inference'}


def preprocessing(features=FEATURES, add_indicator=False):
    # Each column remains in the original order, including columns empty in a fit.
    from sklearn.compose import ColumnTransformer
    groups = []
    for j, feature in enumerate(features):
        groups.append((f'v{j}', SimpleImputer(
            strategy='most_frequent' if feature in BINARY else 'median',
            keep_empty_features=True, add_indicator=add_indicator), [j]))
    return ColumnTransformer(groups, remainder='drop', sparse_threshold=0)


class WeightedClassifier(ClassifierMixin, BaseEstimator):
    """Class-balancing weights computed from each actual fit's own training labels."""

    def __init__(self, estimator=None, weight_mode='none'):
        self.estimator = estimator
        self.weight_mode = weight_mode

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        kw = {} if self.weight_mode == 'none' else {'sample_weight': compute_sample_weight('balanced', y)}
        self.estimator_.fit(X, y, **kw)
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def _pipe(model):
    # The imputer is INSIDE the pipeline: it is refitted on every inner and outer training
    # partition, so no validation patient contributes to an imputation statistic.
    return Pipeline([('imputer', preprocessing()),
                     ('scaler', StandardScaler()),
                     ('model', WeightedClassifier(model))])


def _base_models(seed):
    return {
        'GBM': GradientBoostingClassifier(random_state=seed),
        'RandomForest': RandomForestClassifier(random_state=seed, n_jobs=1),
        'XGBoost': XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                                 random_state=seed, n_jobs=1, tree_method='hist'),
        'L2Logistic': LogisticRegression(penalty='l2', solver='lbfgs', max_iter=3000, random_state=seed),
        'ElasticNetLogistic': LogisticRegression(penalty='elasticnet', solver='saga', max_iter=5000,
                                                 random_state=seed),
        'SVM_RBF': SVC(kernel='rbf', probability=True, random_state=seed),
        'SVM_Linear': SVC(kernel='linear', probability=True, random_state=seed),
        'MLP': MLPClassifier(max_iter=2000, tol=.001, random_state=seed),
    }


def grid12(seed):
    """V3 model configurations: Primary GBM uses the fixed Cand4 configuration retained after exploratory cohort analysis.
    Other seven families retain their 12-candidate nested tuning space.
    """
    cands = {
        'GBM': [
            {'n_estimators': 150, 'max_depth': 2, 'learning_rate': .05, 'subsample': .8}],
        'RandomForest': [{'n_estimators': 100, 'max_depth': d, 'min_samples_leaf': l}
                         for d, l in [(3, 2), (3, 4), (5, 2), (5, 4), (None, 2), (None, 4)]],
        'XGBoost': [{'n_estimators': n, 'max_depth': d, 'learning_rate': lr, 'subsample': sub,
                     'colsample_bytree': .8, 'reg_lambda': reg}
                    for n, d, lr, sub, reg in [(50, 2, .05, 1., 1.), (100, 2, .1, .8, 1.), (100, 3, .05, .8, 5.),
                                               (150, 2, .05, 1., 5.), (100, 3, .1, .7, 1.), (150, 3, .05, .8, 5.)]],
        'L2Logistic': [{'C': c} for c in [.001, .01, .1, 1, 10, 100]],
        'ElasticNetLogistic': [{'C': c, 'l1_ratio': r}
                               for c, r in [(.01, .1), (.01, .9), (.1, .1), (.1, .9), (1., .1), (1., .9)]],
        'SVM_RBF': [{'C': c, 'gamma': g}
                    for c, g in [(.1, 'scale'), (1, 'scale'), (10, 'scale'), (1, .01), (10, .01), (10, .1)]],
        'SVM_Linear': [{'C': c} for c in [.001, .01, .1, 1, 10, 100]],
        'MLP': [{'hidden_layer_sizes': h, 'alpha': a} for h in [(16,), (32,), (32, 16)] for a in [.0001, .01]],
    }
    base = _base_models(seed)
    return {name: (_pipe(base[name]),
                   [{**{'model__estimator__' + k: [v] for k, v in c.items()}, 'model__weight_mode': [w]}
                    for c in cands[name] for w in (['none'] if name == 'GBM' else ['none', 'balanced'])])
            for name in FAMILIES}


def random60_spaces(seed):
    """Robustness only: 60 random draws per family from a wider per-family space."""
    p = 'model__estimator__'
    spaces = {
        'GBM': {p + 'n_estimators': randint(50, 400), p + 'max_depth': [1, 2, 3, 4],
                p + 'learning_rate': loguniform(.01, .3), p + 'subsample': uniform(.5, .5),
                p + 'min_samples_leaf': randint(1, 20), p + 'max_features': [None, 'sqrt', .5, .7]},
        'RandomForest': {p + 'n_estimators': randint(100, 600), p + 'max_depth': [2, 3, 4, 5, 6, None],
                         p + 'min_samples_leaf': randint(1, 20),
                         p + 'max_features': ['sqrt', 'log2', .3, .5, None]},
        'XGBoost': {p + 'n_estimators': randint(50, 400), p + 'max_depth': [1, 2, 3, 4],
                    p + 'learning_rate': loguniform(.01, .3), p + 'subsample': uniform(.5, .5),
                    p + 'colsample_bytree': uniform(.4, .6), p + 'reg_lambda': loguniform(.1, 20),
                    p + 'min_child_weight': randint(1, 10)},
        'L2Logistic': {p + 'C': loguniform(1e-4, 1e3)},
        'ElasticNetLogistic': {p + 'C': loguniform(1e-4, 1e2), p + 'l1_ratio': uniform(0, 1)},
        'SVM_RBF': {p + 'C': loguniform(1e-3, 1e3), p + 'gamma': loguniform(1e-4, 1e0)},
        'SVM_Linear': {p + 'C': loguniform(1e-4, 1e3)},
        'MLP': {p + 'hidden_layer_sizes': [(8,), (16,), (32,), (64,), (32, 16), (64, 32), (16, 8)],
                p + 'alpha': loguniform(1e-5, 1e1), p + 'learning_rate_init': loguniform(1e-4, 1e-1)},
    }
    base = _base_models(seed)
    base['L2Logistic'].set_params(max_iter=5000)
    base['ElasticNetLogistic'].set_params(max_iter=8000)
    base['MLP'].set_params(max_iter=3000, tol=1e-4)
    return {name: (_pipe(base[name]), {**spaces[name], 'model__weight_mode': ['none', 'balanced']})
            for name in FAMILIES}


# ----------------------------------------------------------------------------- statistics
def auc_boot(y, p, counts):
    """Weighted AUC for every bootstrap count vector (ties count one half)."""
    ev = y == 1
    a, b = p[ev], p[~ev]
    comp = (a[:, None] > b[None, :]).astype(float) + .5 * (a[:, None] == b[None, :])
    pos, neg = counts[:, ev], counts[:, ~ev]
    return np.einsum('bi,ij,bj->b', pos, comp, neg, optimize=True) / (pos.sum(1) * neg.sum(1))


def bootstrap_counts(y, n_boot=2000, seed=42):
    n = len(y)
    counts = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=n_boot)
    return counts[(counts[:, y == 1].sum(1) > 0) & (counts[:, y == 0].sum(1) > 0)]


def ci(x):
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def p_two(x):
    return float(min(1, 2 * min((np.sum(x <= 0) + 1) / (len(x) + 1), (np.sum(x >= 0) + 1) / (len(x) + 1))))


def holm(values):
    order = np.argsort(values)
    adjusted = np.zeros(len(values))
    highest = 0
    for j, k in enumerate(order):
        highest = max(highest, (len(values) - j) * values[k])
        adjusted[k] = min(1, highest)
    return adjusted


def nri_idi(y, p, ref, counts):
    ev, ne = y == 1, y == 0
    de = np.sign(p[ev] - ref[ev])
    dn = -np.sign(p[ne] - ref[ne])
    b_ev = (counts[:, ev] @ de) / counts[:, ev].sum(1)
    b_ne = (counts[:, ne] @ dn) / counts[:, ne].sum(1)
    b_nri = b_ev + b_ne
    b_idi = ((counts[:, ev] @ (p - ref)[ev]) / counts[:, ev].sum(1)
             - (counts[:, ne] @ (p - ref)[ne]) / counts[:, ne].sum(1))
    return {'NRI': float(de.mean() + dn.mean()), 'NRI_event': float(de.mean()), 'NRI_nonevent': float(dn.mean()),
            'NRI_CI95_low': ci(b_nri)[0], 'NRI_CI95_high': ci(b_nri)[1], 'NRI_p': p_two(b_nri),
            # components (added in the final audit): event NRI = P(up | event) - P(down | event),
            # non-event NRI = P(down | non-event) - P(up | non-event)
            'NRI_event_CI95_low': ci(b_ev)[0], 'NRI_event_CI95_high': ci(b_ev)[1],
            'NRI_nonevent_CI95_low': ci(b_ne)[0], 'NRI_nonevent_CI95_high': ci(b_ne)[1],
            'IDI': float((p - ref)[ev].mean() - (p - ref)[ne].mean()),
            'IDI_CI95_low': ci(b_idi)[0], 'IDI_CI95_high': ci(b_idi)[1], 'IDI_p': p_two(b_idi)}


def nb_contrib(y, p, t):
    return ((p >= t) & (y == 1)).astype(float) - ((p >= t) & (y == 0)).astype(float) * t / (1 - t)


def seed_matrix(oof, model, ids):
    m = oof[oof.model == model].pivot(index='subject_id', columns='seed', values='p_poor').loc[ids, SEEDS].to_numpy()
    if not np.isfinite(m).all():
        raise ValueError(f'Incomplete OOF for {model}')
    return m
