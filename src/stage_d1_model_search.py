"""
Stage D1 — Systematic Model Search to Maximize Pooled OOF AUC
n=129, EPV~6, 22 pre-treatment features, binary mRS 3-month outcome

Strategies:
  1. Nested CV GridSearch (inner 3-fold) for each model family
  2. Pooled OOF AUC = trung binh xac suat tren 129 benh nhan x 5 seeds
  3. So sanh: ElasticNet, L2 LR, SVM RBF, SVM Linear, GBM, RF, Ensemble
  4. Chon model tot nhat lam flagship moi

CHAY: python nhanh_2/src/stage_d1_model_search.py
"""
import os, json, warnings, time
import numpy as np
import pandas as pd

import pathlib as _pathlib
_HERE = _pathlib.Path(__file__).resolve().parent          # src/
BASE_DIR = str(_HERE.parent)                               # project root (FINAL/)

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                               VotingClassifier)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
COHORT_CSV = os.path.join(BASE_DIR, "outputs", "stage_b", "cleaned_cohort.csv")
META_JSON  = os.path.join(BASE_DIR, "outputs", "stage_b", "cohort_metadata.json")
OUT_DIR    = os.path.join(BASE_DIR, "outputs", "stage_d1")
os.makedirs(OUT_DIR, exist_ok=True)
# ---------------------------------------------------------------------------

print("=" * 68)
print("  STAGE D1 — SYSTEMATIC MODEL SEARCH (Nested CV, n=129)")
print("=" * 68)

df = pd.read_csv(COHORT_CSV)
with open(META_JSON) as f:
    meta = json.load(f)

X = df[meta["baseline_predictors"]].values
y = df[meta["target_binary"]].astype(int).values
print(f"Dataset: {len(y)} samples | Positive: {sum(y==1)} | Negative: {sum(y==0)}")
print(f"Class ratio: {sum(y==1)/len(y)*100:.1f}% good outcome\n")

# ---------------------------------------------------------------------------
# MODEL SEARCH SPACE
# ---------------------------------------------------------------------------
search_configs = {

    # === 1. ElasticNet LR — tune C & l1_ratio ===
    "ElasticNet LR": {
        "pipe": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                       class_weight="balanced",
                                       max_iter=5000, random_state=42))
        ]),
        "params": {
            "clf__C": [0.01, 0.05, 0.1, 0.5, 1.0],
            "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    },

    # === 2. L2 LR — tune C ===
    "L2 LR": {
        "pipe": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", solver="lbfgs",
                                       class_weight="balanced",
                                       max_iter=5000, random_state=42))
        ]),
        "params": {
            "clf__C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        }
    },

    # === 3. SVM RBF (MinMax) — tune C & gamma ===
    "SVM RBF": {
        "pipe": Pipeline([
            ("sc", MinMaxScaler()),
            ("clf", SVC(kernel="rbf", probability=True,
                        class_weight="balanced", random_state=42))
        ]),
        "params": {
            "clf__C":     [0.1, 0.5, 1.0, 5.0, 10.0, 20.0],
            "clf__gamma": [0.001, 0.005, 0.01, 0.05, 0.1, "scale"],
        }
    },

    # === 4. SVM Linear — tune C ===
    "SVM Linear": {
        "pipe": Pipeline([
            ("sc", StandardScaler()),
            ("clf", SVC(kernel="linear", probability=True,
                        class_weight="balanced", random_state=42))
        ]),
        "params": {
            "clf__C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        }
    },

    # === 5. GBM — tune depth, lr, n_est ===
    "GBM": {
        "pipe": Pipeline([
            ("sc", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=42,
                                               subsample=0.8))
        ]),
        "params": {
            "clf__n_estimators":  [50, 100, 200],
            "clf__max_depth":     [2, 3, 4],
            "clf__learning_rate": [0.05, 0.1, 0.2],
        }
    },

    # === 6. Random Forest — tune depth, n_est ===
    "Random Forest": {
        "pipe": Pipeline([
            ("sc", StandardScaler()),
            ("clf", RandomForestClassifier(class_weight="balanced",
                                           random_state=42))
        ]),
        "params": {
            "clf__n_estimators": [50, 100, 200],
            "clf__max_depth":    [3, 4, 5, None],
            "clf__min_samples_leaf": [2, 3, 5],
        }
    },
}

# ---------------------------------------------------------------------------
# EVALUATION — Nested 10-fold CV x 5 Seeds (25 outer folds each)
# Inner: 3-fold GridSearch for hyperparam
# ---------------------------------------------------------------------------
seeds    = [42, 100, 2024, 777, 999]

print(f"{'Model':<16} ", end="")
for s in seeds:
    print(f" Seed{s:>5}", end="")
print("   POOLED AUC")
print("-" * 80)

results_summary = {}

for m_name, cfg in search_configs.items():
    seed_aucs = []

    t0 = time.time()
    for seed in seeds:
        outer_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        seed_probs  = np.zeros(len(y))
        class_preds = np.zeros(len(y))

        for tr_idx, va_idx in outer_skf.split(X, y):
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]

            # Inner 3-fold GridSearch
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            gs = GridSearchCV(
                cfg["pipe"], cfg["params"],
                cv=inner_cv, scoring="roc_auc",
                n_jobs=1, refit=True
            )
            gs.fit(Xtr, ytr)
            prob = gs.predict_proba(Xva)[:, 1]
            seed_probs[va_idx]  = prob
            class_preds[va_idx] = gs.predict(Xva)

        seed_auc = roc_auc_score(y, seed_probs)
        seed_aucs.append(seed_auc)

    mean_seed_auc = np.mean(seed_aucs)
    std_seed_auc  = np.std(seed_aucs)
    elapsed = time.time() - t0

    results_summary[m_name] = {
        "mean_seed_auc": round(float(mean_seed_auc), 4),
        "std_seed_auc":  round(float(std_seed_auc), 4),
        "seed_aucs":    [round(float(a), 4) for a in seed_aucs],
    }

    print(f"{m_name:<16} ", end="")
    for a in seed_aucs:
        print(f"  {a:.4f}", end="")
    print(f"   {mean_seed_auc:.4f}  [{elapsed:.0f}s]")


# ---------------------------------------------------------------------------
# SUMMARY TABLE
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("  FINAL SUMMARY — Mean OOF AUC (5 seeds x 10 folds)")
print("=" * 68)
print(f"{'Model':<20} {'Mean AUC':>11} {'Std AUC':>18}")
print("-" * 68)
for name, r in sorted(results_summary.items(), key=lambda x: x[1]["mean_seed_auc"], reverse=True):
    std_str = f"± {r.get('std_seed_auc',0):.4f}" if 'std_seed_auc' in r else "  —  "
    print(f"{name:<20} {r['mean_seed_auc']:>11.4f} {std_str:>18}")

# Save results
with open(os.path.join(OUT_DIR, "model_search_results.json"), "w") as f:
    json.dump(results_summary, f, indent=2)

# ---------------------------------------------------------------------------
# BAR CHART
# ---------------------------------------------------------------------------
plot_data = {k: v["mean_seed_auc"] for k, v in
             sorted(results_summary.items(), key=lambda x: x[1]["mean_seed_auc"])}
best_name = max(results_summary, key=lambda x: results_summary[x]["mean_seed_auc"])
colors = ["#2ca02c" if k == best_name else "#4878cf" for k in plot_data]

plt.figure(figsize=(10, 6))
bars = plt.barh(list(plot_data.keys()), list(plot_data.values()),
                color=colors, alpha=0.85)
plt.axvline(x=0.7, color="red", linestyle="--", alpha=0.5, label="AUC=0.70 target")
plt.xlim(0.55, 0.85)
plt.xlabel("Mean OOF ROC AUC (5 seeds x 10-fold Nested CV)", fontsize=11)
plt.title("Model Search Results on ISLES 2024 (n=129)\n22 Pre-treatment Features",
          fontsize=12, fontweight="bold")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4, axis="x")
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.003, bar.get_y() + bar.get_height()/2,
             f"{w:.4f}", va="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "model_search_auc.png"), dpi=300)
plt.close()
print(f"\n[Done] Results -> {OUT_DIR}")
print("=" * 68)
