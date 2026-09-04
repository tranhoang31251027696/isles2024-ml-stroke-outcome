"""
Stage D2 - Fine-tune GBM (flagship moi) + Ensemble GBM+RF
AUC baseline: GBM=0.7639, Ensemble=0.7684

CHAY: python nhanh_2/src/stage_d2_gbm_finetune.py
"""
import os, json, warnings, pickle
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

COHORT_CSV = os.path.join(BASE_DIR, "outputs", "stage_b", "cleaned_cohort.csv")
META_JSON  = os.path.join(BASE_DIR, "outputs", "stage_b", "cohort_metadata.json")
OUT_DIR    = os.path.join(BASE_DIR, "outputs", "stage_d2")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print("  STAGE D2 - FINE-TUNE GBM (New Flagship)")
print("=" * 65)

df = pd.read_csv(COHORT_CSV)
with open(META_JSON) as f:
    meta = json.load(f)

feature_names = meta["baseline_predictors"]
X = df[feature_names].values
y = df[meta["target_binary"]].astype(int).values
print(f"n={len(y)} | Positive={sum(y==1)} | Negative={sum(y==0)}\n")

# ---------------------------------------------------------------------------
# FINE-TUNE GRID - quanh GBM params tot nhat
# ---------------------------------------------------------------------------
gbm_grid = {
    "clf__n_estimators":       [80],
    "clf__max_depth":          [3],
    "clf__learning_rate":      [0.1],
    "clf__subsample":          [0.7],
}

from sklearn.feature_selection import SelectKBest, f_classif

gbm_pipe = Pipeline([
    ("sc",  StandardScaler()),
    ("fs", SelectKBest(score_func=f_classif, k='all')),
    ("clf", GradientBoostingClassifier(random_state=42)),
])

rf_grid = {
    "clf__n_estimators":    [100],
    "clf__max_depth":       [5],
    "clf__min_samples_leaf":[3],
}

rf_pipe = Pipeline([
    ("sc",  StandardScaler()),
    ("clf", RandomForestClassifier(class_weight="balanced", random_state=42)),
])

rf_pipe = Pipeline([
    ("sc",  StandardScaler()),
    ("clf", RandomForestClassifier(class_weight="balanced", random_state=42)),
])

seeds = [42, 100, 2024, 777, 999]

def nested_cv_oof(pipe, param_grid, X, y, seeds, label):
    all_probs  = {"pooled": np.zeros(len(y))}
    all_counts = 0
    seed_aucs  = []
    best_params_list = []

    print(f"\n  [{label}]")
    print(f"  {'Seed':>6}  {'AUC':>8}  {'Acc%':>7}  Best Params (most freq)")
    print(f"  {'-'*55}")

    for seed in seeds:
        outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        seed_probs  = np.zeros(len(y))
        class_preds = np.zeros(len(y))
        fold_params = []

        for tr_idx, va_idx in outer_cv.split(X, y):
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr       = y[tr_idx]

            gs = GridSearchCV(pipe, param_grid, cv=inner_cv,
                              scoring="roc_auc", n_jobs=1, refit=True)
            gs.fit(Xtr, ytr)
            prob = gs.predict_proba(Xva)[:, 1]
            seed_probs[va_idx]  = prob
            class_preds[va_idx] = gs.predict(Xva)
            fold_params.append(gs.best_params_)
            best_params_list.append(gs.best_params_)

        seed_auc = roc_auc_score(y, seed_probs)
        seed_acc = accuracy_score(y, class_preds)
        seed_aucs.append(seed_auc)
        all_probs[f"seed_{seed}"] = seed_probs
        all_probs["pooled"] += seed_probs
        all_counts += 1

        # Most freq param in this seed
        key0 = list(param_grid.keys())[0]
        vals = [p[key0] for p in fold_params]
        freq_val = max(set(vals), key=vals.count)
        print(f"  {seed:>6}  {seed_auc:>8.4f}  {seed_acc*100:>7.2f}%  {key0}={freq_val}")

    pooled = all_probs["pooled"] / all_counts
    p_auc   = np.mean(seed_aucs)
    p_acc   = accuracy_score(y, (pooled >= 0.5).astype(int))
    p_brier = brier_score_loss(y, pooled)

    print(f"\n  MEAN OOF -> AUC={p_auc:.4f}  Acc={p_acc*100:.2f}%  "
          f"Brier={p_brier:.4f}  [MeanSeed={np.mean(seed_aucs):.4f}+/-{np.std(seed_aucs):.4f}]")

    return pooled, p_auc, p_acc, p_brier, best_params_list, seed_aucs, all_probs

# Run GBM fine-tune
gbm_prob, gbm_auc, gbm_acc, gbm_brier, gbm_params, gbm_seed_aucs, gbm_all_probs = \
    nested_cv_oof(gbm_pipe, gbm_grid, X, y, seeds, "GBM Fine-tune")

# Run RF fine-tune
rf_prob, rf_auc, rf_acc, rf_brier, rf_params, rf_seed_aucs, rf_all_probs = \
    nested_cv_oof(rf_pipe, rf_grid, X, y, seeds, "Random Forest Fine-tune")

# Ensemble GBM + RF (Mean Seed AUC properly calculated)
ens_seed_aucs = []
for seed in seeds:
    ens_seed_prob = (gbm_all_probs[f"seed_{seed}"] + rf_all_probs[f"seed_{seed}"]) / 2
    ens_seed_aucs.append(roc_auc_score(y, ens_seed_prob))

ens_prob  = (gbm_prob + rf_prob) / 2
ens_auc   = np.mean(ens_seed_aucs)
ens_acc   = accuracy_score(y, (ens_prob >= 0.5).astype(int))
ens_brier = brier_score_loss(y, ens_prob)

print("\n" + "=" * 65)
print("  FINAL COMPARISON")
print("=" * 65)
print(f"  {'Model':<28} {'AUC':>8} {'Acc%':>7} {'Brier':>8}")
print(f"  {'-'*55}")
print(f"  {'GBM (fine-tuned)':<28} {gbm_auc:>8.4f} {gbm_acc*100:>7.2f} {gbm_brier:>8.4f}")
print(f"  {'RF (fine-tuned)':<28} {rf_auc:>8.4f} {rf_acc*100:>7.2f} {rf_brier:>8.4f}")
print(f"  {'Ensemble GBM+RF':<28} {ens_auc:>8.4f} {ens_acc*100:>7.2f} {ens_brier:>8.4f}")
print(f"  {'--- Baseline (Stage D1) ---':}")
print(f"  {'GBM (D2)':<28} {'0.7639':>8} {'72.87':>7} {'0.1840':>8}")
print(f"  {'Ensemble 3-model (D2)':<28} {'0.7684':>8} {'71.32':>7} {'0.1816':>8}")

# Best model recommendation
best_auc = max(gbm_auc, ens_auc)
flagship = "GBM (fine-tuned)" if gbm_auc >= ens_auc else "Ensemble GBM+RF"
print(f"\n  >> FLAGSHIP MOI: {flagship} (AUC={best_auc:.4f})")

# Most common GBM params
def most_common_params(params_list, grid_keys):
    result = {}
    for k in grid_keys:
        vals = [p[k] for p in params_list if k in p]
        result[k] = max(set(vals), key=vals.count)
    return result

gbm_best = most_common_params(gbm_params, list(gbm_grid.keys()))
print(f"\n  Most selected GBM params (across 50 folds):")
for k, v in gbm_best.items():
    print(f"    {k}: {v}")

# Train final model on ALL data with best params
final_pipe = Pipeline([
    ("sc", StandardScaler()),
    ("fs", SelectKBest(score_func=f_classif, k='all')),
    ("clf", GradientBoostingClassifier(
        n_estimators=gbm_best.get("clf__n_estimators", 80),
        max_depth=gbm_best.get("clf__max_depth", 3),
        learning_rate=gbm_best.get("clf__learning_rate", 0.1),
        subsample=gbm_best.get("clf__subsample", 0.7),
        random_state=42
    ))
])
final_pipe.fit(X, y)

model_path = os.path.join(OUT_DIR, "flagship_gbm_pipeline.pkl")
with open(model_path, "wb") as f:
    pickle.dump(final_pipe, f)

# Save metadata
results = {
    "flagship_model": flagship,
    "gbm_pooled_auc":   round(gbm_auc, 4),
    "gbm_pooled_acc":   round(gbm_acc * 100, 2),
    "gbm_pooled_brier": round(gbm_brier, 4),
    "gbm_seed_aucs":    [round(a, 4) for a in gbm_seed_aucs],
    "rf_pooled_auc":    round(rf_auc, 4),
    "ensemble_auc":     round(ens_auc, 4),
    "best_gbm_params":  {k.replace("clf__", ""): str(v) for k, v in gbm_best.items()},
    "model_path":       model_path,
    "feature_names":    feature_names,
    "n_features":       len(feature_names),
}
with open(os.path.join(OUT_DIR, "flagship_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# Export OOF predictions
oof_data = {
    'subject_id': df['subject_id'] if 'subject_id' in df.columns else np.arange(len(y)),
    'y_true': y,
}
for seed in seeds:
    oof_data[f'pred_prob_gbm_seed_{seed}'] = gbm_all_probs[f"seed_{seed}"]
    oof_data[f'pred_prob_rf_seed_{seed}'] = rf_all_probs[f"seed_{seed}"]
    oof_data[f'pred_prob_ensemble_seed_{seed}'] = (gbm_all_probs[f"seed_{seed}"] + rf_all_probs[f"seed_{seed}"]) / 2

df_oof = pd.DataFrame(oof_data)
df_oof.to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False)

# AUC comparison bar chart
names  = ["GBM\n(D2 baseline)", "RF\n(D2 baseline)", "GBM\n(fine-tuned)", "RF\n(fine-tuned)", "Ensemble\nGBM+RF"]
aucs   = [0.7639, 0.7396, gbm_auc, rf_auc, ens_auc]
colors = ["#aec7e8", "#aec7e8", "#2ca02c", "#2ca02c", "#ff7f0e"]

plt.figure(figsize=(9, 5))
bars = plt.bar(names, aucs, color=colors, alpha=0.85, edgecolor="white", width=0.6)
plt.axhline(y=0.7639, color="gray", linestyle="--", alpha=0.6, label="D2 GBM baseline")
plt.ylim(0.65, 0.82)
plt.ylabel("Mean OOF ROC AUC", fontsize=11)
plt.title("GBM Fine-tune vs Baseline\n(ISLES 2024, n=129, 50-fold CV)", fontsize=12, fontweight="bold")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3, axis="y")
for bar, auc in zip(bars, aucs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f"{auc:.4f}", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gbm_finetune_comparison.png"), dpi=300)
plt.close()

print(f"\n[Done] Model saved -> {model_path}")
print("=" * 65)
