"""
Stage F — SHAP Feature Importance & Sensitivity Analysis
Uses GBM TreeSHAP (shap.TreeExplainer) on the full-cohort fitted flagship pipeline.

Outputs (stage_f/):
  - shap_feature_importance.csv   : mean |SHAP|, rank, mean_shap_good, shap_source
  - shap_summary_plot.png         : horizontal bar chart coloured by direction

CHAY: python nhanh_2/src/stage_f_shap_sensitivity.py
"""
import os, json, pickle, warnings, sys

import pathlib as _pathlib
_HERE = _pathlib.Path(__file__).resolve().parent          # src/
BASE_DIR = str(_HERE.parent)                               # project root (FINAL/)

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
warnings.filterwarnings('ignore')

OUT_DIR  = os.path.join(BASE_DIR, "outputs", "stage_f")
STAGE_B  = os.path.join(BASE_DIR, "outputs", "stage_b")
STAGE_D2 = os.path.join(BASE_DIR, "outputs", "stage_d2")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print("  STAGE F — GBM TREESHAP FEATURE IMPORTANCE")
print("=" * 65)

# ── Load data & pipeline ──────────────────────────────────────────────────
meta  = json.load(open(os.path.join(STAGE_B, "cohort_metadata.json")))
df    = pd.read_csv(os.path.join(STAGE_B, "cleaned_cohort.csv"))
preds = meta["baseline_predictors"]
X     = df[preds].values
y     = df[meta["target_binary"]].astype(int).values   # 1=good, 0=poor

with open(os.path.join(STAGE_D2, "flagship_gbm_pipeline.pkl"), "rb") as f:
    pipe = pickle.load(f)

clf = pipe.named_steps["clf"]
print(f"Model  : {type(clf).__name__}")
print(f"N      : {len(y)}  |  Good={int((y==1).sum())}  |  Poor={int((y==0).sum())}")
print(f"Features: {len(preds)}")

# ── Transform X through pipeline (scaler + SelectKBest k=all = passthrough)
X_sc = pipe.named_steps["sc"].transform(X)
X_fs = pipe.named_steps["fs"].transform(X_sc)
print(f"X_transformed shape: {X_fs.shape}")

# ── TreeSHAP ──────────────────────────────────────────────────────────────
print("\nRunning shap.TreeExplainer ...")
explainer  = shap.TreeExplainer(clf)
shap_vals  = explainer.shap_values(X_fs)      # (n, p) for good=1 class
mean_abs   = np.abs(shap_vals).mean(axis=0)   # mean |SHAP| per feature
mean_dir   = shap_vals.mean(axis=0)           # signed mean (P(good) direction)

print(f"SHAP values shape : {shap_vals.shape}")
print(f"Nonzero features  : {np.sum(mean_abs > 1e-6)} / {len(preds)}")

# ── Build importance table ────────────────────────────────────────────────
df_shap = pd.DataFrame({
    "feature":         preds,
    "mean_abs_shap":   mean_abs,
    "mean_shap_good":  mean_dir,   # positive = higher value → more likely good outcome
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
df_shap["rank"]        = range(1, len(df_shap) + 1)
df_shap["shap_source"] = "TreeSHAP on GBM (full-cohort refit)"

print("\nTop 12 features:")
print(df_shap[["rank", "feature", "mean_abs_shap", "mean_shap_good"]].head(12).to_string(index=False))

csv_path = os.path.join(OUT_DIR, "shap_feature_importance.csv")
df_shap[["feature", "mean_abs_shap", "rank", "mean_shap_good", "shap_source"]].to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# ── SHAP bar plot ─────────────────────────────────────────────────────────
active = df_shap[df_shap["mean_abs_shap"] > 1e-6].copy()
plot_df = active.sort_values("mean_abs_shap")           # ascending for barh

colors = ["#2ca02c" if v > 0 else "#d62728" for v in plot_df["mean_shap_good"]]

fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.45)))
ax.barh(plot_df["feature"], plot_df["mean_abs_shap"],
        color=colors, alpha=0.85, edgecolor="white")
ax.set_xlabel("Mean |SHAP value| — Impact on P(good outcome)", fontsize=18)
ax.set_title("GBM Feature Importance (TreeSHAP)\n"
             "ISLES 2024, n=129\nGreen: average SHAP contribution toward good outcome\nRed: average SHAP contribution toward poor outcome",
             fontsize=18)
ax.tick_params(axis="y", which="major", labelsize=18)
ax.tick_params(axis="x", which="major", labelsize=16)
ax.grid(True, linestyle="--", alpha=0.4, axis="x")
legend_elements = [
    Patch(facecolor="#2ca02c", label="Mean SHAP → Good outcome"),
    Patch(facecolor="#d62728", label="Mean SHAP → Poor outcome"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=18)
plt.tight_layout()
png_path = os.path.join(OUT_DIR, "shap_summary_plot.png")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {png_path}")

# ── Sensitivity: 22 baseline vs 26 with leakage ──────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

leakage_cols = meta["leakage_predictors"]
all_cols     = preds + leakage_cols
X_all        = df[all_cols].values

sc22  = StandardScaler(); X22_s = sc22.fit_transform(X)
sc26  = StandardScaler(); X26_s = sc26.fit_transform(X_all)
lr22  = LogisticRegression(max_iter=2000, random_state=42).fit(X22_s, y)
lr26  = LogisticRegression(max_iter=2000, random_state=42).fit(X26_s, y)
auc22 = roc_auc_score(y, lr22.predict_proba(X22_s)[:, 1])
auc26 = roc_auc_score(y, lr26.predict_proba(X26_s)[:, 1])

sens = {
    "auc_22_baseline_predictors":          round(float(auc22), 4),
    "auc_26_with_leakage_predictors":      round(float(auc26), 4),
    "auc_uplift_from_leakage":             round(float(auc26 - auc22), 4),
    "leakage_predictors_tested":           leakage_cols,
    "note": "LR on full data — shows how much intra-procedural info adds (leakage check)"
}
import json
with open(os.path.join(OUT_DIR, "sensitivity_summary.json"), "w", encoding="utf-8") as f:
    json.dump(sens, f, indent=2)
print(f"\nLeakage sensitivity: LR-22={auc22:.4f}  LR-26={auc26:.4f}  "
      f"Δ={auc26-auc22:+.4f} (upside from leakage vars)")
print(f"\nStage F complete! Results saved to: {OUT_DIR}")


def run_stage_f():
    """Entry point — all code runs at module level above."""
    pass


if __name__ == "__main__":
    run_stage_f()
