"""
Stage G — Final Ledger & TRIPOD+AI Report
Consolidates all stage outputs into a single gbm_final_ledger.json.

Outputs (stage_g/):
  - gbm_final_ledger.json   : single source of truth for all Results
  - performance_summary.md  : human-readable markdown table

CHAY: python nhanh_2/src/stage_g_ledger.py
"""
import os, json, datetime, sys

import pathlib as _pathlib
_HERE = _pathlib.Path(__file__).resolve().parent          # src/
BASE_DIR = str(_HERE.parent)                               # project root (FINAL/)

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd

OUT_DIR  = os.path.join(BASE_DIR, "outputs", "stage_g")
OUTS     = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print("  STAGE G — FINAL LEDGER & TRIPOD+AI REPORT (GBM)")
print("=" * 65)

# ── Load all stage outputs ─────────────────────────────────────────────────
meta   = json.load(open(os.path.join(OUTS, "stage_b", "cohort_metadata.json")))
d2_res = json.load(open(os.path.join(OUTS, "stage_d2", "flagship_results.json")))
stage_e= json.load(open(os.path.join(OUTS, "stage_e", "stage_e_summary.json"), encoding="utf-8"))
df_sh  = pd.read_csv(os.path.join(OUTS, "stage_f", "shap_feature_importance.csv"))
df_h   = json.load(open(os.path.join(OUTS, "stage_h", "literature_comparison_summary.json")))

# Stage A audit (optional – may not exist)
stage_a_path = os.path.join(OUTS, "stage_a", "stage_a_audit_report.json")
stage_a = json.load(open(stage_a_path)) if os.path.exists(stage_a_path) else {}

# ── Derive metrics from stage_e ────────────────────────────────────────────
gbm_e   = stage_e["gbm"]
paired  = stage_e["paired_gbm_vs_thrive"]
nri_idi = stage_e["nri_idi_vs_thrive"]

# Stage H primary AUC (repeated 10-fold mean-of-seeds)
flagship_h = next(r for r in df_h if "FLAGSHIP" in r["Model"])
primary_auc   = flagship_h["Mean OOF AUC"]
primary_auc_std = flagship_h["AUC Std"]
seed_aucs_h   = flagship_h.get("Seed AUCs", [])

# Bootstrap CI from stage_e
boot_ci = gbm_e["ci"]

# ── Assemble ledger ────────────────────────────────────────────────────────
ledger = {
    "run_id":          f"branch2_gbm_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "pipeline_branch": "Branch 2 — Clinical-only Prognosis (GBM Flagship)",
    "dataset":         "ISLES 2024 (processed tabular data)",
    "generated_at":    datetime.datetime.now().isoformat(),

    "cohort": {
        "n_total":      meta["n_total"],
        "n_included":   meta["n_included"],
        "n_excluded":   meta["n_excluded"],
        "exclusion_reason": "Missing 90-day mRS outcome",
        "outcome":      {
            "variable":  meta["target_binary"],
            "coding":    "1 = good (mRS 0-2), 0 = poor (mRS 3-6)",
            "positive":  "poor outcome mRS 3-6 = 1  [analysis flips: P(poor)=1-P(good)]",
            "n_poor":    stage_e["n_poor"],
            "n_good":    stage_e["n_good"],
        },
    },

    "predictors": {
        "n_baseline":        len(meta["baseline_predictors"]),
        "features":          meta["baseline_predictors"],
        "n_leakage_excluded": len(meta["leakage_predictors"]),
        "leakage_excluded":  meta["leakage_predictors"],
        "missing_in_predictors": 0,
    },

    "flagship": {
        "algorithm":    "GBM + SelectKBest(f_classif)",
        "library":      "sklearn.ensemble.GradientBoostingClassifier",
        "hyperparams":  d2_res["best_gbm_params"],
        "pipeline_pkl": d2_res["model_path"],
        "protocol":     "Stratified Repeated 10-Fold CV, 5 seeds = 50 evaluations",
        "seeds":        [42, 100, 2024, 777, 999],
        "note_selectkbest": "SelectKBest k='all' = passthrough; GBM self-regularises via max_depth & subsample",
    },

    "performance": {
        "primary_metric":     "Mean OOF AUC (mean-of-seed-AUCs, Stage H protocol)",
        "Mean_AUC":           primary_auc,
        "AUC_Std":            primary_auc_std,
        "Seed_AUCs":          seed_aucs_h if seed_aucs_h else d2_res.get("gbm_seed_aucs", []),
        "AUC_CI_95":          boot_ci,
        "AUC_CI_method":      "Bootstrap n=2000, subject-level resampling",
        "D2_OOF_pooled_AUC":  d2_res["gbm_pooled_auc"],
        "D2_vs_H_note":       "D4=seed-mean on wide OOF (0.7530); H=repeated SKF new models per fold (0.7495). Manuscript uses H.",
        "Accuracy_pct":       round(flagship_h["Accuracy (%)"], 2),
        "Brier_poor":         gbm_e["brier"],
        "F1":                 round(flagship_h["F1 Score"], 4),
        "calibration_intercept": gbm_e["calibration_intercept"],
        "calibration_slope":     gbm_e["calibration_slope"],
        "calibration_note":   "slope < 1 indicates overconfidence; calibration adjustment recommended before clinical use",
    },

    "clinical_comparisons": {
        "THRIVE": {
            "AUC":       stage_e["thrive"]["auc"],
            "AUC_CI_95": stage_e["thrive"]["ci"],
        },
        "SPAN-100": {
            "AUC":       stage_e["span100"]["auc"],
            "AUC_CI_95": stage_e["span100"]["ci"],
        },
        "paired_bootstrap_GBM_vs_THRIVE": {
            "delta_AUC": paired["delta"],
            "CI_95":     paired["ci"],
            "p_value":   paired["p"],
            "significant": paired["p"] < 0.05,
        },
        "paired_bootstrap_GBM_vs_SPAN100": {
            "delta_AUC": stage_e["paired_gbm_vs_span"]["delta"],
            "CI_95":     stage_e["paired_gbm_vs_span"]["ci"],
            "p_value":   stage_e["paired_gbm_vs_span"]["p"],
            "significant": stage_e["paired_gbm_vs_span"]["p"] < 0.05,
        },
        "reclassification_vs_THRIVE": {
            "NRI":        nri_idi["NRI"],
            "NRI_CI_95":  nri_idi["NRI_ci"],
            "NRI_event":  nri_idi["NRI_event"],
            "NRI_nonevent": nri_idi["NRI_nonevent"],
            "IDI":        nri_idi["IDI"],
            "IDI_CI_95":  nri_idi["IDI_ci"],
            "IDI_significant": nri_idi["IDI_ci"][0] > 0,
        },
    },

    "shap": {
        "method":       df_sh["shap_source"].iloc[0] if "shap_source" in df_sh.columns else "TreeSHAP on GBM",
        "n_nonzero":    int((df_sh["mean_abs_shap"] > 1e-6).sum()),
        "top_features": df_sh.head(12)[["feature", "mean_abs_shap", "mean_shap_good"]].to_dict("records"),
    },

    "literature_benchmark": {
        "protocol":    "Stratified Repeated 10-Fold CV, 5 seeds (same as flagship)",
        "n_comparators": len(df_h) - 1,
        "models":      [
            {k: v for k, v in row.items() if k != "Seed AUCs"} for row in df_h
        ],
        "weng_2021_excluded": not any("Weng" in r["Model"] for r in df_h),
        "proxy_note": "Literature models are proxy reimplementations; exact original datasets differ",
    },

    "outputs": {
        "oof_predictions":          os.path.join(OUTS, "stage_d2", "oof_predictions.csv"),
        "fold_manifest":            os.path.join(OUTS, "stage_d2", "fold_manifest.csv"),
        "pipeline_pkl":             os.path.join(OUTS, "stage_d2", "flagship_gbm_pipeline.pkl"),
        "stage_e_summary":          os.path.join(OUTS, "stage_e", "stage_e_summary.json"),
        "roc_curve":                os.path.join(OUTS, "stage_e", "roc_curve.png"),
        "calibration_plot":         os.path.join(OUTS, "stage_e", "calibration_plot.png"),
        "dca_results":              os.path.join(OUTS, "stage_e", "dca_results.csv"),
        "dca_plot":                 os.path.join(OUTS, "stage_e", "dca_plot.png"),
        "shap_csv":                 os.path.join(OUTS, "stage_f", "shap_feature_importance.csv"),
        "shap_plot":                os.path.join(OUTS, "stage_f", "shap_summary_plot.png"),
        "literature_comparison":    os.path.join(OUTS, "stage_h", "literature_comparison_summary.json"),
        "literature_plot":          os.path.join(OUTS, "stage_h", "literature_comparison_plot.png"),
    },

    "tripod_ai": {
        "source_of_data":           "ISLES 2024 (publicly available, multi-center ischemic stroke dataset)",
        "n_included":               meta["n_included"],
        "missing_data_handling":    "Complete-case analysis: outcome missing in 20/149 → excluded",
        "predictor_missing":        "0% missing in all 22 baseline predictors",
        "validation_type":          "Internal validation only (single-centre equivalent, ISLES 2024 source)",
        "cv_protocol":              "Stratified Repeated 10-Fold CV, 5 seeds = 50 fold evaluations",
        "leakage_prevention":       "4 intra-procedural variables excluded; all preprocessing inside Pipeline",
        "calibration_reported":     True,
        "dca_reported":             True,
        "nri_idi_reported":         True,
        "shap_reported":            True,
        "limitations":              [
            "Internal validation only; external validation required",
            "Single-centre equivalent dataset; generalisability uncertain",
            "Calibration slope 0.527: model overconfident; Platt scaling recommended pre-deployment",
            "20x10-fold sensitivity not performed (pre-specified scope: 5x10-fold)",
            "Ordinal mRS outcome not modelled (future work)",
        ],
    },
}

# ── Save ledger ────────────────────────────────────────────────────────────
ledger_path = os.path.join(OUT_DIR, "gbm_final_ledger.json")
with open(ledger_path, "w", encoding="utf-8") as f:
    json.dump(ledger, f, indent=2, ensure_ascii=False)
print(f"\nLedger saved: {ledger_path}")

# ── Human-readable markdown table ─────────────────────────────────────────
lines = [
    "# Branch 2 — Performance Summary\n",
    f"Generated: {ledger['generated_at'][:16]}  |  "
    f"Cohort n={meta['n_included']}  |  "
    f"Poor={stage_e['n_poor']}  Good={stage_e['n_good']}\n",
    "## Primary Model: GBM + SelectKBest  (5×10-fold, poor=1)\n",
    "| Metric | Value |",
    "|---|---|",
    f"| Mean OOF AUC (Stage H) | **{primary_auc} ± {primary_auc_std}** |",
    f"| Bootstrap 95% CI       | [{boot_ci[0]:.3f} – {boot_ci[1]:.3f}] |",
    f"| Accuracy               | {flagship_h['Accuracy (%)']:.2f}% |",
    f"| Brier Score (poor=1)   | {gbm_e['brier']} |",
    f"| Calibration slope      | {gbm_e['calibration_slope']} (< 1 = overconfident) |",
    f"| ΔAUC vs THRIVE         | +{paired['delta']:.3f} [{paired['ci'][0]:.3f}–{paired['ci'][1]:.3f}] p={paired['p']:.4f} |",
    f"| NRI vs THRIVE          | {nri_idi['NRI']} [{nri_idi['NRI_ci'][0]}–{nri_idi['NRI_ci'][1]}] |",
    f"| IDI vs THRIVE          | {nri_idi['IDI']} [{nri_idi['IDI_ci'][0]}–{nri_idi['IDI_ci'][1]}] |",
    "",
    "## Literature Benchmark\n",
    "| Model | Mean AUC | AUC Std | Acc (%) | F1 |",
    "|---|---|---|---|---|",
]
for row in sorted(df_h, key=lambda x: -x["Mean OOF AUC"]):
    tag = " ⬅ **OUR MODEL**" if "FLAGSHIP" in row["Model"] else ""
    lines.append(f"| {row['Model']}{tag} | {row['Mean OOF AUC']} | "
                 f"{row['AUC Std']} | {row['Accuracy (%)']} | {row['F1 Score']} |")

md_text = "\n".join(lines)
md_path = os.path.join(OUT_DIR, "performance_summary.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_text)
print(f"Markdown summary saved: {md_path}")
print(md_text[:600])

print(f"\nStage G complete!")
print(f"  Primary AUC  = {primary_auc} ± {primary_auc_std}")
print(f"  Bootstrap CI = {boot_ci}")
print(f"  ΔAUC THRIVE  = {paired['delta']:.3f}  p={paired['p']:.4f}")


def run_stage_g():
    """Entry point — all code runs at module level above."""
    pass


if __name__ == "__main__":
    run_stage_g()
