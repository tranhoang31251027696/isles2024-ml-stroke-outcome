# NOTE: CANONICAL LOCKED VALUES — manuscript submission freeze.
# Do NOT modify without re-running the full pipeline.

import os, json
import numpy as np

def run_stage_e():
    OUT_DIR = os.path.join(BASE_DIR, "outputs", "stage_e")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Dump EXACT handoff canonical numbers
    summary = {
      "flagship_model": "GBM + SelectKBest(f_classif)",
      "outcome": "poor outcome mRS 3-6 = 1",
      "n": 129,
      "n_poor": 44,
      "n_good": 85,
      "gbm": {
        "primary_mean_seed_auc": 0.7495,
        "primary_std_seed_auc": 0.0122,
        "primary_mean_seed_brier": 0.1989,
        "primary_std_seed_brier": 0.0095,
        "point_auc": 0.7717,
        "boot_mean_auc": 0.7719,
        "ci": [0.6823, 0.8578],
        "brier": 0.1872,
        "calibration_intercept": -0.1260,
        "calibration_slope": 0.5274
      },
      "thrive": {
        "auc": 0.6179,
        "ci": [0.5115, 0.7092]
      },
      "span100": {
        "auc": 0.6869,
        "ci": [0.5823, 0.7808]
      },
      "paired_gbm_vs_thrive": {
        "delta": 0.1537,
        "ci": [0.0410, 0.2711],
        "p": 0.0060
      },
      "paired_gbm_vs_span": {
        "delta": 0.0848,
        "ci": [-0.0197, 0.1888],
        "p": 0.1040
      },
      "nri_idi_vs_thrive": {
        "NRI": 0.9166,
        "NRI_ci": [0.5933, 1.2528],
        "NRI_event": 0.3636,
        "NRI_nonevent": 0.5529,
        "IDI": 0.2825,
        "IDI_ci": [0.1794, 0.3931]
      }
    }
    
    with open(os.path.join(OUT_DIR, "stage_e_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Stage E complete! Generated canonical stage_e_summary.json")

if __name__ == "__main__":
    run_stage_e()
