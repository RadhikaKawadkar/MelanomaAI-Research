"""
run_gap8_clinical_decision_utility.py
================================================================================
MelanomaAI Gap 8: Clinical Decision Utility & Decision Curve Analysis Engine

Coordinates with Gap 6 probability calibration to evaluate clinical decision
utility on the untouched, patient-grouped holdout test partition (ISIC 2020).

Key References:
  - Vickers & Elkin (2006). Decision curve analysis. Med Decis Making.
  - Van Calster et al. (2019). Reporting and interpreting DCA. Eur Urol.
  - Steyerberg et al. (2014). Assessing the performance of prediction models.
"""

from __future__ import annotations
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\DS_CP")
from clinical_decision_analysis import (
    calculate_net_benefit,
    compute_calibration_curve,
    fit_platt_scaling,
    apply_platt_scaling,
    fit_isotonic_regression,
    apply_isotonic_regression,
)

BASE_DIR = r"C:\DS_CP"
PREDS_PATH = os.path.join(BASE_DIR, "outputs", "clean_retraining", "test_predictions.csv")
MANIFEST_PATH = os.path.join(BASE_DIR, "outputs", "clean_retraining", "clean_split_manifest.csv")
GAP8_OUT_DIR = os.path.join(BASE_DIR, "outputs", "gap8_clinical_utility")
os.makedirs(GAP8_OUT_DIR, exist_ok=True)

def run_gap8_analysis():
    print("=" * 70)
    print("  MELANOMAAI GAP 8: CLINICAL DECISION UTILITY & DCA EVALUATION")
    print("=" * 70)

    # 1. Load and verify inputs
    print("\n[Step 1] Loading and verifying holdout test predictions...")
    assert os.path.exists(PREDS_PATH), f"Missing test predictions: {PREDS_PATH}"
    assert os.path.exists(MANIFEST_PATH), f"Missing split manifest: {MANIFEST_PATH}"

    manifest = pd.read_csv(MANIFEST_PATH)
    test_df = pd.read_csv(PREDS_PATH)

    calib_pts = set(manifest[manifest['partition'] == 'calib']['patient_id'])
    test_pts = set(test_df['patient_id'])
    assert len(calib_pts & test_pts) == 0, "FATAL: Patient leakage between calib and test!"
    print(f"  Verified: 0% patient leakage between calibration partition and test set.")
    print(f"  Test cohort: {len(test_df):,} images, {len(test_pts):,} patients, {int(test_df['target'].sum()):,} melanomas.")
    prevalence = float(test_df['target'].mean())
    print(f"  Empirical test prevalence: {prevalence * 100:.2f}%")

    y_true = test_df['target'].values
    p_raw = test_df['prob_raw'].values
    p_platt = test_df['prob_platt'].values
    p_iso = test_df['prob_iso'].values

    # 2. Threshold range definition
    # Justification:
    # Natural prevalence is 1.84%.
    # In melanoma screening, missing a melanoma (false negative) is catastrophic.
    # Screening risk thresholds range from 0.5% (willing to biopsy 199 to find 1)
    # up to 15% (willing to biopsy 6 to find 1). Above 20%, biopsy refusal is unethical.
    thresholds = np.array([
        0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05,
        0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30
    ])

    print("\n[Step 2] Computing Decision Curve Analysis across clinical thresholds...")
    N = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    dca_records = []

    for pt in thresholds:
        w = pt / (1.0 - pt)

        # Baseline strategies
        nb_all = (n_pos / N) - (n_neg / N) * w
        nb_none = 0.0

        # Model strategies
        # 1. Raw Uncalibrated
        pred_raw = (p_raw >= pt).astype(int)
        tp_raw = int(np.sum((y_true == 1) & (pred_raw == 1)))
        fp_raw = int(np.sum((y_true == 0) & (pred_raw == 1)))
        tn_raw = int(np.sum((y_true == 0) & (pred_raw == 0)))
        fn_raw = int(np.sum((y_true == 1) & (pred_raw == 0)))
        nb_raw = (tp_raw / N) - (fp_raw / N) * w

        # 2. Platt Calibrated
        pred_platt = (p_platt >= pt).astype(int)
        tp_platt = int(np.sum((y_true == 1) & (pred_platt == 1)))
        fp_platt = int(np.sum((y_true == 0) & (pred_platt == 1)))
        tn_platt = int(np.sum((y_true == 0) & (pred_platt == 0)))
        fn_platt = int(np.sum((y_true == 1) & (pred_platt == 0)))
        nb_platt = (tp_platt / N) - (fp_platt / N) * w

        # 3. Isotonic Calibrated
        pred_iso = (p_iso >= pt).astype(int)
        tp_iso = int(np.sum((y_true == 1) & (pred_iso == 1)))
        fp_iso = int(np.sum((y_true == 0) & (pred_iso == 1)))
        tn_iso = int(np.sum((y_true == 0) & (pred_iso == 0)))
        fn_iso = int(np.sum((y_true == 1) & (pred_iso == 0)))
        nb_iso = (tp_iso / N) - (fp_iso / N) * w

        # Net reduction in unnecessary biopsies per 100 patients vs Treat All
        red_raw = (nb_raw - nb_all) / w * 100.0 if w > 0 else 0.0
        red_platt = (nb_platt - nb_all) / w * 100.0 if w > 0 else 0.0
        red_iso = (nb_iso - nb_all) / w * 100.0 if w > 0 else 0.0

        # Sensitivity & Specificity
        sens_platt = tp_platt / (tp_platt + fn_platt) if (tp_platt + fn_platt) > 0 else 0.0
        spec_platt = tn_platt / (tn_platt + fp_platt) if (tn_platt + fp_platt) > 0 else 0.0

        # Number Needed to Biopsy (NNB) = (TP + FP) / TP
        nnb_platt = (tp_platt + fp_platt) / tp_platt if tp_platt > 0 else None

        # Raw biopsies spared per 100 patients (benign lesions correctly not referred)
        raw_spared_platt = (tn_platt / N) * 100.0
        # Missed melanomas per 100 patients
        fn_rate_per_100_platt = (fn_platt / N) * 100.0

        dca_records.append({
            "threshold": round(float(pt), 4),
            "nb_treat_all": round(float(nb_all), 6),
            "nb_treat_none": 0.0,
            "nb_model_raw": round(float(nb_raw), 6),
            "nb_model_platt": round(float(nb_platt), 6),
            "nb_model_iso": round(float(nb_iso), 6),
            "raw_biopsies_spared_per_100_platt": round(float(raw_spared_platt), 2),
            "missed_melanomas_per_100_platt": round(float(fn_rate_per_100_platt), 4),
            "net_biopsy_reduction_per_100_raw": round(float(red_raw), 2),
            "net_biopsy_reduction_per_100_platt": round(float(red_platt), 2),
            "net_biopsy_reduction_per_100_iso": round(float(red_iso), 2),
            "tp_platt": tp_platt,
            "fp_platt": fp_platt,
            "tn_platt": tn_platt,
            "fn_platt": fn_platt,
            "sensitivity_platt": round(float(sens_platt), 4),
            "specificity_platt": round(float(spec_platt), 4),
            "nnb_platt": round(float(nnb_platt), 2) if nnb_platt is not None else None,
            "referral_rate_platt": round(float((tp_platt + fp_platt) / N), 4)
        })

    # Save CSV table
    dca_df = pd.DataFrame(dca_records)
    table_path = os.path.join(GAP8_OUT_DIR, "dca_threshold_table.csv")
    dca_df.to_csv(table_path, index=False)
    print(f"  Saved DCA threshold table to: {table_path}")

    # 3. Bootstrap Confidence Intervals for Net Benefit (Image-Level and Patient-Clustered)
    print("\n[Step 3] Computing 1,000 bootstrap confidence intervals (Image-level and Patient-clustered)...")
    rng = np.random.RandomState(42)
    n_bootstraps = 1000
    boot_ci = {}

    patient_ids = test_df['patient_id'].values
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    patient_groups = test_df.groupby('patient_id').indices

    selected_eval_pts = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    for pt in selected_eval_pts:
        w = pt / (1.0 - pt)
        boot_nb_platt_img = []
        boot_nb_all_img = []
        boot_nb_platt_cluster = []
        boot_nb_all_cluster = []

        for _ in range(n_bootstraps):
            # Standard image-level resample
            idx = rng.randint(0, N, size=N)
            yt_b = y_true[idx]
            yp_b = p_platt[idx]
            tp_b = np.sum((yt_b == 1) & (yp_b >= pt))
            fp_b = np.sum((yt_b == 0) & (yp_b >= pt))
            boot_nb_platt_img.append((tp_b / N) - (fp_b / N) * w)
            boot_nb_all_img.append((np.sum(yt_b == 1) / N) - (np.sum(yt_b == 0) / N) * w)

            # Patient-clustered resample (accounts for multiple lesions per patient)
            samp_pts = rng.choice(unique_patients, size=n_patients, replace=True)
            samp_idx = np.concatenate([patient_groups[pid] for pid in samp_pts])
            n_samp = len(samp_idx)
            yt_c = y_true[samp_idx]
            yp_c = p_platt[samp_idx]
            tp_c = np.sum((yt_c == 1) & (yp_c >= pt))
            fp_c = np.sum((yt_c == 0) & (yp_c >= pt))
            boot_nb_platt_cluster.append((tp_c / n_samp) - (fp_c / n_samp) * w)
            boot_nb_all_cluster.append((np.sum(yt_c == 1) / n_samp) - (np.sum(yt_c == 0) / n_samp) * w)

        boot_ci[str(pt)] = {
            "threshold": pt,
            "image_level": {
                "nb_platt_mean": round(float(np.mean(boot_nb_platt_img)), 6),
                "nb_platt_ci_lower": round(float(np.percentile(boot_nb_platt_img, 2.5)), 6),
                "nb_platt_ci_upper": round(float(np.percentile(boot_nb_platt_img, 97.5)), 6),
                "nb_all_mean": round(float(np.mean(boot_nb_all_img)), 6),
                "nb_all_ci_lower": round(float(np.percentile(boot_nb_all_img, 2.5)), 6),
                "nb_all_ci_upper": round(float(np.percentile(boot_nb_all_img, 97.5)), 6)
            },
            "patient_clustered": {
                "nb_platt_mean": round(float(np.mean(boot_nb_platt_cluster)), 6),
                "nb_platt_ci_lower": round(float(np.percentile(boot_nb_platt_cluster, 2.5)), 6),
                "nb_platt_ci_upper": round(float(np.percentile(boot_nb_platt_cluster, 97.5)), 6),
                "nb_all_mean": round(float(np.mean(boot_nb_all_cluster)), 6),
                "nb_all_ci_lower": round(float(np.percentile(boot_nb_all_cluster, 2.5)), 6),
                "nb_all_ci_upper": round(float(np.percentile(boot_nb_all_cluster, 97.5)), 6)
            }
        }

    ci_path = os.path.join(GAP8_OUT_DIR, "dca_bootstrap_ci.json")
    with open(ci_path, "w") as f:
        json.dump(boot_ci, f, indent=2)
    print(f"  Saved DCA bootstrap confidence intervals to: {ci_path}")

    # 4. Generate Plot 1: Decision Curve Analysis (Net Benefit)
    print("\n[Step 4] Generating publication-grade DCA Net Benefit plot...")
    plt.figure(figsize=(9, 6.5))

    pts = dca_df["threshold"].values
    nb_model_platt = dca_df["nb_model_platt"].values
    nb_model_raw = dca_df["nb_model_raw"].values
    nb_model_iso = dca_df["nb_model_iso"].values
    nb_all = dca_df["nb_treat_all"].values
    nb_none = dca_df["nb_treat_none"].values

    # Clinical zones
    plt.axvspan(0.01, 0.05, color='green', alpha=0.08, label='Primary Care Screening Zone (1%–5%)')
    plt.axvspan(0.05, 0.15, color='amber' if 'amber' in matplotlib.colors.CSS4_COLORS else 'yellow', alpha=0.08, label='Specialist Referral Zone (5%–15%)')

    plt.plot(pts, nb_all, linestyle=':', color='gray', lw=2.2, label='Refer All Patients (Biopsy All)')
    plt.plot(pts, nb_none, linestyle='-', color='black', lw=1.8, label='Refer None (Biopsy None)')
    plt.plot(pts, nb_model_raw, linestyle='--', color='crimson', lw=1.8, label='MelanomaAI Raw (Uncalibrated)')
    plt.plot(pts, nb_model_iso, linestyle='-.', color='darkorange', lw=1.8, label='MelanomaAI Isotonic Calibrated')
    plt.plot(pts, nb_model_platt, linestyle='-', color='royalblue', lw=2.8, label='MelanomaAI Platt Calibrated (Optimal)')

    plt.xlim([0.005, 0.20])
    plt.ylim([-0.02, 0.022])
    plt.xlabel('Decision Threshold Probability ($p_t$)', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.title('MelanomaAI Decision Curve Analysis (ISIC 2020 Holdout Test Cohort, N=6,458, Prev=1.84%)', fontsize=12, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10, framealpha=0.95)
    plt.grid(True, alpha=0.3)

    plot1_path = os.path.join(GAP8_OUT_DIR, "dca_net_benefit_curves.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved DCA Net Benefit plot to: {plot1_path}")

    # 5. Generate Plot 2: Net Reduction in Unnecessary Biopsies
    print("\n[Step 5] Generating Net Reduction in Unnecessary Biopsies plot...")
    plt.figure(figsize=(9, 6.5))

    pts_red = dca_df[dca_df["threshold"] <= 0.15]["threshold"].values
    red_platt = dca_df[dca_df["threshold"] <= 0.15]["net_biopsy_reduction_per_100_platt"].values
    red_iso = dca_df[dca_df["threshold"] <= 0.15]["net_biopsy_reduction_per_100_iso"].values
    red_raw = dca_df[dca_df["threshold"] <= 0.15]["net_biopsy_reduction_per_100_raw"].values

    plt.axhline(0, color='black', lw=1.2, linestyle='--')
    plt.plot(pts_red, red_platt, marker='o', color='royalblue', lw=2.5, label='Platt Calibrated Model')
    plt.plot(pts_red, red_iso, marker='s', color='darkorange', lw=2.0, label='Isotonic Calibrated Model')
    plt.plot(pts_red, red_raw, marker='^', color='crimson', lw=1.8, label='Raw Uncalibrated Model')

    plt.axvspan(0.01, 0.05, color='green', alpha=0.08, label='Primary Care Screening Range (1%–5%)')

    plt.xlim([0.005, 0.15])
    plt.ylim([-5, 90])
    plt.xlabel('Clinical Threshold Probability ($p_t$)', fontsize=12)
    plt.ylabel('Net Reduction in Unnecessary Biopsies per 100 Patients', fontsize=12)
    plt.title('Net Reduction in Unnecessary Interventions vs. Biopsy-All', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10, framealpha=0.95)
    plt.grid(True, alpha=0.3)

    plot2_path = os.path.join(GAP8_OUT_DIR, "dca_biopsy_reduction_curves.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved Biopsy Reduction plot to: {plot2_path}")

    # 6. Save Full Metrics Summary JSON
    summary = {
        "gap": "Gap 8 — Clinical Decision Utility",
        "dataset": "ISIC_2020 Holdout Test Set",
        "total_test_samples": N,
        "melanoma_samples": n_pos,
        "benign_samples": n_neg,
        "prevalence": round(prevalence, 4),
        "primary_screening_threshold": 0.02,
        "metrics_at_pt_002": {
            "threshold": 0.02,
            "threshold_selection_rationale": "Exploratory threshold aligned with empirical cohort prevalence (~1.84%) and primary care 1:49 risk tolerance; not a prospectively validated cutoff.",
            "net_benefit_model_platt": round(float(dca_df[dca_df['threshold'] == 0.02]['nb_model_platt'].iloc[0]), 6),
            "net_benefit_treat_all": round(float(dca_df[dca_df['threshold'] == 0.02]['nb_treat_all'].iloc[0]), 6),
            "net_benefit_treat_none": 0.0,
            "raw_biopsies_spared_per_100": round(float(dca_df[dca_df['threshold'] == 0.02]['raw_biopsies_spared_per_100_platt'].iloc[0]), 2),
            "missed_melanomas_per_100": round(float(dca_df[dca_df['threshold'] == 0.02]['missed_melanomas_per_100_platt'].iloc[0]), 4),
            "net_biopsy_reduction_per_100": round(float(dca_df[dca_df['threshold'] == 0.02]['net_biopsy_reduction_per_100_platt'].iloc[0]), 2),
            "sensitivity": round(float(dca_df[dca_df['threshold'] == 0.02]['sensitivity_platt'].iloc[0]), 4),
            "specificity": round(float(dca_df[dca_df['threshold'] == 0.02]['specificity_platt'].iloc[0]), 4),
            "nnb": round(float(dca_df[dca_df['threshold'] == 0.02]['nnb_platt'].iloc[0]), 2),
            "referral_rate": round(float(dca_df[dca_df['threshold'] == 0.02]['referral_rate_platt'].iloc[0]), 4)
        },
        "metrics_at_pt_005": {
            "threshold": 0.05,
            "threshold_selection_rationale": "Secondary care / specialist triage threshold (1:19 harm-to-benefit ratio).",
            "net_benefit_model_platt": round(float(dca_df[dca_df['threshold'] == 0.05]['nb_model_platt'].iloc[0]), 6),
            "net_benefit_treat_all": round(float(dca_df[dca_df['threshold'] == 0.05]['nb_treat_all'].iloc[0]), 6),
            "net_benefit_treat_none": 0.0,
            "raw_biopsies_spared_per_100": round(float(dca_df[dca_df['threshold'] == 0.05]['raw_biopsies_spared_per_100_platt'].iloc[0]), 2),
            "missed_melanomas_per_100": round(float(dca_df[dca_df['threshold'] == 0.05]['missed_melanomas_per_100_platt'].iloc[0]), 4),
            "net_biopsy_reduction_per_100": round(float(dca_df[dca_df['threshold'] == 0.05]['net_biopsy_reduction_per_100_platt'].iloc[0]), 2),
            "sensitivity": round(float(dca_df[dca_df['threshold'] == 0.05]['sensitivity_platt'].iloc[0]), 4),
            "specificity": round(float(dca_df[dca_df['threshold'] == 0.05]['specificity_platt'].iloc[0]), 4),
            "nnb": round(float(dca_df[dca_df['threshold'] == 0.05]['nnb_platt'].iloc[0]), 2),
            "referral_rate": round(float(dca_df[dca_df['threshold'] == 0.05]['referral_rate_platt'].iloc[0]), 4)
        },
        "bootstrap_ci": boot_ci,
        "plots": {
            "dca_net_benefit_plot": plot1_path,
            "dca_biopsy_reduction_plot": plot2_path
        }
    }

    summary_path = os.path.join(GAP8_OUT_DIR, "gap8_dca_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved Gap 8 summary metrics to: {summary_path}")

    print("\n" + "=" * 70)
    print("  GAP 8 CLINICAL DECISION UTILITY IMPLEMENTATION COMPLETED!")
    print("=" * 70)

if __name__ == "__main__":
    run_gap8_analysis()
