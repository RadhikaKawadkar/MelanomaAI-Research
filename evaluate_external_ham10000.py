"""
evaluate_external_ham10000.py — Exploratory External Evaluation of MelanomaAI
============================================================================
Performs a rigorous, leak-free, external clinical utility evaluation of the
frozen EfficientNet-B3 model (melanoma_final.pth) on the HAM10000 benchmark.

Key Methodological Guarantees:
  1. Zero Lesion Overlap: Calibration and Test splits are strictly partitioned
     by lesion_id to eliminate duplicate lesion contamination.
  2. Independent Testing: Threshold selection and probability calibration
     (Platt scaling / Isotonic regression) are fitted exclusively on the
     calibration partition; test partition is evaluated strictly out-of-sample.
  3. Preprocessing Fidelity: Robustness layer (ITA estimation + inference CLAHE)
     and EfficientNet resolution matching are applied identically to inference.
  4. Decision Curve Analysis: Net Benefit evaluated with and without clinical
     prevalence adjustment (2% and 5% primary-care screening benchmarks).
  5. Uncertainty Quantification: 95% bootstrap percentile confidence intervals.
"""

from __future__ import annotations
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    roc_curve, f1_score, precision_score, recall_score,
    average_precision_score, precision_recall_curve,
    brier_score_loss
)

# Project imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from backend.model import build_model
from risk_engine import robustness_preprocess, estimate_ita, get_skin_tone
from split_dataset import partition_dataset_grouped
from clinical_decision_analysis import (
    compute_calibration_curve, fit_platt_scaling, apply_platt_scaling,
    fit_isotonic_regression, apply_isotonic_regression,
    calculate_net_benefit, bootstrap_metrics
)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
CHECKPOINT_PATH = Path("D:/EDI - Copy/models/melanoma_final.pth")
METADATA_PATH   = Path("D:/EDI - Copy/data/raw/HAM10000_metadata.csv")
IMAGES_DIR_1    = Path("D:/EDI - Copy/data/raw/HAM10000_images_part_1")
IMAGES_DIR_2    = Path("D:/EDI - Copy/data/raw/HAM10000_images_part_2")
OUTPUT_DIR      = ROOT / "outputs" / "clinical_utility_external"

IMG_SIZE   = 384
BATCH_SIZE = 32
SEED       = 42
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Visual styling
BG = "#0A101E"; BG2 = "#060D1A"; CYAN = "#00D4FF"; RED = "#FF3B5C"; GREEN = "#00E5A0"; WARN = "#FFB800"
TEXT = "#CBD5E1"; MUTED = "#64748B"

def plt_dark():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG2,
        "axes.edgecolor": "#1E293B", "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
        "grid.color": "#1E293B", "grid.alpha": 0.4, "font.family": "sans-serif",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.facecolor": BG, "legend.edgecolor": "#1E293B", "legend.labelcolor": TEXT,
    })


# ─── DATASET & DATALOADER ─────────────────────────────────────────────────────
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

class HAM10000Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_lookup: Dict[str, Path]):
        self.df = df.reset_index(drop=True)
        self.image_lookup = image_lookup

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_id = row["image_id"]
        img_path = self.image_lookup[img_id]

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Preprocessing with robustness layer (ITA & CLAHE)
        processed, ita, tone, rel, enhanced = robustness_preprocess(img_rgb)
        tensor = val_transform(image=processed)["image"]
        label = int(row["target"])

        return tensor, label, float(ita), img_id, str(row["lesion_id"])


def build_image_lookup() -> Dict[str, Path]:
    lookup = {}
    for folder in [IMAGES_DIR_1, IMAGES_DIR_2]:
        for p in folder.glob("*.jpg"):
            lookup[p.stem] = p
    return lookup


# ─── MODEL INFERENCE ──────────────────────────────────────────────────────────
def run_model_inference(model: nn.Module, loader: DataLoader) -> Dict[str, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []
    all_itas = []
    all_image_ids = []
    all_lesion_ids = []

    with torch.no_grad():
        for tensors, labels, itas, img_ids, lesion_ids in tqdm(loader, desc="Inference", unit="batch"):
            tensors = tensors.to(DEVICE)
            with torch.amp.autocast("cuda") if torch.cuda.is_available() else torch.no_grad():
                preds = model(tensors).squeeze()
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)

            all_probs.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_itas.extend(itas.numpy().tolist())
            all_image_ids.extend(img_ids)
            all_lesion_ids.extend(lesion_ids)

    return {
        "probs": np.array(all_probs, dtype=float),
        "labels": np.array(all_labels, dtype=int),
        "itas": np.array(all_itas, dtype=float),
        "image_ids": np.array(all_image_ids),
        "lesion_ids": np.array(all_lesion_ids),
    }


# ─── PLOTTING ROUTINES ────────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, out_path: Path):
    plt_dark()
    fig, ax = plt.subplots(figsize=(6, 5))
    tn, fp, fn, tp = cm.ravel()
    vals = [[tn, fp], [fn, tp]]
    colors = [[GREEN, RED], [WARN, CYAN]]
    lbls = [["TN (Benign)", "FP (False Alarm)"], ["FN (Missed Melanoma)", "TP (Detected)"]]

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{vals[i][j]}\n{lbls[i][j]}", ha="center", va="center",
                    color=colors[i][j], fontsize=12, fontweight="bold")

    ax.imshow([[0, 1], [2, 3]], cmap="Blues", alpha=0.15)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Benign", "Predicted Malignant"])
    ax.set_yticklabels(["Actual Benign", "Actual Malignant"])
    ax.set_title("Test Set Confusion Matrix", color="white", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, auc: float, out_path: Path):
    plt_dark()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.fill_between(fpr, tpr, alpha=0.12, color=CYAN)
    ax.plot(fpr, tpr, color=CYAN, lw=2.5, label=f"EfficientNet-B3 (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="#334155", lw=1.5, label="Random Guess")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("External Test ROC Curve (HAM10000)", color="white", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)


def plot_calibration(raw_curve: Dict[str, Any], cal_curve: Dict[str, Any], out_path: Path):
    plt_dark()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#475569", lw=1.5, label="Perfect Calibration")

    ax.plot(raw_curve["bin_centers"], raw_curve["empirical_probabilities"], "o-",
            color=RED, lw=2, label=f"Uncalibrated (ECE = {raw_curve['expected_calibration_error']:.3f})")
    ax.plot(cal_curve["bin_centers"], cal_curve["empirical_probabilities"], "s-",
            color=GREEN, lw=2, label=f"Platt Calibrated (ECE = {cal_curve['expected_calibration_error']:.3f})")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Empirical True Fraction")
    ax.set_title("Reliability Diagram (Test Partition)", color="white", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)


def plot_decision_curves(dca_cohort: Dict[str, Any], dca_prev2: Dict[str, Any], out_path: Path):
    plt_dark()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Cohort-prevalence DCA
    ax1 = axes[0]
    th = dca_cohort["thresholds"]
    ax1.plot(th, dca_cohort["net_benefit_model"], color=CYAN, lw=2.5, label="MelanomaAI Model")
    ax1.plot(th, dca_cohort["net_benefit_treat_all"], color=WARN, lw=1.8, ls="--", label="Refer All")
    ax1.axhline(0, color="#475569", lw=1.5, ls=":", label="Refer None")
    ax1.set_xlabel("Threshold Probability (p_t)")
    ax1.set_ylabel("Net Benefit")
    ax1.set_title(f"DCA — Raw Cohort Prevalence ({dca_cohort['sample_prevalence']:.1%})", color="white", fontsize=11, fontweight="bold")
    ax1.set_ylim([-0.05, max(dca_cohort["net_benefit_model"]) * 1.25])
    ax1.legend(loc="upper right")

    # 2. Adjusted-prevalence DCA (2% primary care screening prevalence)
    ax2 = axes[1]
    ax2.plot(th, dca_prev2["net_benefit_model"], color=GREEN, lw=2.5, label="MelanomaAI (Adjusted)")
    ax2.plot(th, dca_prev2["net_benefit_treat_all"], color=WARN, lw=1.8, ls="--", label="Refer All")
    ax2.axhline(0, color="#475569", lw=1.5, ls=":", label="Refer None")
    ax2.axvspan(0.02, 0.08, color="#00D4FF", alpha=0.08, label="PCP Triage Zone (2-8%)")
    ax2.set_xlabel("Threshold Probability (p_t)")
    ax2.set_ylabel("Net Benefit (per patient)")
    ax2.set_title("DCA — Primary Care Adjusted (2% Prevalence)", color="white", fontsize=11, fontweight="bold")
    ax2.set_ylim([-0.01, max(dca_prev2["net_benefit_model"]) * 1.35 if max(dca_prev2["net_benefit_model"]) > 0 else 0.05])
    ax2.legend(loc="upper right")

    plt.suptitle("Decision Curve Analysis (DCA)", color="white", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)


# ─── MAIN EVALUATION WORKFLOW ─────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("  MelanomaAI — Exploratory External Evaluation (HAM10000)")
    print(f"  Device: {DEVICE}")
    print(f"  Outputs: {OUTPUT_DIR}")
    print("=" * 70)

    # 1. Model Loading
    print("\n[1/7] Loading frozen model checkpoint...")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")
    model = build_model()
    ck = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE)
    state_dict = ck.get("model_state_dict", ck)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE).eval()
    print(f"   [OK] Model loaded: {CHECKPOINT_PATH.name} ({len(state_dict)} state_dict keys)")

    # 2. Metadata & Partitions
    print("\n[2/7] Preparing HAM10000 metadata & leak-free lesion partitions...")
    df_meta = pd.read_csv(METADATA_PATH)
    image_lookup = build_image_lookup()
    print(f"   Total metadata rows: {len(df_meta)}")
    print(f"   Available disk images: {len(image_lookup)}")

    # Target definition: Binary melanoma (mel = 1) vs non-melanoma (all other = 0)
    df_meta["target"] = (df_meta["dx"] == "mel").astype(int)
    print(f"   Melanoma cases: {df_meta['target'].sum()} | Non-melanoma: {(df_meta['target'] == 0).sum()}")

    # 50% Calibration / 50% Test split strictly grouped by lesion_id
    splits = partition_dataset_grouped(
        df_meta,
        group_col="lesion_id",
        label_col="target",
        positive_label=1,
        train_ratio=0.0,
        calib_ratio=0.50,
        test_ratio=0.50,
        seed=SEED,
    )
    df_calib = splits["calibration"].copy()
    df_test = splits["test"].copy()

    calib_lesions = set(df_calib["lesion_id"])
    test_lesions = set(df_test["lesion_id"])
    assert len(calib_lesions.intersection(test_lesions)) == 0, "Lesion leakage detected!"
    print(f"   Calibration partition: {len(df_calib)} images across {len(calib_lesions)} unique lesions")
    print(f"   Test partition:        {len(df_test)} images across {len(test_lesions)} unique lesions")
    print(f"   [OK] Verified 0% lesion overlap between calibration and test partitions.")

    calib_loader = DataLoader(HAM10000Dataset(df_calib, image_lookup), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(HAM10000Dataset(df_test, image_lookup), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. Calibration Inference & Fitting
    print("\n[3/7] Running inference on calibration partition...")
    calib_res = run_model_inference(model, calib_loader)
    calib_probs = calib_res["probs"]
    calib_labels = calib_res["labels"]

    print("   Fitting Platt scaling & Isotonic regression on calibration split...")
    platt_model = fit_platt_scaling(calib_probs, calib_labels)
    isotonic_model = fit_isotonic_regression(calib_probs, calib_labels)

    # Determine optimal threshold on calibration split (maximizing F1)
    prec_c, rec_c, thresholds_c = precision_recall_curve(calib_labels, calib_probs)
    f1_scores_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-9)
    best_idx = np.argmax(f1_scores_c)
    f1_thresh = float(thresholds_c[best_idx]) if best_idx < len(thresholds_c) else 0.50
    print(f"   Calibration F1-optimal threshold: {f1_thresh:.4f} (Max Calibration F1 = {f1_scores_c[best_idx]:.4f})")

    # 4. Out-of-Sample Test Inference
    print("\n[4/7] Running out-of-sample inference on held-out test partition...")
    test_res = run_model_inference(model, test_loader)
    test_raw_probs = test_res["probs"]
    test_labels    = test_res["labels"]
    test_itas      = test_res["itas"]

    # Apply pre-fitted calibration models
    test_platt_probs = apply_platt_scaling(platt_model, test_raw_probs)
    test_iso_probs   = apply_isotonic_regression(isotonic_model, test_raw_probs)

    # 5. Metrics & Uncertainty Estimation
    print("\n[5/7] Computing test metrics, bootstrap intervals, and calibration curves...")
    raw_calib_eval   = compute_calibration_curve(test_labels, test_raw_probs)
    platt_calib_eval = compute_calibration_curve(test_labels, test_platt_probs)
    iso_calib_eval   = compute_calibration_curve(test_labels, test_iso_probs)

    # Binary metrics at default 0.50 threshold (Uncalibrated)
    preds_default = (test_raw_probs >= 0.50).astype(int)
    cm_default = confusion_matrix(test_labels, preds_default, labels=[0, 1])

    # Binary metrics at Platt-calibrated 0.50 threshold
    preds_platt = (test_platt_probs >= 0.50).astype(int)
    cm_platt = confusion_matrix(test_labels, preds_platt, labels=[0, 1])

    # Binary metrics at calibration-tuned F1 threshold
    preds_tuned = (test_raw_probs >= f1_thresh).astype(int)
    cm_tuned = confusion_matrix(test_labels, preds_tuned, labels=[0, 1])

    # Bootstrap 95% Confidence Intervals
    boot_stats = bootstrap_metrics(test_labels, test_raw_probs, threshold=0.50, n_bootstraps=1000, seed=SEED)
    auc_val = roc_auc_score(test_labels, test_raw_probs)

    summary_metrics = {
        "evaluation_target": "HAM10000 External Cohort (mel vs all non-mel)",
        "total_test_samples": int(len(test_labels)),
        "melanoma_count": int(test_labels.sum()),
        "non_melanoma_count": int((test_labels == 0).sum()),
        "sample_prevalence": float(np.mean(test_labels)),
        "uncalibrated_default_050": {
            "threshold": 0.50,
            "accuracy": float(accuracy_score(test_labels, preds_default)),
            "sensitivity": float(recall_score(test_labels, preds_default)),
            "specificity": float(cm_default[0, 0] / (cm_default[0, 0] + cm_default[0, 1])),
            "precision": float(precision_score(test_labels, preds_default, zero_division=0)),
            "f1_score": float(f1_score(test_labels, preds_default, zero_division=0)),
            "auc_roc": float(auc_val),
            "brier_score": raw_calib_eval["brier_score"],
            "expected_calibration_error": raw_calib_eval["expected_calibration_error"],
            "confusion_matrix": cm_default.tolist(),
        },
        "platt_calibrated_050": {
            "threshold": 0.50,
            "accuracy": float(accuracy_score(test_labels, preds_platt)),
            "sensitivity": float(recall_score(test_labels, preds_platt)),
            "specificity": float(cm_platt[0, 0] / (cm_platt[0, 0] + cm_platt[0, 1])),
            "precision": float(precision_score(test_labels, preds_platt, zero_division=0)),
            "f1_score": float(f1_score(test_labels, preds_platt, zero_division=0)),
            "auc_roc": float(roc_auc_score(test_labels, test_platt_probs)),
            "brier_score": platt_calib_eval["brier_score"],
            "expected_calibration_error": platt_calib_eval["expected_calibration_error"],
            "confusion_matrix": cm_platt.tolist(),
        },
        "tuned_threshold": {
            "threshold": f1_thresh,
            "accuracy": float(accuracy_score(test_labels, preds_tuned)),
            "sensitivity": float(recall_score(test_labels, preds_tuned)),
            "specificity": float(cm_tuned[0, 0] / (cm_tuned[0, 0] + cm_tuned[0, 1])),
            "precision": float(precision_score(test_labels, preds_tuned, zero_division=0)),
            "f1_score": float(f1_score(test_labels, preds_tuned, zero_division=0)),
            "confusion_matrix": cm_tuned.tolist(),
        },
        "bootstrap_95_ci": boot_stats,
    }

    # 6. Decision Curve Analysis (DCA)
    print("\n[6/7] Computing Decision Curve Analysis (DCA)...")
    dca_thresholds = np.linspace(0.01, 0.45, 45)
    dca_cohort = calculate_net_benefit(test_labels, test_platt_probs, dca_thresholds)
    dca_prev2  = calculate_net_benefit(test_labels, test_platt_probs, dca_thresholds, target_prevalence=0.02)
    dca_prev5  = calculate_net_benefit(test_labels, test_platt_probs, dca_thresholds, target_prevalence=0.05)

    dca_summary = {
        "cohort_prevalence": dca_cohort,
        "primary_care_2pct_prevalence": dca_prev2,
        "specialist_5pct_prevalence": dca_prev5,
    }

    # Subgroup analysis by skin tone
    light_mask = test_itas > 28
    dark_mask  = test_itas <= 28
    subgroups = {
        "light_skin_ITA_gt_28": {
            "count": int(light_mask.sum()),
            "melanoma_count": int(test_labels[light_mask].sum()),
            "auc": float(roc_auc_score(test_labels[light_mask], test_raw_probs[light_mask])) if light_mask.sum() > 5 else None,
            "sensitivity": float(recall_score(test_labels[light_mask], preds_default[light_mask])) if light_mask.sum() > 5 else None,
            "specificity": float(np.mean(preds_default[light_mask][test_labels[light_mask] == 0] == 0)) if (test_labels[light_mask] == 0).sum() > 0 else None,
        },
        "dark_skin_ITA_le_28": {
            "count": int(dark_mask.sum()),
            "melanoma_count": int(test_labels[dark_mask].sum()),
            "auc": float(roc_auc_score(test_labels[dark_mask], test_raw_probs[dark_mask])) if dark_mask.sum() > 5 else None,
            "sensitivity": float(recall_score(test_labels[dark_mask], preds_default[dark_mask])) if dark_mask.sum() > 5 else None,
            "specificity": float(np.mean(preds_default[dark_mask][test_labels[dark_mask] == 0] == 0)) if (test_labels[dark_mask] == 0).sum() > 0 else None,
        }
    }
    summary_metrics["skin_tone_subgroups"] = subgroups

    # Save data artifacts
    print("\n[7/7] Generating and saving plots and CSV/JSON deliverables...")
    with open(OUTPUT_DIR / "test_metrics_summary.json", "w") as f:
        json.dump(summary_metrics, f, indent=2)

    with open(OUTPUT_DIR / "dca_curves.json", "w") as f:
        json.dump(dca_summary, f, indent=2)

    # Export sample-level predictions
    df_preds = pd.DataFrame({
        "image_id": test_res["image_ids"],
        "lesion_id": test_res["lesion_ids"],
        "true_label": test_labels,
        "uncalibrated_prob": np.round(test_raw_probs, 5),
        "platt_calibrated_prob": np.round(test_platt_probs, 5),
        "isotonic_prob": np.round(test_iso_probs, 5),
        "ita_angle": np.round(test_itas, 2),
    })
    df_preds.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    # Render plots
    plot_confusion_matrix(cm_default, OUTPUT_DIR / "confusion_matrix_test.png")
    plot_roc_curve(test_labels, test_raw_probs, auc_val, OUTPUT_DIR / "roc_curve_test.png")
    plot_calibration(raw_calib_eval, platt_calib_eval, OUTPUT_DIR / "calibration_curve_test.png")
    plot_decision_curves(dca_cohort, dca_prev2, OUTPUT_DIR / "decision_curve_analysis_test.png")

    print("\n" + "=" * 70)
    print("  [OK] EXTERNAL EVALUATION COMPLETED SUCCESSFULLY")
    print(f"  Test Samples:      {len(test_labels)} (Melanoma: {test_labels.sum()}, Non-Melanoma: {(test_labels == 0).sum()})")
    print(f"  Test AUC-ROC:      {auc_val:.4f} (95% CI: [{boot_stats['auc_roc']['ci_lower']:.4f}, {boot_stats['auc_roc']['ci_upper']:.4f}])")
    print(f"  Sensitivity @0.50: {summary_metrics['uncalibrated_default_050']['sensitivity']*100:.2f}%")
    print(f"  Specificity @0.50: {summary_metrics['uncalibrated_default_050']['specificity']*100:.2f}%")
    print(f"  Uncalibrated ECE:  {raw_calib_eval['expected_calibration_error']:.4f} | Brier: {raw_calib_eval['brier_score']:.4f}")
    print(f"  Calibrated ECE:    {platt_calib_eval['expected_calibration_error']:.4f} | Brier: {platt_calib_eval['brier_score']:.4f}")
    print(f"  Deliverables saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
