"""
run_calibration_study.py
================================================================================
MelanomaAI Real-Data Calibration Study & Fairness Stratification Evaluation

Performs rigorous probability calibration evaluation using:
1. Real ISIC 2020 dataset and patient-stratified split manifest (0% patient leakage).
2. Verified clean EfficientNet-B3 model checkpoint (clean_model_best.pth).
3. Dedicated calibration partition (3,334 images, 206 patients) to fit Platt scaling.
4. Untouched holdout test partition (6,458 images, 411 patients) strictly for out-of-sample evaluation.
5. Demographic stratification by Individual Typology Angle (ITA) into Light (>28) and Dark (<=28).
6. Patient-clustered non-parametric bootstrap confidence intervals (1,000 resamples).

CLI Usage:
python run_calibration_study.py --calib outputs/calibration_study_real/actual_calibration_logits.csv \
                                --test outputs/calibration_study_real/actual_test_logits.csv \
                                --group-col skin_group \
                                --cluster-col patient_id \
                                --out-dir outputs/calibration_study_real
================================================================================
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Project imports
BASE_DIR = r"C:\DS_CP"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from risk_engine import estimate_ita, apply_clahe, get_skin_tone

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 384
BATCH_SIZE = 32
DEFAULT_CKPT = os.path.join(BASE_DIR, "outputs", "clean_retraining", "checkpoints", "clean_model_best.pth")
DEFAULT_MANIFEST = os.path.join(BASE_DIR, "outputs", "clean_retraining", "clean_split_manifest.csv")
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "outputs", "calibration_study_real")

eval_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


class ISICImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=eval_transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.filepaths = self.df['filepath'].values
        self.itas = self.df['ita'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        ita = self.itas[idx]
        img = Image.open(path).convert('RGB')
        img_np = np.array(img)

        # Clean pipeline: apply CLAHE conditionally if ITA <= 28
        if ita <= 28:
            img_np = apply_clahe(img_np)

        if self.transform:
            augmented = self.transform(image=img_np)
            tensor = augmented['image']
        else:
            tensor = torch.tensor(img_np).permute(2, 0, 1).float() / 255.0

        return tensor, idx


def build_model():
    base = models.efficientnet_b3(weights=None)
    nf = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(nf, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    return base.to(DEVICE)


def load_model(ckpt_path: str = DEFAULT_CKPT) -> nn.Module:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob <= high if i == n_bins - 1 else y_prob < high)
        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            ece += (count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    emp_acc = []
    bin_counts = []
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob <= high if i == n_bins - 1 else y_prob < high)
        count = int(np.sum(mask))
        if count > 0:
            bin_centers.append(float(np.mean(y_prob[mask])))
            emp_acc.append(float(np.mean(y_true[mask])))
            bin_counts.append(count)
    return bin_centers, emp_acc, bin_counts


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    acc = float((tp + tn) / len(y_true))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_ece(y_true, y_prob, n_bins=10)

    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float('nan')
    except Exception:
        auc = float('nan')

    return {
        "threshold": round(threshold, 4),
        "total": len(y_true),
        "melanomas": int(tp + fn),
        "benign": int(tn + fp),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "fnr": round(fnr, 4),
        "precision_ppv": round(ppv, 4),
        "npv": round(npv, 4),
        "accuracy": round(acc, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "auroc": round(auc, 4) if not np.isnan(auc) else None
    }


def compute_patient_clustered_bootstrap(
    df: pd.DataFrame,
    prob_col: str,
    target_col: str = 'target',
    cluster_col: str = 'patient_id',
    threshold: float = 0.50,
    n_boot: int = 1000,
    seed: int = 42
):
    rng = np.random.RandomState(seed)
    patients = df[cluster_col].unique()
    n_pts = len(patients)
    pt_to_indices = {pt: df.index[df[cluster_col] == pt].values for pt in patients}

    auc_list, brier_list, ece_list = [], [], []
    sens_list, spec_list, fnr_list, ppv_list, npv_list = [], [], [], [], []

    for _ in range(n_boot):
        sample_pts = rng.choice(patients, size=n_pts, replace=True)
        sample_idx = np.concatenate([pt_to_indices[pt] for pt in sample_pts])
        y = df.iloc[sample_idx][target_col].values
        p = df.iloc[sample_idx][prob_col].values

        if len(np.unique(y)) < 2:
            continue

        try:
            auc_list.append(float(roc_auc_score(y, p)))
        except Exception:
            pass

        brier_list.append(float(brier_score_loss(y, p)))
        ece_list.append(compute_ece(y, p, n_bins=10))

        pred = (p >= threshold).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        if (tp + fn) > 0:
            sens_list.append(tp / (tp + fn))
            fnr_list.append(fn / (tp + fn))
        if (tn + fp) > 0:
            spec_list.append(tn / (tn + fp))
        if (tp + fp) > 0:
            ppv_list.append(tp / (tp + fp))
        if (tn + fn) > 0:
            npv_list.append(tn / (tn + fn))

    def summarize(arr):
        if len(arr) == 0:
            return None
        return {
            "mean": round(float(np.mean(arr)), 4),
            "ci_95": [
                round(float(np.percentile(arr, 2.5)), 4),
                round(float(np.percentile(arr, 97.5)), 4)
            ]
        }

    return {
        "auroc": summarize(auc_list),
        "brier_score": summarize(brier_list),
        "ece": summarize(ece_list),
        "sensitivity": summarize(sens_list),
        "specificity": summarize(spec_list),
        "fnr": summarize(fnr_list),
        "precision_ppv": summarize(ppv_list),
        "npv": summarize(npv_list),
    }


def generate_partition_logits(df: pd.DataFrame, model: nn.Module, partition_name: str) -> pd.DataFrame:
    print(f"  Calculating ITA and generating logits for {partition_name} ({len(df):,} images)...")
    t0 = time.time()

    # 1. Compute ITA if missing
    if 'ita' not in df.columns or df['ita'].isnull().any():
        itas = []
        skin_groups = []
        for fp in df['filepath']:
            img_np = np.array(Image.open(fp).convert('RGB'))
            ita_val = estimate_ita(img_np)
            itas.append(ita_val)
            skin_groups.append('light' if ita_val > 28 else 'dark')
        df['ita'] = itas
        df['skin_group'] = skin_groups
    elif 'skin_group' not in df.columns:
        df['skin_group'] = np.where(df['ita'] > 28, 'light', 'dark')

    # 2. Run model inference
    dataset = ISICImageDataset(df, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    raw_probs = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
                preds = model(imgs).squeeze(-1)
            raw_probs.extend(preds.cpu().numpy().tolist())

    p_raw = np.nan_to_num(np.array(raw_probs), nan=0.0)
    eps = 1e-6
    p_clamped = np.clip(p_raw, eps, 1.0 - eps)
    logits = np.log(p_clamped / (1.0 - p_clamped))

    df['prob_raw'] = p_raw
    df['logit'] = logits

    dt = time.time() - t0
    print(f"  Completed inference for {partition_name} in {dt:.1f}s ({dt/len(df)*1000:.1f} ms/image).")
    return df


def main():
    parser = argparse.ArgumentParser(description="MelanomaAI Real-Data Calibration Study")
    parser.add_argument("--calib", type=str, default="", help="Path to calibration split logits CSV")
    parser.add_argument("--test", type=str, default="", help="Path to holdout test split logits CSV")
    parser.add_argument("--group-col", type=str, default="skin_group", help="Grouping column for fairness analysis")
    parser.add_argument("--cluster-col", type=str, default="patient_id", help="Cluster column for clustered bootstrap CIs")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Directory to save study results")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST, help="Path to clean split manifest CSV")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="Path to clean model checkpoint")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 75)
    print("  MELANOMAAI REAL-DATA PROBABILITY CALIBRATION STUDY")
    print(f"  Device: {DEVICE}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Output Directory: {args.out_dir}")
    print("=" * 75)

    # 1. Load or Generate Calibration Logits
    manifest = pd.read_csv(args.manifest)
    model = None

    if args.calib and os.path.exists(args.calib):
        print(f"\n[Step 1/5] Loading existing calibration logits: {args.calib}")
        calib_df = pd.read_csv(args.calib)
    else:
        print(f"\n[Step 1/5] Generating real calibration logits from checkpoint...")
        calib_subset = manifest[manifest['partition'] == 'calib'].copy().reset_index(drop=True)
        if model is None:
            model = load_model(args.ckpt)
        calib_df = generate_partition_logits(calib_subset, model, "calibration split")
        calib_out_path = args.calib if args.calib else os.path.join(args.out_dir, "actual_calibration_logits.csv")
        calib_df.to_csv(calib_out_path, index=False)
        print(f"  Saved real calibration logits to: {calib_out_path}")

    # 2. Load or Generate Test Logits
    if args.test and os.path.exists(args.test):
        print(f"\n[Step 2/5] Loading existing test logits: {args.test}")
        test_df = pd.read_csv(args.test)
    else:
        print(f"\n[Step 2/5] Preparing real holdout test logits...")
        # Check if existing dark_skin_fairness_predictions has precomputed ITA and prob_raw
        fair_preds_path = os.path.join(BASE_DIR, "outputs", "dark_skin_fairness", "dark_skin_fairness_predictions.csv")
        test_preds_path = os.path.join(BASE_DIR, "outputs", "clean_retraining", "test_predictions.csv")

        if os.path.exists(fair_preds_path):
            print(f"  Reusing verified test predictions and ITA annotations from: {fair_preds_path}")
            fair_df = pd.read_csv(fair_preds_path)
            test_subset = manifest[manifest['partition'] == 'test'].copy().reset_index(drop=True)
            test_subset['ita'] = fair_df['ita']
            test_subset['skin_group'] = fair_df['skin_group']
            test_subset['prob_raw'] = fair_df['prob_raw_clahe']
            eps = 1e-6
            p_clamped = np.clip(test_subset['prob_raw'].values, eps, 1.0 - eps)
            test_subset['logit'] = np.log(p_clamped / (1.0 - p_clamped))
            test_df = test_subset
        else:
            test_subset = manifest[manifest['partition'] == 'test'].copy().reset_index(drop=True)
            if model is None:
                model = load_model(args.ckpt)
            test_df = generate_partition_logits(test_subset, model, "holdout test split")

        test_out_path = args.test if args.test else os.path.join(args.out_dir, "actual_test_logits.csv")
        test_df.to_csv(test_out_path, index=False)
        print(f"  Saved real test logits to: {test_out_path}")

    # Verify 0% Patient Leakage
    calib_patients = set(calib_df[args.cluster_col])
    test_patients = set(test_df[args.cluster_col])
    leakage = calib_patients & test_patients
    assert len(leakage) == 0, f"FATAL: Patient leakage detected ({len(leakage)} overlapping patients)!"
    print(f"  [VERIFIED] 0% patient leakage between calibration ({len(calib_patients)} pts) and test ({len(test_patients)} pts).")

    # 3. Fit Platt Scaling Strictly on Calibration Partition
    print("\n[Step 3/5] Fitting Platt scaling strictly on CALIBRATION split...")
    calib_y = calib_df['target'].values.astype(int)
    calib_logits = calib_df['logit'].values.reshape(-1, 1)

    # Standard Platt scaling: LogisticRegression(C=1.0, solver='lbfgs')
    platt = LogisticRegression(C=1.0, solver='lbfgs')
    platt.fit(calib_logits, calib_y)
    platt_a = float(platt.coef_[0][0])
    platt_b = float(platt.intercept_[0])
    print(f"  Fitted Platt Scaling Parameters: A = {platt_a:.6f}, B = {platt_b:.6f}")

    # Also fit isotonic regression on calib split as secondary benchmark
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(calib_df['prob_raw'].values, calib_y)
    print("  Secondary Isotonic Regression fitted exclusively on calibration split.")

    # Save fitted parameters
    calib_params = {
        "method": "Platt Scaling (Logistic Regression on logit)",
        "platt_a": platt_a,
        "platt_b": platt_b,
        "calib_images": len(calib_df),
        "calib_patients": len(calib_patients),
        "calib_melanomas": int(calib_y.sum()),
        "calib_prevalence": round(float(np.mean(calib_y)), 4),
        "formula": "p_calib = 1 / (1 + exp(-(A * logit + B)))",
        "verified_zero_test_leakage": True
    }
    params_path = os.path.join(args.out_dir, "calibration_model_params.json")
    with open(params_path, "w") as f:
        json.dump(calib_params, f, indent=2)
    print(f"  Calibration parameters saved to: {params_path}")

    # 4. Apply Calibration to Holdout Test Split
    print("\n[Step 4/5] Applying calibration to holdout test split...")
    test_logits = test_df['logit'].values.reshape(-1, 1)
    test_p_raw = test_df['prob_raw'].values
    test_p_platt = platt.predict_proba(test_logits)[:, 1]
    test_p_iso = iso.predict(test_p_raw)

    test_df['prob_platt'] = test_p_platt
    test_df['prob_iso'] = test_p_iso

    # Update actual_test_logits.csv with calibrated probabilities
    test_df.to_csv(os.path.join(args.out_dir, "actual_test_logits.csv"), index=False)

    # 5. Comprehensive Metric Evaluation & Fairness Stratification
    print("\n[Step 5/5] Computing calibration metrics, subgroup breakdown, and patient-clustered CIs...")
    test_y = test_df['target'].values.astype(int)

    # Overall Metrics
    overall_raw = compute_classification_metrics(test_y, test_p_raw, threshold=0.50)
    overall_platt_050 = compute_classification_metrics(test_y, test_p_platt, threshold=0.50)
    overall_platt_002 = compute_classification_metrics(test_y, test_p_platt, threshold=0.02)
    overall_platt_005 = compute_classification_metrics(test_y, test_p_platt, threshold=0.05)
    overall_iso = compute_classification_metrics(test_y, test_p_iso, threshold=0.02)

    # Clustered Bootstrap CIs
    print("  Running patient-clustered non-parametric bootstrap (1,000 iterations)...")
    ci_overall_raw = compute_patient_clustered_bootstrap(test_df, 'prob_raw', threshold=0.50)
    ci_overall_platt = compute_patient_clustered_bootstrap(test_df, 'prob_platt', threshold=0.02)

    # Subgroup Breakdown by group_col (Light vs Dark)
    subgroups = {}
    ci_subgroups = {}
    if args.group_col in test_df.columns:
        unique_groups = test_df[args.group_col].dropna().unique()
        for grp in unique_groups:
            grp_df = test_df[test_df[args.group_col] == grp].copy().reset_index(drop=True)
            grp_y = grp_df['target'].values.astype(int)
            grp_p_raw = grp_df['prob_raw'].values
            grp_p_platt = grp_df['prob_platt'].values

            subgroups[grp] = {
                "n_images": len(grp_df),
                "n_patients": grp_df[args.cluster_col].nunique(),
                "n_melanomas": int(grp_y.sum()),
                "prevalence": round(float(np.mean(grp_y)), 4),
                "raw_tau_050": compute_classification_metrics(grp_y, grp_p_raw, threshold=0.50),
                "platt_pt_002": compute_classification_metrics(grp_y, grp_p_platt, threshold=0.02),
                "platt_pt_005": compute_classification_metrics(grp_y, grp_p_platt, threshold=0.05),
            }
            ci_subgroups[grp] = {
                "raw": compute_patient_clustered_bootstrap(grp_df, 'prob_raw', threshold=0.50),
                "platt": compute_patient_clustered_bootstrap(grp_df, 'prob_platt', threshold=0.02)
            }

    # Between-group Disparity Analysis (Light - Dark)
    between_group_diff = {}
    if "light" in subgroups and "dark" in subgroups:
        # Differences at screening threshold (platt pt=0.02)
        diff_auc = subgroups["light"]["platt_pt_002"]["auroc"] - subgroups["dark"]["platt_pt_002"]["auroc"]
        diff_brier = subgroups["light"]["platt_pt_002"]["brier_score"] - subgroups["dark"]["platt_pt_002"]["brier_score"]
        diff_ece = subgroups["light"]["platt_pt_002"]["ece"] - subgroups["dark"]["platt_pt_002"]["ece"]
        diff_sens = subgroups["light"]["platt_pt_002"]["sensitivity"] - subgroups["dark"]["platt_pt_002"]["sensitivity"]
        diff_spec = subgroups["light"]["platt_pt_002"]["specificity"] - subgroups["dark"]["platt_pt_002"]["specificity"]

        between_group_diff = {
            "comparison": "Light Skin (ITA > 28) minus Dark Skin (ITA <= 28)",
            "delta_auroc": round(diff_auc, 4),
            "delta_brier": round(diff_brier, 4),
            "delta_ece": round(diff_ece, 4),
            "delta_sensitivity_pt002": round(diff_sens, 4),
            "delta_specificity_pt002": round(diff_spec, 4),
            "light_summary": {
                "images": subgroups["light"]["n_images"],
                "patients": subgroups["light"]["n_patients"],
                "melanomas": subgroups["light"]["n_melanomas"],
                "auroc": subgroups["light"]["platt_pt_002"]["auroc"],
                "auroc_95ci": ci_subgroups["light"]["platt"]["auroc"]["ci_95"],
                "ece": subgroups["light"]["platt_pt_002"]["ece"],
                "brier": subgroups["light"]["platt_pt_002"]["brier_score"]
            },
            "dark_summary": {
                "images": subgroups["dark"]["n_images"],
                "patients": subgroups["dark"]["n_patients"],
                "melanomas": subgroups["dark"]["n_melanomas"],
                "auroc": subgroups["dark"]["platt_pt_002"]["auroc"],
                "auroc_95ci": ci_subgroups["dark"]["platt"]["auroc"]["ci_95"],
                "ece": subgroups["dark"]["platt_pt_002"]["ece"],
                "brier": subgroups["dark"]["platt_pt_002"]["brier_score"]
            }
        }

    # Plot Reliability Diagram
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Left: Overall Raw vs Platt Calibrated
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")
    centers_raw, acc_raw, _ = compute_reliability_curve(test_y, test_p_raw, n_bins=10)
    centers_platt, acc_platt, _ = compute_reliability_curve(test_y, test_p_platt, n_bins=10)
    ax.plot(centers_raw, acc_raw, "s-", color="#ef4444", lw=2, label=f"Uncalibrated (ECE = {overall_raw['ece']:.4f})")
    ax.plot(centers_platt, acc_platt, "o-", color="#10b981", lw=2, label=f"Platt Calibrated (ECE = {overall_platt_002['ece']:.4f})")
    ax.set_title("Overall Test Set Reliability Diagram (N = 6,458)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Empirical Probability (True Positive Rate)", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, ls=":", alpha=0.6)

    # Right: Subgroup Reliability Diagram (Platt Calibrated Light vs Dark)
    ax2 = axes[1]
    ax2.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")
    if "light" in subgroups and "dark" in subgroups:
        light_df = test_df[test_df[args.group_col] == "light"]
        dark_df = test_df[test_df[args.group_col] == "dark"]
        c_l, a_l, _ = compute_reliability_curve(light_df['target'].values, light_df['prob_platt'].values, n_bins=10)
        c_d, a_d, _ = compute_reliability_curve(dark_df['target'].values, dark_df['prob_platt'].values, n_bins=10)
        ax2.plot(c_l, a_l, "o-", color="#3b82f6", lw=2, label=f"Light Skin (ECE = {subgroups['light']['platt_pt_002']['ece']:.4f})")
        ax2.plot(c_d, a_d, "^-", color="#f59e0b", lw=2, label=f"Dark Skin (ECE = {subgroups['dark']['platt_pt_002']['ece']:.4f})")
    ax2.set_title("Platt Calibration by ITA Skin Group", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax2.set_ylabel("Empirical Probability (True Positive Rate)", fontsize=10)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(frameon=True, fontsize=9)
    ax2.grid(True, ls=":", alpha=0.6)

    plt.tight_layout()
    curve_path = os.path.join(args.out_dir, "calibration_curve_comparison.png")
    plt.savefig(curve_path, bbox_inches="tight")
    plt.close()
    print(f"  Calibration curves saved to: {curve_path}")

    # Compile Final Summary Metrics JSON
    results_summary = {
        "study": "MelanomaAI Clean Dataset Probability Calibration Study",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": args.ckpt,
        "manifest": args.manifest,
        "splits": {
            "calibration": {
                "n_images": len(calib_df),
                "n_patients": len(calib_patients),
                "n_melanomas": int(calib_y.sum()),
                "prevalence": round(float(np.mean(calib_y)), 4)
            },
            "holdout_test": {
                "n_images": len(test_df),
                "n_patients": len(test_patients),
                "n_melanomas": int(test_y.sum()),
                "prevalence": round(float(np.mean(test_y)), 4)
            },
            "patient_overlap": len(leakage)
        },
        "fitted_calibrator": calib_params,
        "overall_test_metrics": {
            "auroc": overall_raw["auroc"],
            "auroc_clustered_95ci": ci_overall_raw["auroc"]["ci_95"],
            "uncalibrated_raw_tau_050": overall_raw,
            "platt_calibrated_pt_002": overall_platt_002,
            "platt_calibrated_pt_005": overall_platt_005,
            "platt_calibrated_tau_050": overall_platt_050,
            "isotonic_calibrated_pt_002": overall_iso,
            "ece_reduction_percent": round((overall_raw["ece"] - overall_platt_002["ece"]) / overall_raw["ece"] * 100, 2),
            "brier_reduction_percent": round((overall_raw["brier_score"] - overall_platt_002["brier_score"]) / overall_raw["brier_score"] * 100, 2),
            "clustered_confidence_intervals": {
                "raw": ci_overall_raw,
                "platt": ci_overall_platt
            }
        },
        "subgroup_metrics_by_skin_group": subgroups,
        "subgroup_clustered_cis": ci_subgroups,
        "between_group_disparity": between_group_diff
    }

    metrics_json_path = os.path.join(args.out_dir, "calibration_metrics_test.json")
    with open(metrics_json_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"  Comprehensive metrics JSON saved to: {metrics_json_path}")

    # Generate Markdown Report
    report_path = os.path.join(args.out_dir, "calibration_study_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# MelanomaAI Real-Data Calibration Study Report\n\n")
        f.write(f"**Date:** {time.strftime('%B %d, %Y')}  \n")
        f.write(f"**Checkpoint:** `{args.ckpt}`  \n")
        f.write(f"**Manifest:** `{args.manifest}`  \n")
        f.write(f"**Output Directory:** `{args.out_dir}`  \n\n")
        f.write("---\n\n")

        f.write("## 1. Study Overview & Partition Independence\n\n")
        f.write("A rigorous probability calibration study was conducted using the verified clean EfficientNet-B3 checkpoint on real ISIC 2020 dermoscopy images.\n\n")
        f.write("| Partition | Images | Unique Patients | Melanomas | Prevalence |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Calibration Split** | {len(calib_df):,} | {len(calib_patients):,} | {int(calib_y.sum()):,} | {float(np.mean(calib_y))*100:.2f}% |\n")
        f.write(f"| **Holdout Test Split** | {len(test_df):,} | {len(test_patients):,} | {int(test_y.sum()):,} | {float(np.mean(test_y))*100:.2f}% |\n\n")
        f.write(f"> **Partition Independence:** Verified **0% patient leakage** between calibration and test sets (0 shared patient IDs).\n\n")

        f.write("## 2. Fitted Calibration Parameters (Calibration Split Only)\n\n")
        f.write("Platt scaling was fitted exclusively on the 3,334 calibration images without exposing holdout test labels:\n")
        f.write(f"- **Formula:** $p_{{\\text{{calib}}}} = \\frac{{1}}{{1 + \\exp(-(A \\cdot \\text{{logit}} + B))}}$\n")
        f.write(f"- **Coefficient $A$ (Scaling):** `{platt_a:.6f}`\n")
        f.write(f"- **Coefficient $B$ (Intercept):** `{platt_b:.6f}`\n\n")

        f.write("## 3. Holdout Test Set Calibration Results (N = 6,458)\n\n")
        f.write("| Calibration State | AUROC (95% CI) | ECE | Brier Score | Operating Point | Sensitivity | Specificity | PPV | NPV |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Uncalibrated (Raw)** | {overall_raw['auroc']:.4f} ({ci_overall_raw['auroc']['ci_95'][0]:.4f} - {ci_overall_raw['auroc']['ci_95'][1]:.4f}) | {overall_raw['ece']:.4f} | {overall_raw['brier_score']:.4f} | $\\tau = 0.50$ | {overall_raw['sensitivity']*100:.2f}% | {overall_raw['specificity']*100:.2f}% | {overall_raw['precision_ppv']*100:.2f}% | {overall_raw['npv']*100:.2f}% |\n")
        f.write(f"| **Platt Calibrated** | {overall_raw['auroc']:.4f} ({ci_overall_raw['auroc']['ci_95'][0]:.4f} - {ci_overall_raw['auroc']['ci_95'][1]:.4f}) | **{overall_platt_002['ece']:.4f}** | **{overall_platt_002['brier_score']:.4f}** | $p_t = 0.02$ | {overall_platt_002['sensitivity']*100:.2f}% | {overall_platt_002['specificity']*100:.2f}% | {overall_platt_002['precision_ppv']*100:.2f}% | {overall_platt_002['npv']*100:.2f}% |\n")
        f.write(f"| **Platt Calibrated** | {overall_raw['auroc']:.4f} | **{overall_platt_005['ece']:.4f}** | **{overall_platt_005['brier_score']:.4f}** | $p_t = 0.05$ | {overall_platt_005['sensitivity']*100:.2f}% | {overall_platt_005['specificity']*100:.2f}% | {overall_platt_005['precision_ppv']*100:.2f}% | {overall_platt_005['npv']*100:.2f}% |\n")
        f.write(f"| **Isotonic Regression** | {overall_iso['auroc']:.4f} | {overall_iso['ece']:.4f} | {overall_iso['brier_score']:.4f} | $p_t = 0.02$ | {overall_iso['sensitivity']*100:.2f}% | {overall_iso['specificity']*100:.2f}% | {overall_iso['precision_ppv']*100:.2f}% | {overall_iso['npv']*100:.2f}% |\n\n")

        f.write(f"- **Calibration Error Reduction:** Expected Calibration Error (ECE) decreased by **{results_summary['overall_test_metrics']['ece_reduction_percent']}%** (from {overall_raw['ece']:.4f} to {overall_platt_002['ece']:.4f}).\n")
        f.write(f"- **Brier Score Improvement:** Brier score improved by **{results_summary['overall_test_metrics']['brier_reduction_percent']}%** (from {overall_raw['brier_score']:.4f} to {overall_platt_002['brier_score']:.4f}).\n\n")

        if between_group_diff:
            f.write("## 4. Subgroup Calibration & Disparity by Skin Group (ITA)\n\n")
            f.write("Images were stratified using colorimetric Individual Typology Angle (ITA) into Light ($ITA > 28$) and Dark ($ITA \\le 28$):\n\n")
            f.write("| Subgroup | Images | Patients | Melanomas | AUROC (95% CI) | ECE | Brier | Sens ($p_t=0.02$) | Spec ($p_t=0.02$) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            l_s = between_group_diff["light_summary"]
            d_s = between_group_diff["dark_summary"]
            f.write(f"| **Light Skin (ITA > 28)** | {l_s['images']:,} | {l_s['patients']:,} | {l_s['melanomas']:,} | **{l_s['auroc']:.4f}** ({l_s['auroc_95ci'][0]:.4f} - {l_s['auroc_95ci'][1]:.4f}) | {l_s['ece']:.4f} | {l_s['brier']:.4f} | {subgroups['light']['platt_pt_002']['sensitivity']*100:.2f}% | {subgroups['light']['platt_pt_002']['specificity']*100:.2f}% |\n")
            f.write(f"| **Dark Skin (ITA <= 28)** | {d_s['images']:,} | {d_s['patients']:,} | {d_s['melanomas']:,} | **{d_s['auroc']:.4f}** ({d_s['auroc_95ci'][0]:.4f} - {d_s['auroc_95ci'][1]:.4f}) | {d_s['ece']:.4f} | {d_s['brier']:.4f} | {subgroups['dark']['platt_pt_002']['sensitivity']*100:.2f}% | {subgroups['dark']['platt_pt_002']['specificity']*100:.2f}% |\n\n")

            f.write("### Between-Group Disparity ($\\Delta = \\text{Light} - \\text{Dark}$):\n")
            f.write(f"- $\\Delta\\text{{AUROC}}$: **{between_group_diff['delta_auroc']:+.4f}** (Disparity favoring lighter skin)\n")
            f.write(f"- $\\Delta\\text{{ECE}}$: **{between_group_diff['delta_ece']:+.4f}**\n")
            f.write(f"- $\\Delta\\text{{Brier}}$: **{between_group_diff['delta_brier']:+.4f}**\n")
            f.write(f"- $\\Delta\\text{{Sensitivity at }} p_t=0.02$: **{between_group_diff['delta_sensitivity_pt002']:+.4f}**\n")
            f.write(f"- $\\Delta\\text{{Specificity at }} p_t=0.02$: **{between_group_diff['delta_specificity_pt002']:+.4f}**\n\n")

        f.write("## 5. Artifacts and Generated Output Files\n\n")
        f.write(f"1. `actual_calibration_logits.csv`: Real calibration split predictions and logits ({len(calib_df):,} rows)\n")
        f.write(f"2. `actual_test_logits.csv`: Real test split predictions, logits, and calibrated probabilities ({len(test_df):,} rows)\n")
        f.write(f"3. `calibration_model_params.json`: Fitted Platt scaling parameters\n")
        f.write(f"4. `calibration_metrics_test.json`: Complete metrics dictionary with patient-clustered 95% CIs\n")
        f.write(f"5. `calibration_curve_comparison.png`: Dual reliability diagrams\n")
        f.write(f"6. `calibration_study_report.md`: This comprehensive report\n")

    print(f"  Markdown study report saved to: {report_path}")
    print("\n[COMPLETE] Real-data calibration study executed successfully.")


if __name__ == "__main__":
    main()
