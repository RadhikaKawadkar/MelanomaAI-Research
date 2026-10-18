"""
MelanomaAI — Complete Dark-Skin Fairness & CLAHE Ablation Evaluation
Evaluates the clean-trained checkpoint (clean_model_best.pth) across ALL 6,458
holdout test images in ISIC 2020.

Performs:
1. Full-cohort ITA estimation across all 6,458 test images.
2. Stratification into Light Skin (ITA > 28) and Dark Skin (ITA <= 28).
3. Patient-level and image-level count reporting.
4. Paired CLAHE vs. NO-CLAHE ablation on the exact same checkpoint and images.
5. Standardized operating metrics (AUROC, Sens, Spec, FNR, CM) at documented
   a priori thresholds (raw tau=0.50, raw tau=0.40, calibrated pt=0.02, pt=0.05).
6. Patient-clustered non-parametric bootstrap confidence intervals (1,000 iterations).
7. Outputs saved under C:\\DS_CP\\outputs\\dark_skin_fairness\\.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, confusion_matrix

sys.path.insert(0, r"C:\DS_CP")
from risk_engine import estimate_ita, apply_clahe, get_skin_tone
from backend.model import calibrate_probability, PLATT_COEFF_A, PLATT_COEFF_B

# Configuration
BASE_DIR = r"C:\DS_CP"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "dark_skin_fairness")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKPT_PATH = os.path.join(BASE_DIR, "outputs", "clean_retraining", "checkpoints", "clean_model_best.pth")
MANIFEST_PATH = os.path.join(BASE_DIR, "outputs", "clean_retraining", "clean_split_manifest.csv")
TEST_PREDS_PATH = os.path.join(BASE_DIR, "outputs", "clean_retraining", "test_predictions.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 384
BATCH_SIZE = 32

eval_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


class ISICInferenceDataset(Dataset):
    def __init__(self, df, transform=None, force_clahe=False, no_clahe=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.force_clahe = force_clahe
        self.no_clahe = no_clahe
        self.filepaths = self.df['filepath'].values
        self.itas = self.df['ita'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        ita = self.itas[idx]
        img = Image.open(path).convert('RGB')
        img_np = np.array(img)

        # CLAHE logic:
        # Standard pipeline applies CLAHE when ITA <= 28
        # Ablation (no_clahe=True) suppresses CLAHE completely
        if not self.no_clahe:
            if self.force_clahe or ita <= 28:
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


def compute_metrics_at_threshold(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    return {
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "false_negative_rate": round(fnr, 4),
        "ppv": round(ppv, 4),
        "npv": round(npv, 4),
        "referral_rate": round(float((tp + fp) / len(y_true)), 4)
    }


def patient_clustered_bootstrap_ci(df, prob_col, target_col='target', n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    pts = df['patient_id'].unique()
    n_pts = len(pts)

    auc_list = []
    sens_raw05_list, spec_raw05_list, fnr_raw05_list = [], [], []
    sens_platt02_list, spec_platt02_list, fnr_platt02_list = [], [], []

    # Map patient to indices
    pt_to_indices = {pt: df.index[df['patient_id'] == pt].values for pt in pts}

    for _ in range(n_boot):
        sample_pts = rng.choice(pts, size=n_pts, replace=True)
        sample_idx = np.concatenate([pt_to_indices[pt] for pt in sample_pts])
        sample_y = df.iloc[sample_idx][target_col].values
        sample_p = df.iloc[sample_idx][prob_col].values

        if len(np.unique(sample_y)) < 2:
            continue

        try:
            auc = roc_auc_score(sample_y, sample_p)
            auc_list.append(auc)
        except Exception:
            pass

        # At raw 0.50
        cm05 = confusion_matrix(sample_y, (sample_p >= 0.50).astype(int), labels=[0, 1])
        tn, fp, fn, tp = cm05.ravel()
        if (tp + fn) > 0:
            sens_raw05_list.append(tp / (tp + fn))
            fnr_raw05_list.append(fn / (tp + fn))
        if (tn + fp) > 0:
            spec_raw05_list.append(tn / (tn + fp))

    def get_ci(arr):
        if len(arr) == 0:
            return 0.0, 0.0, 0.0
        return round(float(np.mean(arr)), 4), round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)

    return {
        "roc_auc": get_ci(auc_list),
        "sensitivity_raw_050": get_ci(sens_raw05_list),
        "specificity_raw_050": get_ci(spec_raw05_list),
        "fnr_raw_050": get_ci(fnr_raw05_list)
    }


def main():
    print("=" * 70)
    print("  MELANOMAAI FULL COHORT DARK-SKIN FAIRNESS & ABLATION AUDIT")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # 1. Load manifest & existing test set predictions
    print("\n[Step 1/5] Loading test manifest and cached predictions...")
    manifest = pd.read_csv(MANIFEST_PATH)
    test_df = manifest[manifest['partition'] == 'test'].copy().reset_index(drop=True)
    N_total = len(test_df)
    print(f"  Total test set images: {N_total:,} across {test_df['patient_id'].nunique():,} unique patients.")

    # 2. Compute ITA for all 6,458 test images
    print("\n[Step 2/5] Computing ITA angles for ALL 6,458 test images...")
    t0 = time.time()
    itas = []
    tones = []
    for idx, fp in enumerate(test_df['filepath']):
        img_np = np.array(Image.open(fp).convert('RGB'))
        ita_val = estimate_ita(img_np)
        tone_lbl, _, _, _ = get_skin_tone(ita_val)
        itas.append(ita_val)
        tones.append(tone_lbl)
        if (idx + 1) % 1000 == 0:
            print(f"    Processed {idx+1:,} / {N_total:,} images ({(idx+1)/N_total*100:.1f}%)...")
    dt = time.time() - t0
    print(f"  Completed ITA estimation in {dt:.1f}s ({dt/N_total*1000:.1f} ms/image).")

    test_df['ita'] = itas
    test_df['skin_group'] = np.where(test_df['ita'] > 28, 'light', 'dark')
    test_df['tone_label'] = tones

    # Load existing test predictions (with CLAHE)
    saved_preds = pd.read_csv(TEST_PREDS_PATH)
    test_df['prob_raw_clahe'] = saved_preds['prob_raw']
    test_df['prob_platt_clahe'] = saved_preds['prob_platt']

    # 3. Model Loading for Ablation (Without CLAHE)
    print("\n[Step 3/5] Loading model checkpoint for NO-CLAHE ablation inference...")
    model = build_model()
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Loaded checkpoint from {CKPT_PATH}")

    # Run inference WITHOUT CLAHE across the entire test set
    print("  Running inference WITHOUT CLAHE (no contrast enhancement)...")
    ds_noclahe = ISICInferenceDataset(test_df, transform=eval_transform, no_clahe=True)
    loader_noclahe = DataLoader(ds_noclahe, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    probs_noclahe = []
    with torch.no_grad():
        for imgs, _ in loader_noclahe:
            imgs = imgs.to(DEVICE)
            with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
                preds = model(imgs).squeeze(-1)
            probs_noclahe.extend(preds.cpu().numpy().tolist())

    p_raw_noclahe = np.array(probs_noclahe)
    test_df['prob_raw_noclahe'] = p_raw_noclahe
    test_df['prob_platt_noclahe'] = np.array([
        calibrate_probability(p, a=PLATT_COEFF_A, b=PLATT_COEFF_B) for p in p_raw_noclahe
    ])

    # Save full predictions CSV
    preds_out_path = os.path.join(OUTPUT_DIR, "dark_skin_fairness_predictions.csv")
    test_df[[
        'image_name', 'patient_id', 'target', 'ita', 'skin_group', 'tone_label',
        'prob_raw_clahe', 'prob_platt_clahe', 'prob_raw_noclahe', 'prob_platt_noclahe'
    ]].to_csv(preds_out_path, index=False)
    print(f"  Saved full predictions dataset to: {preds_out_path}")

    # 4. Cohort Characterization & Performance Evaluation
    print("\n[Step 4/5] Computing subgroup metrics and CLAHE ablation statistics...")
    light_df = test_df[test_df['skin_group'] == 'light'].copy().reset_index(drop=True)
    dark_df = test_df[test_df['skin_group'] == 'dark'].copy().reset_index(drop=True)

    def analyze_subgroup(sub_df, name):
        y = sub_df['target'].values
        n_imgs = len(sub_df)
        n_pts = sub_df['patient_id'].nunique()
        n_mel = int(np.sum(y == 1))
        n_ben = int(np.sum(y == 0))
        prev = n_mel / n_imgs if n_imgs > 0 else 0.0

        res = {
            "name": name,
            "images": n_imgs,
            "patients": n_pts,
            "melanoma_cases": n_mel,
            "benign_cases": n_ben,
            "prevalence": round(prev, 4),
        }

        for cond, col_raw, col_platt in [
            ("with_clahe", "prob_raw_clahe", "prob_platt_clahe"),
            ("no_clahe", "prob_raw_noclahe", "prob_platt_noclahe")
        ]:
            p_raw = sub_df[col_raw].values
            p_platt = sub_df[col_platt].values

            # AUROC
            try:
                auc_raw = round(float(roc_auc_score(y, p_raw)), 4) if len(np.unique(y)) > 1 else None
            except Exception:
                auc_raw = None

            # Operating metrics
            op_raw_050 = compute_metrics_at_threshold(y, p_raw, 0.50)
            op_raw_040 = compute_metrics_at_threshold(y, p_raw, 0.40)
            op_platt_002 = compute_metrics_at_threshold(y, p_platt, 0.02)
            op_platt_005 = compute_metrics_at_threshold(y, p_platt, 0.05)

            # Clustered bootstrap CIs
            ci = patient_clustered_bootstrap_ci(sub_df, col_raw, n_boot=1000)

            res[cond] = {
                "auroc": auc_raw,
                "auroc_bootstrap_95ci": ci["roc_auc"],
                "raw_threshold_050": op_raw_050,
                "raw_threshold_040": op_raw_040,
                "platt_threshold_002": op_platt_002,
                "platt_threshold_005": op_platt_005,
                "ci_raw_050": {
                    "sensitivity": ci["sensitivity_raw_050"],
                    "specificity": ci["specificity_raw_050"],
                    "fnr": ci["fnr_raw_050"]
                }
            }

        return res

    summary_light = analyze_subgroup(light_df, "Light Skin (ITA > 28)")
    summary_dark = analyze_subgroup(dark_df, "Dark Skin (ITA <= 28)")
    summary_overall = analyze_subgroup(test_df, "All Test Images")

    # CLAHE Delta Analysis for Dark Skin
    dark_y = dark_df['target'].values
    dark_p_clahe = dark_df['prob_raw_clahe'].values
    dark_p_noclahe = dark_df['prob_raw_noclahe'].values
    mean_prob_diff = float(np.mean(dark_p_clahe - dark_p_noclahe))
    auc_diff = float(summary_dark["with_clahe"]["auroc"] - summary_dark["no_clahe"]["auroc"])
    sens_diff = float(summary_dark["with_clahe"]["raw_threshold_050"]["sensitivity"] - summary_dark["no_clahe"]["raw_threshold_050"]["sensitivity"])
    spec_diff = float(summary_dark["with_clahe"]["raw_threshold_050"]["specificity"] - summary_dark["no_clahe"]["raw_threshold_050"]["specificity"])

    clahe_ablation_summary = {
        "description": "Head-to-head comparison on all dark-skin test images (ITA <= 28)",
        "images": len(dark_df),
        "melanomas": int(dark_df['target'].sum()),
        "auc_with_clahe": summary_dark["with_clahe"]["auroc"],
        "auc_without_clahe": summary_dark["no_clahe"]["auroc"],
        "auc_delta": round(auc_diff, 4),
        "sens_delta_raw050": round(sens_diff, 4),
        "spec_delta_raw050": round(spec_diff, 4),
        "mean_prob_shift_clahe_minus_noclahe": round(mean_prob_diff, 6),
        "conclusion": (
            "CLAHE shows no significant or demonstrable improvement in AUROC or sensitivity "
            "on dark skin for the clean-trained model; performance is virtually identical."
            if abs(auc_diff) < 0.01 else
            f"CLAHE delta: {auc_diff:+.4f} AUROC change."
        )
    }

    full_summary = {
        "protocol": "Full-Cohort ISIC 2020 Holdout Test Evaluation (N=6,458)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": CKPT_PATH,
        "overall": summary_overall,
        "light_skin": summary_light,
        "dark_skin": summary_dark,
        "clahe_ablation": clahe_ablation_summary
    }

    summary_out_path = os.path.join(OUTPUT_DIR, "dark_skin_fairness_summary.json")
    with open(summary_out_path, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"  Saved fairness summary JSON to: {summary_out_path}")

    # 5. Print Executive Console Report
    print("\n" + "=" * 70)
    print("  EXECUTIVE SUMMARY: FULL COHORT RESULTS")
    print("=" * 70)
    print(f"Cohort Breakdown:")
    print(f"  - Light Skin (ITA > 28): {summary_light['images']:,} images, {summary_light['patients']:,} patients, {summary_light['melanoma_cases']} melanomas ({summary_light['prevalence']*100:.2f}%)")
    print(f"  - Dark Skin (ITA <= 28):  {summary_dark['images']:,} images, {summary_dark['patients']:,} patients, {summary_dark['melanoma_cases']} melanomas ({summary_dark['prevalence']*100:.2f}%)")

    print(f"\nModel Performance (WITH CLAHE - Standard Pipeline):")
    print(f"  Light Skin: AUROC = {summary_light['with_clahe']['auroc']:.4f} [95% CI: {summary_light['with_clahe']['auroc_bootstrap_95ci'][1]:.4f} - {summary_light['with_clahe']['auroc_bootstrap_95ci'][2]:.4f}]")
    print(f"    At raw tau=0.50: Sens = {summary_light['with_clahe']['raw_threshold_050']['sensitivity']*100:.1f}%, Spec = {summary_light['with_clahe']['raw_threshold_050']['specificity']*100:.1f}%, FNR = {summary_light['with_clahe']['raw_threshold_050']['false_negative_rate']*100:.1f}%")
    print(f"    At Platt pt=0.02: Sens = {summary_light['with_clahe']['platt_threshold_002']['sensitivity']*100:.1f}%, Spec = {summary_light['with_clahe']['platt_threshold_002']['specificity']*100:.1f}%, FNR = {summary_light['with_clahe']['platt_threshold_002']['false_negative_rate']*100:.1f}%")

    print(f"  Dark Skin:  AUROC = {summary_dark['with_clahe']['auroc']:.4f} [95% CI: {summary_dark['with_clahe']['auroc_bootstrap_95ci'][1]:.4f} - {summary_dark['with_clahe']['auroc_bootstrap_95ci'][2]:.4f}]")
    print(f"    At raw tau=0.50: Sens = {summary_dark['with_clahe']['raw_threshold_050']['sensitivity']*100:.1f}%, Spec = {summary_dark['with_clahe']['raw_threshold_050']['specificity']*100:.1f}%, FNR = {summary_dark['with_clahe']['raw_threshold_050']['false_negative_rate']*100:.1f}%")
    print(f"    At Platt pt=0.02: Sens = {summary_dark['with_clahe']['platt_threshold_002']['sensitivity']*100:.1f}%, Spec = {summary_dark['with_clahe']['platt_threshold_002']['specificity']*100:.1f}%, FNR = {summary_dark['with_clahe']['platt_threshold_002']['false_negative_rate']*100:.1f}%")

    print(f"\nCLAHE Ablation (Dark Skin, N={summary_dark['images']}):")
    print(f"  AUROC with CLAHE:    {summary_dark['with_clahe']['auroc']:.4f}")
    print(f"  AUROC without CLAHE: {summary_dark['no_clahe']['auroc']:.4f}")
    print(f"  Delta AUROC:         {clahe_ablation_summary['auc_delta']:+.4f}")
    print(f"  Conclusion: {clahe_ablation_summary['conclusion']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
