"""
Melanoma CNN - Robustness & Fairness Evaluation
Tests model performance across skin tone groups (ITA-based)
Generates fairness report comparing light vs dark skin predictions
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────
IMG_SIZE = 224
DATA_DIR = "data/processed"
MODEL_PATH = "models/melanoma_final.pth"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    base = models.efficientnet_b3(weights=None)
    nf = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(.5), nn.Linear(nf,512), nn.ReLU(),
        nn.BatchNorm1d(512), nn.Dropout(.3), nn.Linear(512,256),
        nn.ReLU(), nn.Dropout(.2), nn.Linear(256,1), nn.Sigmoid()
    )
    return base

# ─── ITA SKIN TONE DETECTION ─────────────────────────────────────────────────
def estimate_ita(image_rgb):
    """Estimate Individual Typology Angle from RGB image (numpy array 0-255)"""
    try:
        lab = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)
        h, w = lab.shape[:2]
        margin = int(min(h, w) * 0.1)

        # Sample corners for background skin tone
        corners = [
            lab[:margin, :margin],
            lab[:margin, -margin:],
            lab[-margin:, :margin],
            lab[-margin:, -margin:]
        ]
        corner_pixels = np.vstack([c.reshape(-1, 3) for c in corners])
        L = np.mean(corner_pixels[:, 0]) * 100 / 255
        b = np.mean(corner_pixels[:, 2]) - 128

        if abs(b) < 1e-6:
            b = 1e-6
        ita = np.arctan((L - 50) / b) * (180 / np.pi)
        return ita
    except:
        return 0.0

def classify_skin_tone(ita):
    """Classify ITA value into skin tone group"""
    if ita > 55:
        return "Very Light"
    elif ita > 41:
        return "Light"
    elif ita > 28:
        return "Intermediate"
    elif ita > 10:
        return "Tan"
    elif ita > -30:
        return "Brown"
    else:
        return "Dark"

def is_dark_skin(ita):
    return ita <= 10  # Brown or Dark categories

# ─── CLAHE PREPROCESSING ─────────────────────────────────────────────────────
def apply_clahe(image_rgb):
    """Apply CLAHE enhancement for dark skin tones"""
    lab = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return enhanced

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
def load_model():
    if os.path.exists(MODEL_PATH):
        print(f"✅ Loaded model: {MODEL_PATH}")
        model = build_model()
        ck = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        model = model.to(device).eval()
        return model
    raise FileNotFoundError("No model found. Train model first.")

# ─── EVALUATE ─────────────────────────────────────────────────────────────────
def evaluate_by_skin_tone():
    model = load_model()
    results = {
        "light_skin": {"labels": [], "preds_standard": [], "preds_robustness": []},
        "dark_skin":  {"labels": [], "preds_standard": [], "preds_robustness": []}
    }

    total = 0
    for label_idx, label_name in enumerate(["benign", "malignant"]):
        folder = os.path.join(DATA_DIR, label_name)
        if not os.path.exists(folder):
            print(f"⚠️  Folder not found: {folder}")
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # Use up to 500 samples per class for speed
        files = files[:500]
        print(f"📂 Processing {label_name}: {len(files)} images...")

        for fname in files:
            img_path = os.path.join(folder, fname)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

                # Detect skin tone
                ita = estimate_ita(img_resized)
                group = "dark_skin" if is_dark_skin(ita) else "light_skin"

                # Standard prediction
                tfm = A.Compose([
                    A.Resize(IMG_SIZE, IMG_SIZE),
                    A.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225]),
                    ToTensorV2()
                ])
                inp_t = tfm(image=img_rgb)["image"].unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred_std = float(model(inp_t).item())

                # Robustness prediction (CLAHE for dark skin)
                if is_dark_skin(ita):
                    img_enhanced = apply_clahe(img_resized)
                    inp_enh_t = tfm(image=img_enhanced)["image"].unsqueeze(0).to(device)
                    with torch.no_grad():
                        pred_rob = float(model(inp_enh_t).item())
                else:
                    pred_rob = pred_std

                results[group]["labels"].append(label_idx)
                results[group]["preds_standard"].append(pred_std)
                results[group]["preds_robustness"].append(pred_rob)
                total += 1

            except Exception as e:
                continue

    print(f"\n✅ Processed {total} images total")
    return results

# ─── COMPUTE METRICS ─────────────────────────────────────────────────────────
def compute_group_metrics(labels, preds, threshold=0.5):
    if len(labels) < 2:
        return {"accuracy": 0, "auc": 0, "f1": 0, "sensitivity": 0, "specificity": 0, "n": len(labels)}

    labels = np.array(labels)
    preds = np.array(preds)
    binary = (preds >= threshold).astype(int)

    acc = (binary == labels).mean()
    f1 = f1_score(labels, binary, zero_division=0)

    try:
        auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.0
    except:
        auc = 0.0

    cm = confusion_matrix(labels, binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)

    return {
        "accuracy": float(acc), "auc": float(auc), "f1": float(f1),
        "sensitivity": float(sensitivity), "specificity": float(specificity),
        "n": int(len(labels))
    }

# ─── PLOT FAIRNESS REPORT ─────────────────────────────────────────────────────
def plot_fairness_report(light_std, light_rob, dark_std, dark_rob):
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor('#0D1117')

    metrics = ['accuracy', 'auc', 'sensitivity', 'specificity', 'f1']
    groups = ['Light Skin\n(Standard)', 'Light Skin\n(Robustness)', 'Dark Skin\n(Standard)', 'Dark Skin\n(Robustness)']
    colors = ['#58A6FF', '#79C0FF', '#F85149', '#3FB950']

    for i, metric in enumerate(['accuracy', 'auc', 'f1']):
        ax = axes[i]
        ax.set_facecolor('#161B22')
        vals = [light_std[metric], light_rob[metric], dark_std[metric], dark_rob[metric]]
        bars = ax.bar(groups, vals, color=colors, alpha=0.85, width=0.6)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')

        ax.set_title(metric.upper().replace('_', ' '), color='white', fontsize=13, pad=10)
        ax.set_ylim(0, 1.15)
        ax.tick_params(colors='#8B949E', labelsize=9)
        ax.spines[:].set_color('#30363D')
        ax.set_facecolor('#161B22')
        ax.yaxis.label.set_color('#8B949E')

    fig.suptitle('Fairness Report: Light vs Dark Skin Tone Performance',
                 color='white', fontsize=15, fontweight='bold')

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/fairness_report.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    print(f"📊 Fairness report saved: {out_path}")
    plt.close()

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  MELANOMA CNN — ROBUSTNESS & FAIRNESS EVALUATION")
    print("=" * 60)

    results = evaluate_by_skin_tone()

    light = results["light_skin"]
    dark = results["dark_skin"]

    print(f"\n📊 Light skin samples: {len(light['labels'])}")
    print(f"📊 Dark skin samples:  {len(dark['labels'])}")

    light_std = compute_group_metrics(light["labels"], light["preds_standard"])
    light_rob = compute_group_metrics(light["labels"], light["preds_robustness"])
    dark_std  = compute_group_metrics(dark["labels"],  dark["preds_standard"])
    dark_rob  = compute_group_metrics(dark["labels"],  dark["preds_robustness"])

    print("\n" + "="*60)
    print("FAIRNESS RESULTS")
    print("="*60)
    print(f"{'Group':<30} {'Acc':>8} {'AUC':>8} {'F1':>8} {'Sens':>8} {'Spec':>8}")
    print("-" * 65)
    for name, m in [("Light Skin (Standard)", light_std), ("Light Skin (Robustness)", light_rob),
                    ("Dark Skin (Standard)", dark_std), ("Dark Skin (Robustness)", dark_rob)]:
        print(f"{name:<30} {m['accuracy']:>8.4f} {m['auc']:>8.4f} {m['f1']:>8.4f} {m['sensitivity']:>8.4f} {m['specificity']:>8.4f}")

    # Fairness gap
    gap_std = abs(light_std["accuracy"] - dark_std["accuracy"])
    gap_rob = abs(light_rob["accuracy"] - dark_rob["accuracy"])
    print(f"\n⚖️  Fairness gap (standard):   {gap_std:.4f}")
    print(f"⚖️  Fairness gap (robustness): {gap_rob:.4f}")
    improvement = ((gap_std - gap_rob) / (gap_std + 1e-9)) * 100
    print(f"✅ Gap reduction by robustness layer: {improvement:.1f}%")

    # Plot
    plot_fairness_report(light_std, light_rob, dark_std, dark_rob)

    # Save
    report = {
        "timestamp": datetime.now().isoformat(),
        "light_skin_standard": light_std,
        "light_skin_robustness": light_rob,
        "dark_skin_standard": dark_std,
        "dark_skin_robustness": dark_rob,
        "fairness_gap_standard": float(gap_std),
        "fairness_gap_robustness": float(gap_rob),
        "gap_improvement_pct": float(improvement)
    }
    with open(f"{OUTPUT_DIR}/robustness_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 Report saved: {OUTPUT_DIR}/robustness_report.json")
    print("✅ Robustness evaluation complete!")