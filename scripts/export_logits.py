"""
export_logits.py -- RUN THIS ON THE MACHINE THAT HAS THE MODEL AND IMAGES.

Exports raw logits (scores BEFORE the sigmoid) for the calibration and test
partitions, using the same preprocessing as run_dark_skin_fairness_evaluation.py
(CLAHE is applied when ITA <= 28). Send the output CSV back; no images are included.

Example (from the project root):
    python scripts/export_logits.py ^
        --ckpt outputs/clean_retraining/checkpoints/clean_model_best.pth ^
        --manifest outputs/clean_retraining/clean_split_manifest.csv ^
        --check-against outputs/clean_retraining/test_predictions.csv ^
        --out calib_test_logits.csv

The manifest must have columns: partition, patient_id, target, filepath
(image_name is kept if present). --check-against compares sigmoid(logit) with the
saved prob_raw on the test rows; the max difference should be tiny (< 0.01).
If it is large, the preprocessing does not match -- do not send the file.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.special import expit
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.model import build_model, val_tf, DEVICE          # val_tf = Resize(384)+Normalize
from risk_engine import estimate_ita, apply_clahe


class ExportDataset(Dataset):
    def __init__(self, df):
        self.paths = df["filepath"].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = np.array(Image.open(self.paths[i]).convert("RGB"))
        ita = float(estimate_ita(img))
        if ita <= 28:
            img = apply_clahe(img)
        return val_tf(image=img)["image"], ita


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="calib_test_logits.csv")
    ap.add_argument("--partitions", nargs="+", default=["calib", "test"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--check-against", default=None, help="saved test_predictions.csv with prob_raw")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    df = df[df["partition"].isin(args.partitions)].reset_index(drop=True)
    print(f"Rows to export: {len(df)}  ({df['partition'].value_counts().to_dict()})")

    model = build_model()
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.classifier[-1] = torch.nn.Identity()      # remove Sigmoid -> raw logits
    model.to(DEVICE).eval()

    loader = DataLoader(ExportDataset(df), batch_size=args.batch_size, shuffle=False, num_workers=0)
    logits, itas = [], []
    with torch.no_grad():                            # full float32 on purpose (no autocast)
        for k, (x, ita) in enumerate(loader):
            logits.extend(model(x.to(DEVICE)).squeeze(-1).cpu().numpy().astype(float).tolist())
            itas.extend(ita.numpy().tolist())
            if (k + 1) % 20 == 0:
                print(f"  {len(logits)}/{len(df)} done")

    keep = [c for c in ["image_name", "patient_id", "target", "partition"] if c in df.columns]
    out = df[keep].copy()
    out["ita"] = np.round(itas, 3)
    out["skin_group"] = np.where(out["ita"] > 28, "light", "dark")
    out["logit"] = np.round(logits, 6)
    out.to_csv(args.out, index=False)
    print(f"Saved {args.out}")

    if args.check_against:
        saved = pd.read_csv(args.check_against)
        test_rows = out[out["partition"] == "test"].reset_index(drop=True)
        if len(saved) != len(test_rows):
            print(f"CHECK SKIPPED: row counts differ ({len(saved)} vs {len(test_rows)})")
        else:
            diff = np.abs(expit(test_rows["logit"].to_numpy()) - saved["prob_raw"].to_numpy())
            print(f"CHECK: max |sigmoid(logit) - prob_raw| = {diff.max():.5f}, mean = {diff.mean():.5f}")
            print("  OK" if diff.max() < 0.01 else "  MISMATCH -> preprocessing differs, do NOT send this file")


if __name__ == "__main__":
    main()
