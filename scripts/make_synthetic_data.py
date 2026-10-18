"""
make_synthetic_data.py -- creates FAKE prediction CSVs so you can test
run_calibration_study.py before the real files arrive.

!!! SYNTHETIC DATA. Never report results from these files in the paper. !!!

Output columns: image_name, patient_id, target, skin_group, logit
The fake model is 'trained on balanced batches' (so it overestimates risk) and is
less accurate on the 'dark' group, mimicking the real situation.
"""
import argparse
import os
import numpy as np
import pandas as pd


def make(n, seed, prevalence_shift=3.0):
    rng = np.random.default_rng(seed)
    n_patients = max(n // 15, 20)
    patient_id = rng.integers(0, n_patients, n)
    patient_effect = rng.normal(0, 0.5, n_patients)[patient_id]
    group = np.where(rng.random(n) < 0.3, "dark", "light")
    skill = np.where(group == "dark", 1.0, 1.6)            # weaker signal on dark skin
    signal = rng.normal(0, 1, n)
    true_logodds = -5.1 + skill * signal + patient_effect  # about 2% melanoma
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-true_logodds))).astype(int)
    # model output: overconfident AND shifted upward (balanced-batch training)
    raw_logit = 0.7 * true_logodds + prevalence_shift + rng.normal(0, 0.3, n)
    return pd.DataFrame({
        "image_name": [f"IMG_{seed}_{i:05d}" for i in range(n)],
        "patient_id": [f"S{seed}_P{p:04d}" for p in patient_id],   # different seed -> different patients
        "target": y,
        "skin_group": group,
        "logit": np.round(raw_logit, 5),
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/synthetic")
    ap.add_argument("--n-calib", type=int, default=3300)
    ap.add_argument("--n-test", type=int, default=6500)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    make(args.n_calib, seed=1).to_csv(os.path.join(args.out_dir, "calib_logits.csv"), index=False)
    make(args.n_test, seed=2).to_csv(os.path.join(args.out_dir, "test_logits.csv"), index=False)
    print(f"Wrote synthetic calib_logits.csv and test_logits.csv to {args.out_dir}")
