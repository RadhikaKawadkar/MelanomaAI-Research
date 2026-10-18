"""
run_calibration_study.py -- compares probability calibration methods for MelanomaAI.

Methods compared: raw model output, prior-shift baseline (no fitting), Platt scaling,
temperature scaling, isotonic regression.

Input CSVs need: a logit column, a 0/1 label column. Optional: a subgroup column
(e.g. skin_group) and a cluster column (e.g. patient_id).

MODE A (preferred): fit on a separate calibration file, evaluate on a test file
    python run_calibration_study.py --calib calib_logits.csv --test test_logits.csv \
        --group-col skin_group --cluster-col patient_id

MODE B (fallback, no calibration file): patient-grouped K-fold cross-calibration on
the test file; every score is evaluated out-of-fold.
    python run_calibration_study.py --test test_logits.csv --cv-folds 5 \
        --group-col skin_group --cluster-col patient_id
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from backend import calibration as C

METHODS = ["raw", "prior_shift", "platt", "temperature", "isotonic"]


def calibrate_all(fit_z, fit_y, new_z, train_prev, deploy_prev=None):
    """Fit every calibrator on (fit_z, fit_y) and return probabilities for new_z."""
    fit_z, fit_y, new_z = np.asarray(fit_z, float), np.asarray(fit_y), np.asarray(new_z, float)
    if deploy_prev is None:
        deploy_prev = float(fit_y.mean())
    a, b = C.fit_platt(fit_z, fit_y)
    T = C.fit_temperature(fit_z, fit_y)
    iso = C.fit_isotonic(fit_z, fit_y)
    probs = {
        "raw": expit(new_z),
        "prior_shift": expit(C.prior_shift_logit(new_z, train_prev, deploy_prev)),
        "platt": C.apply_platt(new_z, a, b),
        "temperature": C.apply_temperature(new_z, T),
        "isotonic": C.apply_isotonic(iso, new_z),
    }
    params = {"platt_a": a, "platt_b": b, "temperature": T,
              "train_prev": train_prev, "deploy_prev": deploy_prev, "n_fit": int(len(fit_y))}
    return probs, params


def cross_calibrate(z, y, clusters, folds, train_prev):
    """Out-of-fold probabilities using patient-grouped K-fold."""
    out = {m: np.zeros(len(y)) for m in METHODS}
    all_params = []
    for tr, te in GroupKFold(n_splits=folds).split(z, y, groups=clusters):
        probs, params = calibrate_all(z[tr], y[tr], z[te], train_prev)
        for m in METHODS:
            out[m][te] = probs[m]
        all_params.append(params)
    return out, all_params


def evaluate(y, probs, clusters, n_boot):
    rows = []
    for m in METHODS:
        p = probs[m]
        row = {"method": m, "n": int(len(y)), "positives": int(y.sum()),
               "mean_pred": float(p.mean()), "observed_rate": float(y.mean()),
               "auc": float(roc_auc_score(y, p))}
        metrics = [("ece", C.ece), ("ece_quantile", lambda a, b: C.ece(a, b, strategy="quantile")),
                   ("brier", C.brier), ("nll", C.nll)]
        for name, fn in metrics:
            est, lo, hi = C.bootstrap_ci(fn, y, p, clusters, n_boot)
            row[name], row[name + "_lo"], row[name + "_hi"] = est, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def subgroup_table(y, probs, groups, clusters, n_boot):
    rows = []
    for g in sorted(pd.unique(groups)):
        mask = groups == g
        for m in METHODS:
            p = probs[m][mask]
            est, lo, hi = C.bootstrap_ci(C.ece, y[mask], p,
                                         None if clusters is None else clusters[mask], n_boot)
            rows.append({"group": g, "method": m, "n": int(mask.sum()),
                         "positives": int(y[mask].sum()), "mean_pred": float(p.mean()),
                         "observed_rate": float(y[mask].mean()),
                         "ece": est, "ece_lo": lo, "ece_hi": hi,
                         "brier": C.brier(y[mask], p)})
    return pd.DataFrame(rows)


def plot_reliability(y, probs, path, n_bins=10):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, zoom in zip(axes, [False, True]):
        top = 0.0
        for m in METHODS:
            p = probs[m]
            edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
            if len(edges) < 2:
                continue
            idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
            xs = [p[idx == b].mean() for b in range(len(edges) - 1) if (idx == b).any()]
            ys = [y[idx == b].mean() for b in range(len(edges) - 1) if (idx == b).any()]
            ax.plot(xs, ys, marker="o", label=m)
            top = max(top, max(xs), max(ys))
        lim = 0.25 if zoom else max(top * 1.05, 0.1)
        ax.plot([0, lim], [0, lim], "k--", label="perfect")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed melanoma rate")
        ax.set_title("Reliability (zoom 0-0.25)" if zoom else "Reliability (full range)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", help="CSV used to FIT calibrators (Mode A)")
    ap.add_argument("--test", required=True, help="CSV used to EVALUATE")
    ap.add_argument("--cv-folds", type=int, default=0, help="Mode B: grouped K-fold on --test")
    ap.add_argument("--logit-col", default="logit")
    ap.add_argument("--label-col", default="target")
    ap.add_argument("--group-col", default=None, help="subgroup column, e.g. skin_group")
    ap.add_argument("--cluster-col", default=None, help="e.g. patient_id (cluster bootstrap / CV)")
    ap.add_argument("--train-prev", type=float, default=0.5,
                    help="prevalence seen in training batches (0.5 if balanced)")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out-dir", default="outputs/calibration_study")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    test = pd.read_csv(args.test)
    z = test[args.logit_col].to_numpy(float)
    y = test[args.label_col].to_numpy(int)
    clusters = test[args.cluster_col].to_numpy() if args.cluster_col else None
    groups = test[args.group_col].to_numpy() if args.group_col else None

    if args.calib:
        calib = pd.read_csv(args.calib)
        if args.cluster_col and args.cluster_col in calib:
            overlap = set(calib[args.cluster_col]) & set(test[args.cluster_col])
            if overlap:
                raise SystemExit(f"LEAKAGE: {len(overlap)} {args.cluster_col} values appear in both files.")
        probs, params = calibrate_all(calib[args.logit_col].to_numpy(float),
                                      calib[args.label_col].to_numpy(int), z, args.train_prev)
        mode = "separate_calibration_set"
    elif args.cv_folds >= 2 and clusters is not None:
        probs, params = cross_calibrate(z, y, clusters, args.cv_folds, args.train_prev)
        mode = f"grouped_{args.cv_folds}fold_cross_calibration"
    else:
        raise SystemExit("Give --calib, or --cv-folds >= 2 together with --cluster-col.")

    overall = evaluate(y, probs, clusters, args.n_boot)
    overall.to_csv(os.path.join(args.out_dir, "overall_results.csv"), index=False)
    if groups is not None:
        subgroup_table(y, probs, groups, clusters, args.n_boot).to_csv(
            os.path.join(args.out_dir, "subgroup_results.csv"), index=False)
    plot_reliability(y, probs, os.path.join(args.out_dir, "reliability.png"))
    with open(os.path.join(args.out_dir, "calibration_params.json"), "w") as f:
        json.dump({"mode": mode, "params": params, "test_file": os.path.basename(args.test)}, f, indent=2)

    pd.set_option("display.width", 160)
    print(f"\nMode: {mode}   n={len(y)}   positives={int(y.sum())}   prevalence={y.mean():.4f}\n")
    print(overall[["method", "auc", "ece", "ece_lo", "ece_hi", "brier", "nll"]].round(4).to_string(index=False))
    print(f"\nSaved results to {args.out_dir}")


if __name__ == "__main__":
    main()
