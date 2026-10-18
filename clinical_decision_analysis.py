"""
clinical_decision_analysis.py — Clinical Utility & Decision Curve Analysis Engine
================================================================================
Implements rigorous statistical calibration, Decision Curve Analysis (DCA),
and prevalence-adjusted utility modeling for MelanomaAI.

References:
  - Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis: a novel method
    for evaluating prediction models. Medical Decision Making, 26(6), 565-574.
  - Van Calster, B., et al. (2019). Reporting and interpreting decision curve
    analysis: a guide for investigators. European Urology, 74(6), 796-804.
  - Marchetti, M. A., et al. (2020). Performance of artificial intelligence for
    melanoma diagnosis in clinical practice: a systematic review. JAMA Dermatology.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


# ─────────────────────────────────────────────────────────────────────────────
# 1. PROBABILITY CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """
    Computes empirical reliability diagram coordinates and Expected Calibration Error (ECE).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    emp_probs = []
    bin_counts = []
    ece = 0.0
    total_n = len(y_true)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob <= high if i == n_bins - 1 else y_prob < high)
        n_in_bin = int(mask.sum())
        if n_in_bin > 0:
            mean_pred = float(np.mean(y_prob[mask]))
            mean_true = float(np.mean(y_true[mask]))
            bin_centers.append(round(mean_pred, 4))
            emp_probs.append(round(mean_true, 4))
            bin_counts.append(n_in_bin)
            ece += (n_in_bin / total_n) * abs(mean_pred - mean_true)

    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "bin_centers": bin_centers,
        "empirical_probabilities": emp_probs,
        "bin_counts": bin_counts,
        "expected_calibration_error": round(float(ece), 4),
        "brier_score": round(brier, 4),
        "total_samples": total_n,
    }


def fit_platt_scaling(
    calib_probs: np.ndarray, calib_labels: np.ndarray
) -> LogisticRegression:
    """
    Fits Platt scaling (logistic regression on logit or prob) on a dedicated calibration split.
    """
    calib_probs = np.asarray(calib_probs, dtype=float).clip(1e-6, 1.0 - 1e-6)
    logits = np.log(calib_probs / (1.0 - calib_probs)).reshape(-1, 1)
    lr = LogisticRegression(C=1.0, solver="lbfgs")
    lr.fit(logits, calib_labels)
    return lr


def apply_platt_scaling(
    lr: LogisticRegression, test_probs: np.ndarray
) -> np.ndarray:
    """
    Applies fitted Platt scaler to test probabilities.
    """
    test_probs = np.asarray(test_probs, dtype=float).clip(1e-6, 1.0 - 1e-6)
    logits = np.log(test_probs / (1.0 - test_probs)).reshape(-1, 1)
    return lr.predict_proba(logits)[:, 1]


def fit_isotonic_regression(
    calib_probs: np.ndarray, calib_labels: np.ndarray
) -> IsotonicRegression:
    """
    Fits non-parametric isotonic regression on calibration split.
    """
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(calib_probs, calib_labels)
    return iso


def apply_isotonic_regression(
    iso: IsotonicRegression, test_probs: np.ndarray
) -> np.ndarray:
    """
    Applies fitted isotonic regressor to test probabilities.
    """
    return iso.predict(test_probs)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DECISION CURVE ANALYSIS (DCA)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    target_prevalence: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculates Net Benefit across a range of decision threshold probabilities (p_t).

    Formula:
        Net Benefit = (TP / N) - (FP / N) * (p_t / (1 - p_t))

    If target_prevalence is specified (e.g. 0.02 for primary care melanoma screening):
        We adjust TP and FP rates using Bayes theorem to match population prevalence,
        preventing biased Net Benefit from artificially enriched case-control cohorts:
            TP_adj = Sensitivity(p_t) * target_prevalence
            FP_adj = (1 - Specificity(p_t)) * (1 - target_prevalence)
            Net Benefit_adj = TP_adj - FP_adj * (p_t / (1 - p_t))

    Returns:
        Dictionary with curves for Model, Treat All, and Treat None.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    total_n = len(y_true)
    sample_prev = float(np.mean(y_true))
    effective_prev = target_prevalence if target_prevalence is not None else sample_prev

    model_nb = []
    treat_all_nb = []
    treat_none_nb = []
    sensitivities = []
    specificities = []
    number_needed_to_biopsy = []

    for pt in thresholds:
        if pt <= 0.0 or pt >= 1.0:
            continue
        weight = pt / (1.0 - pt)
        preds = (y_prob >= pt).astype(int)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivities.append(float(sens))
        specificities.append(float(spec))

        if target_prevalence is not None:
            # Prevalence-adjusted rates
            tp_rate = sens * target_prevalence
            fp_rate = (1.0 - spec) * (1.0 - target_prevalence)
            nb = tp_rate - fp_rate * weight
            all_nb = target_prevalence - (1.0 - target_prevalence) * weight
            # Number needed to biopsy to catch 1 true melanoma
            nnb = (tp_rate + fp_rate) / tp_rate if tp_rate > 0 else float("inf")
        else:
            # Empirical cohort rates
            nb = (tp / total_n) - (fp / total_n) * weight
            all_nb = sample_prev - (1.0 - sample_prev) * weight
            nnb = (tp + fp) / tp if tp > 0 else float("inf")

        model_nb.append(float(nb))
        treat_all_nb.append(float(all_nb))
        treat_none_nb.append(0.0)
        number_needed_to_biopsy.append(round(float(nnb), 2) if nnb < 1000 else None)

    return {
        "thresholds": [round(float(t), 4) for t in thresholds if 0.0 < t < 1.0],
        "net_benefit_model": [round(v, 5) for v in model_nb],
        "net_benefit_treat_all": [round(v, 5) for v in treat_all_nb],
        "net_benefit_treat_none": treat_none_nb,
        "sensitivity": [round(v, 4) for v in sensitivities],
        "specificity": [round(v, 4) for v in specificities],
        "number_needed_to_biopsy": number_needed_to_biopsy,
        "sample_prevalence": round(sample_prev, 4),
        "target_prevalence_used": round(effective_prev, 4),
        "is_prevalence_adjusted": bool(target_prevalence is not None),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. STATISTICAL BOOTSTRAPPING FOR CLINICAL UNCERTAINTY
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Computes point estimates and 95% bootstrap confidence intervals for key clinical metrics.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    metrics_boot: Dict[str, List[float]] = {
        "auc_roc": [],
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "f1": [],
        "brier_score": [],
    }

    preds = (y_prob >= threshold).astype(int)
    point_estimates = _calculate_scalar_metrics(y_true, y_prob, preds)

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt_sample = y_true[idx]
        yp_sample = y_prob[idx]
        if len(np.unique(yt_sample)) < 2:
            continue
        p_sample = (yp_sample >= threshold).astype(int)
        sample_metrics = _calculate_scalar_metrics(yt_sample, yp_sample, p_sample)
        for k, v in sample_metrics.items():
            metrics_boot[k].append(v)

    summary: Dict[str, Dict[str, float]] = {}
    lower_pct = (alpha / 2.0) * 100
    upper_pct = (1.0 - alpha / 2.0) * 100

    for k in metrics_boot:
        vals = np.array(metrics_boot[k])
        if len(vals) > 0:
            summary[k] = {
                "point_estimate": round(point_estimates[k], 4),
                "ci_lower": round(float(np.percentile(vals, lower_pct)), 4),
                "ci_upper": round(float(np.percentile(vals, upper_pct)), 4),
            }
        else:
            summary[k] = {
                "point_estimate": round(point_estimates[k], 4),
                "ci_lower": round(point_estimates[k], 4),
                "ci_upper": round(point_estimates[k], 4),
            }

    return summary


def _calculate_scalar_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, preds: np.ndarray
) -> Dict[str, float]:
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5
    brier = brier_score_loss(y_true, y_prob)

    return {
        "auc_roc": float(auc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "brier_score": float(brier),
    }
