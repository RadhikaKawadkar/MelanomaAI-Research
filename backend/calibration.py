"""
calibration.py -- post-hoc probability calibration for MelanomaAI.

All functions work on LOGITS (raw scores before the sigmoid), because logits keep
precision that probabilities lose when they saturate near 0 or 1.

Three calibrators are provided:
  - Platt scaling       : p = sigmoid(a * logit + b)   (2 parameters)
  - Temperature scaling : p = sigmoid(logit / T)        (1 parameter)
  - Isotonic regression : free-form monotonic mapping   (non-parametric)

Always FIT on a calibration set and EVALUATE on a separate test set.
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

EPS = 1e-6


def to_logit(p):
    """Convert probabilities to logits (clipped so 0 and 1 do not give infinity)."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _nll(scores, y):
    """Mean negative log-likelihood of sigmoid(scores) against labels y (0/1)."""
    y = np.asarray(y, dtype=float)
    # logaddexp keeps this numerically stable for very large or small scores
    return float(np.mean(y * np.logaddexp(0.0, -scores) + (1.0 - y) * np.logaddexp(0.0, scores)))


# ---------------- Platt scaling ----------------
def fit_platt(logits, y):
    """Fit a and b so that sigmoid(a * logit + b) matches the labels. Returns (a, b)."""
    z = np.asarray(logits, dtype=float)
    result = minimize(lambda w: _nll(w[0] * z + w[1], y), x0=[1.0, 0.0], method="L-BFGS-B")
    return float(result.x[0]), float(result.x[1])


def apply_platt(logits, a, b):
    return expit(a * np.asarray(logits, dtype=float) + b)


# ---------------- Temperature scaling ----------------
def fit_temperature(logits, y):
    """Fit a single temperature T > 0. T > 1 softens overconfident scores. Returns T."""
    z = np.asarray(logits, dtype=float)
    result = minimize_scalar(lambda T: _nll(z / T, y), bounds=(0.05, 20.0), method="bounded")
    return float(result.x)


def apply_temperature(logits, T):
    return expit(np.asarray(logits, dtype=float) / T)


# ---------------- Isotonic regression ----------------
def fit_isotonic(logits, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(logits, dtype=float), np.asarray(y, dtype=float))
    return iso


def apply_isotonic(iso, logits):
    return iso.predict(np.asarray(logits, dtype=float))


# ---------------- Prior-shift baseline (no fitting) ----------------
def prior_shift_logit(logits, train_prev, deploy_prev):
    """
    Shift logits for a change in class prevalence between training and deployment.
    Example: trained on 50/50 batches, deployed where 1.84% are melanoma:
        shift = log(0.0184 / 0.9816) - log(0.5 / 0.5)  =  about -3.98
    Needs no data to fit, so it shows how much miscalibration is just prevalence.
    """
    def log_odds(q):
        return np.log(q / (1.0 - q))
    return np.asarray(logits, dtype=float) + log_odds(deploy_prev) - log_odds(train_prev)


# ---------------- Calibration metrics ----------------
def ece(y, p, n_bins=10, strategy="uniform"):
    """
    Expected Calibration Error: weighted average gap between predicted probability
    and observed event rate across bins.
      strategy="uniform"  -> equal-width bins (standard)
      strategy="quantile" -> equal-count bins (better when most predictions are tiny)
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        return float(abs(p.mean() - y.mean()))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    total = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def brier(y, p):
    """Mean squared error between probability and outcome (lower is better)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def nll(y, p):
    """Negative log-likelihood / log loss (lower is better)."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def bootstrap_ci(metric, y, p, groups=None, n_boot=1000, seed=0):
    """
    95% bootstrap confidence interval for metric(y, p). Returns (estimate, low, high).
    If `groups` is given (e.g. patient_id), whole groups are resampled together, so
    several images of one patient are not treated as independent.
    NOTE: ECE is biased upward in finite samples, so for a well-calibrated model the
    percentile interval can sit slightly ABOVE the point estimate. Treat Brier and NLL
    as the primary scores and ECE as a secondary, descriptive one.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    n = len(y)
    if groups is None:
        draws = (rng.integers(0, n, n) for _ in range(n_boot))
    else:
        _, inverse = np.unique(np.asarray(groups), return_inverse=True)
        members = [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]

        def one_draw():
            picks = rng.integers(0, len(members), len(members))
            return np.concatenate([members[k] for k in picks])
        draws = (one_draw() for _ in range(n_boot))
    values = [metric(y[i], p[i]) for i in draws]
    low, high = np.percentile(values, [2.5, 97.5])
    return float(metric(y, p)), float(low), float(high)
