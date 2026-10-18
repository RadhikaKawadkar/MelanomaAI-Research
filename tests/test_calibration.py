import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.calibration import (
    to_logit, fit_platt, apply_platt, fit_temperature, apply_temperature,
    fit_isotonic, apply_isotonic, prior_shift_logit, ece, brier, nll, bootstrap_ci,
)


def make_data(n=40000, seed=0):
    """True log-odds z; labels drawn from z; the 'model' reports a distorted version of z."""
    rng = np.random.default_rng(seed)
    z = rng.normal(-3.0, 2.0, n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(int)
    return z, y


class TestCalibration(unittest.TestCase):
    def test_to_logit_is_finite_at_extremes(self):
        self.assertTrue(np.isfinite(to_logit([0.0, 0.5, 1.0])).all())

    def test_temperature_recovers_known_value(self):
        z, y = make_data()
        self.assertAlmostEqual(fit_temperature(2.0 * z, y), 2.0, delta=0.15)

    def test_platt_recovers_known_scale_and_shift(self):
        z, y = make_data()
        a, b = fit_platt(2.0 * z + 0.5, y)          # truth: a=0.5, b=-0.25
        self.assertAlmostEqual(a, 0.5, delta=0.06)
        self.assertAlmostEqual(b, -0.25, delta=0.15)

    def test_isotonic_is_monotone_and_bounded(self):
        z, y = make_data()
        iso = fit_isotonic(z, y)
        out = apply_isotonic(iso, np.sort(z))
        self.assertTrue(np.all(np.diff(out) >= -1e-12))
        self.assertTrue(out.min() >= 0.0 and out.max() <= 1.0)

    def test_platt_and_temperature_keep_ranking(self):
        z, y = make_data(5000)
        a, b = fit_platt(z, y)
        T = fit_temperature(z, y)
        order = np.argsort(z)
        self.assertTrue(np.all(np.diff(apply_platt(z, a, b)[order]) >= 0))
        self.assertTrue(np.all(np.diff(apply_temperature(z, T)[order]) >= 0))

    def test_calibration_lowers_ece_for_distorted_model(self):
        z, y = make_data()
        raw_logits = 2.0 * z + 0.5
        raw_p = 1.0 / (1.0 + np.exp(-raw_logits))
        a, b = fit_platt(raw_logits, y)
        cal_p = apply_platt(raw_logits, a, b)
        self.assertGreater(ece(y, raw_p), 0.03)
        self.assertLess(ece(y, cal_p), 0.01)

    def test_prior_shift_is_identity_when_prevalence_unchanged(self):
        z = np.array([-2.0, 0.0, 3.0])
        np.testing.assert_allclose(prior_shift_logit(z, 0.3, 0.3), z)

    def test_prior_shift_value_for_balanced_training(self):
        self.assertAlmostEqual(float(prior_shift_logit(0.0, 0.5, 0.0184)), np.log(0.0184 / 0.9816), places=6)

    def test_perfect_predictions_have_zero_brier_and_nll(self):
        y = np.array([1, 0, 1, 0])
        p = np.array([1.0, 0.0, 1.0, 0.0])
        self.assertAlmostEqual(brier(y, p), 0.0)
        self.assertLess(nll(y, p), 1e-4)

    def test_bootstrap_interval_contains_estimate(self):
        z, y = make_data(3000)
        p = 1.0 / (1.0 + np.exp(-z))
        est, lo, hi = bootstrap_ci(brier, y, p, n_boot=200)
        self.assertLessEqual(lo, est)
        self.assertLessEqual(est, hi)

    def test_clustered_bootstrap_runs(self):
        z, y = make_data(2000)
        p = 1.0 / (1.0 + np.exp(-z))
        groups = np.repeat(np.arange(500), 4)
        est, lo, hi = bootstrap_ci(ece, y, p, groups=groups, n_boot=100)
        self.assertLessEqual(lo, hi)


if __name__ == "__main__":
    unittest.main()
