import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clinical_decision_analysis import (
    compute_calibration_curve,
    fit_platt_scaling,
    apply_platt_scaling,
    fit_isotonic_regression,
    apply_isotonic_regression,
    calculate_net_benefit,
    bootstrap_metrics,
)


class TestClinicalDecisionAnalysis(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_samples = 200
        # Simulated well-calibrated ground truth and probabilities
        self.y_true = np.random.binomial(1, 0.3, size=self.n_samples)
        noise = np.random.normal(0, 0.1, size=self.n_samples)
        self.y_prob = np.clip(0.3 * (1 - self.y_true) + 0.8 * self.y_true + noise, 0.01, 0.99)

    def test_calibration_curve_and_ece(self):
        calib = compute_calibration_curve(self.y_true, self.y_prob, n_bins=5)
        self.assertIn("expected_calibration_error", calib)
        self.assertIn("brier_score", calib)
        self.assertGreaterEqual(calib["expected_calibration_error"], 0.0)
        self.assertLessEqual(calib["expected_calibration_error"], 1.0)
        self.assertGreaterEqual(calib["brier_score"], 0.0)
        self.assertLessEqual(calib["brier_score"], 1.0)
        self.assertEqual(calib["total_samples"], self.n_samples)

    def test_platt_scaling(self):
        # Fit Platt on half, apply to other half
        mid = self.n_samples // 2
        calib_y, test_y = self.y_true[:mid], self.y_true[mid:]
        calib_p, test_p = self.y_prob[:mid], self.y_prob[mid:]

        lr = fit_platt_scaling(calib_p, calib_y)
        scaled_p = apply_platt_scaling(lr, test_p)

        self.assertEqual(len(scaled_p), len(test_p))
        self.assertTrue(np.all(scaled_p >= 0.0))
        self.assertTrue(np.all(scaled_p <= 1.0))

    def test_isotonic_regression(self):
        mid = self.n_samples // 2
        calib_y, test_y = self.y_true[:mid], self.y_true[mid:]
        calib_p, test_p = self.y_prob[:mid], self.y_prob[mid:]

        iso = fit_isotonic_regression(calib_p, calib_y)
        scaled_p = apply_isotonic_regression(iso, test_p)

        self.assertEqual(len(scaled_p), len(test_p))
        self.assertTrue(np.all(scaled_p >= 0.0))
        self.assertTrue(np.all(scaled_p <= 1.0))

    def test_decision_curve_analysis_net_benefit(self):
        thresholds = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
        # Test cohort DCA
        dca = calculate_net_benefit(self.y_true, self.y_prob, thresholds)
        self.assertEqual(len(dca["thresholds"]), len(thresholds))
        self.assertEqual(len(dca["net_benefit_model"]), len(thresholds))
        self.assertEqual(len(dca["net_benefit_treat_all"]), len(thresholds))
        self.assertFalse(dca["is_prevalence_adjusted"])

        # Test prevalence-adjusted DCA
        dca_prev = calculate_net_benefit(self.y_true, self.y_prob, thresholds, target_prevalence=0.03)
        self.assertTrue(dca_prev["is_prevalence_adjusted"])
        self.assertEqual(dca_prev["target_prevalence_used"], 0.03)

    def test_bootstrap_uncertainty_intervals(self):
        boot = bootstrap_metrics(self.y_true, self.y_prob, threshold=0.50, n_bootstraps=50)
        for metric in ["auc_roc", "sensitivity", "specificity", "precision", "brier_score"]:
            self.assertIn(metric, boot)
            self.assertLessEqual(boot[metric]["ci_lower"], boot[metric]["point_estimate"] + 0.05)
            self.assertGreaterEqual(boot[metric]["ci_upper"], boot[metric]["point_estimate"] - 0.05)


if __name__ == "__main__":
    unittest.main()
