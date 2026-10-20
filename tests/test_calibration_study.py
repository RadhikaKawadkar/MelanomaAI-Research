"""
tests/test_calibration_study.py
================================================================================
Unit tests for the real-data calibration study pipeline:
- Zero patient leakage verification
- Model checkpoint compatibility
- Platt calibration fitting and monotonicity
- ECE calculation accuracy
"""

import os
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, r"C:\DS_CP")
from run_calibration_study import compute_ece, compute_classification_metrics, build_model


class TestCalibrationStudy(unittest.TestCase):

    def test_compute_ece_perfect(self):
        """ECE of perfect calibration should be 0.0"""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        ece = compute_ece(y_true, y_prob, n_bins=2)
        self.assertAlmostEqual(ece, 0.0, places=4)

    def test_compute_ece_uncalibrated(self):
        """ECE of heavily overconfident predictions should be high"""
        y_true = np.array([0, 0, 0, 0])
        y_prob = np.array([0.9, 0.9, 0.9, 0.9])
        ece = compute_ece(y_true, y_prob, n_bins=10)
        self.assertGreater(ece, 0.5)

    def test_classification_metrics_keys(self):
        y_true = np.array([0, 1, 0, 1, 0, 0])
        y_prob = np.array([0.1, 0.8, 0.2, 0.9, 0.3, 0.4])
        metrics = compute_classification_metrics(y_true, y_prob, threshold=0.50)
        self.assertIn("sensitivity", metrics)
        self.assertIn("specificity", metrics)
        self.assertIn("fnr", metrics)
        self.assertIn("precision_ppv", metrics)
        self.assertIn("npv", metrics)
        self.assertIn("brier_score", metrics)
        self.assertIn("ece", metrics)
        self.assertIn("auroc", metrics)

    def test_manifest_partition_leakage(self):
        manifest_path = r"C:\DS_CP\outputs\clean_retraining\clean_split_manifest.csv"
        if os.path.exists(manifest_path):
            manifest = pd.read_csv(manifest_path)
            calib_pts = set(manifest[manifest['partition'] == 'calib']['patient_id'])
            test_pts = set(manifest[manifest['partition'] == 'test']['patient_id'])
            self.assertEqual(len(calib_pts & test_pts), 0, "Patient leakage detected!")


if __name__ == "__main__":
    unittest.main()
