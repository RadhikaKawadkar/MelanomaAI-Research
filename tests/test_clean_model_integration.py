"""
Unit tests for clean model integration, preprocessing resolution,
checkpoint discovery, Platt probability calibration, and error handling.
"""

import os
import unittest
import numpy as np
import torch

from backend.model import (
    IMG_SIZE,
    val_tf,
    build_model,
    load_model,
    calibrate_probability,
    predict,
    predict_image,
    PLATT_COEFF_A,
    PLATT_COEFF_B,
    PLATT_AUDIT_A,
    PLATT_AUDIT_B,
)


class TestCleanModelIntegration(unittest.TestCase):

    def test_resolution_and_preprocessing_transforms(self):
        """Confirm 384x384 resolution and ImageNet normalization."""
        self.assertEqual(IMG_SIZE, 384, "IMG_SIZE must be 384 for clean model.")
        dummy_img = np.zeros((500, 700, 3), dtype=np.uint8)
        transformed = val_tf(image=dummy_img)["image"]
        self.assertEqual(
            transformed.shape,
            (3, 384, 384),
            f"Expected (3, 384, 384) tensor, got {transformed.shape}"
        )
        self.assertEqual(transformed.dtype, torch.float32)

    def test_platt_calibration_matches_clean_evaluation(self):
        """Confirm calibration reproduces saved test_predictions.csv."""
        test_pred_path = os.path.join(
            os.path.dirname(__file__), "..", "outputs", "clean_retraining", "test_predictions.csv"
        )
        if os.path.exists(test_pred_path):
            import pandas as pd
            df = pd.read_csv(test_pred_path)
            # Sample 50 points across different risk levels
            sample = df.iloc[::100][["prob_raw", "prob_platt"]].copy()
            for _, row in sample.iterrows():
                p_raw = float(row["prob_raw"])
                p_expected = float(row["prob_platt"])
                p_app = calibrate_probability(p_raw, a=PLATT_COEFF_A, b=PLATT_COEFF_B)
                self.assertAlmostEqual(
                    p_app, p_expected, places=6,
                    msg=f"Platt calibration mismatch for raw={p_raw}: app={p_app}, expected={p_expected}"
                )

    def test_platt_calibration_with_audit_coefficients(self):
        """Verify calibration works with audit coefficients A=0.7410, B=-1.9791."""
        # For p = 0.5, logit = 0.0 -> calib = 1 / (1 + exp(1.9791)) = 0.121408
        p_calib = calibrate_probability(0.5, a=PLATT_AUDIT_A, b=PLATT_AUDIT_B)
        expected = 1.0 / (1.0 + np.exp(1.9791))
        self.assertAlmostEqual(p_calib, expected, places=5)

    def test_calibration_clipping_and_boundaries(self):
        """Confirm robust clipping on extreme probabilities."""
        self.assertGreater(calibrate_probability(0.0), 0.0)
        self.assertLess(calibrate_probability(1.0), 1.0)
        # Monotonicity test
        probs = [0.001, 0.01, 0.05, 0.20, 0.50, 0.80, 0.99]
        calibs = [calibrate_probability(p) for p in probs]
        for i in range(len(calibs) - 1):
            self.assertLessEqual(calibs[i], calibs[i + 1], "Calibration must be strictly monotonic.")

    def test_model_loader_finds_clean_checkpoint(self):
        """Confirm model loader discovers clean_model_best.pth."""
        model, path = load_model()
        self.assertIsNotNone(model, "Model must load successfully.")
        self.assertIsNotNone(path, "Checkpoint path must not be None.")
        self.assertTrue(os.path.exists(path), f"Checkpoint path {path} must exist.")
        self.assertIn("clean_model_best.pth", path, "Loader should prioritize clean_model_best.pth.")

    def test_inference_pipeline_outputs_both_probabilities(self):
        """Confirm predict_image returns raw_probability, calibrated_probability, and respects threshold."""
        demo_path = os.path.join(
            os.path.dirname(__file__), "..", "demo_samples", "benign_ISIC_0024322.jpg"
        )
        if os.path.exists(demo_path):
            res_default = predict_image(demo_path, threshold=0.50, use_calibration=False)
            self.assertIn("raw_probability", res_default)
            self.assertIn("calibrated_probability", res_default)
            self.assertIn("probability", res_default)
            self.assertEqual(res_default["probability"], res_default["raw_probability"])
            self.assertIn("model_path", res_default)

            # Test explicit calibrated mode
            res_calib = predict_image(demo_path, threshold=0.02, use_calibration=True)
            self.assertEqual(res_calib["probability"], res_calib["calibrated_probability"])

    def test_error_handling_invalid_inputs(self):
        """Confirm graceful error handling for missing files."""
        with self.assertRaises(ValueError):
            predict_image("nonexistent_image_path_12345.jpg")


if __name__ == "__main__":
    unittest.main()
