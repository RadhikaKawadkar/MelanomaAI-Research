import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from risk_engine import compute_risk_score, score_predictions, estimate_abcd, get_skin_tone, estimate_ita


class TestRiskEngineEquitable(unittest.TestCase):
    def test_demographic_non_discrimination(self):
        """
        Verify that for identical model malignancy probabilities, darker skin tones
        never receive a lower risk score or lower risk tier than lighter skin tones.
        """
        probabilities = [0.10, 0.35, 0.50, 0.65, 0.85, 0.95]
        # Compare Very Light (ITA ~ 60) vs Dark (ITA ~ -40)
        ita_light = 60.0
        ita_dark = -40.0

        for p in probabilities:
            score_light = compute_risk_score(p, ita_light, threshold=0.50)
            score_dark = compute_risk_score(p, ita_dark, threshold=0.50)

            # Score for dark skin must be >= score for light skin (no penalization)
            self.assertGreaterEqual(
                score_dark["score"],
                score_light["score"],
                f"Demographic bias violation at prob={p}: Dark skin score {score_dark['score']} < Light skin score {score_light['score']}"
            )

            # If under-represented, uncertainty flag must be raised
            self.assertTrue(score_dark["under_represented_tone"])
            self.assertFalse(score_light["under_represented_tone"])

    def test_probability_monotonicity(self):
        """
        Verify that risk score increases monotonically with malignancy probability.
        """
        probs = np.linspace(0.0, 1.0, 50)
        ita = 35.0  # Intermediate
        prev_score = -1.0
        for p in probs:
            res = compute_risk_score(float(p), ita, threshold=0.50)
            self.assertGreaterEqual(res["score"], prev_score)
            prev_score = res["score"]

    def test_tier_boundaries(self):
        """
        Verify expected tier classifications at standard probability anchors.
        """
        low = compute_risk_score(0.10, 45.0, threshold=0.50)
        self.assertEqual(low["level"], "LOW")

        mod = compute_risk_score(0.40, 45.0, threshold=0.50)
        self.assertEqual(mod["level"], "MODERATE")

        high = compute_risk_score(0.60, 45.0, threshold=0.50)
        self.assertEqual(high["level"], "HIGH")

        crit = compute_risk_score(0.85, 45.0, threshold=0.50)
        self.assertEqual(crit["level"], "CRITICAL")

    def test_abcd_synthetic_disclosure(self):
        """
        Verify that ABCD estimation is explicitly disclosed as a synthetic proxy.
        """
        abcd = estimate_abcd(0.75)
        self.assertTrue(abcd.get("is_synthetic_proxy"))
        self.assertIn("disclaimer", abcd)
        self.assertTrue(abcd["asymmetry"]["is_synthetic"])
        self.assertTrue(abcd["border"]["is_synthetic"])
        self.assertTrue(abcd["color"]["is_synthetic"])
        self.assertTrue(abcd["diameter"]["is_synthetic"])

    def test_batch_scoring_fairness(self):
        """
        Verify batch scoring calculates subgroup mean probabilities cleanly.
        """
        probs = np.array([0.2, 0.7, 0.3, 0.8])
        itas = np.array([50.0, 50.0, 10.0, 10.0])  # 2 light, 2 dark
        res = score_predictions(probs, itas, threshold=0.50)

        self.assertEqual(res["fairness"]["light_skin_n"], 2)
        self.assertEqual(res["fairness"]["dark_skin_n"], 2)
        self.assertAlmostEqual(res["fairness"]["light_mean_prob"], 0.45)
        self.assertAlmostEqual(res["fairness"]["dark_mean_prob"], 0.55)
        self.assertAlmostEqual(res["fairness"]["prob_gap"], 0.10)


if __name__ == "__main__":
    unittest.main()
