import unittest
import numpy as np
import pandas as pd
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clinical_decision_analysis import calculate_net_benefit

class TestGap8ClinicalUtility(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.gap8_dir = os.path.join(self.base_dir, "outputs", "gap8_clinical_utility")
        self.clean_dir = os.path.join(self.base_dir, "outputs", "clean_retraining")

    def test_dca_mathematical_identities(self):
        """Verifies that DCA formulas strictly follow Vickers & Elkin (2006)."""
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # N=10, 2 pos, 8 neg, prev=0.20
        y_prob = np.array([0.9, 0.8, 0.4, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.02])

        # Test at threshold equal to prevalence: p_t = 0.20
        # When p_t = prevalence, Treat All Net Benefit must equal exactly 0.0
        pt = 0.20
        w = pt / (1.0 - pt)  # 0.20 / 0.80 = 0.25
        res = calculate_net_benefit(y_true, y_prob, np.array([pt]))

        self.assertAlmostEqual(res["net_benefit_treat_none"][0], 0.0, places=5)
        self.assertAlmostEqual(res["net_benefit_treat_all"][0], 0.0, places=5)

        # Model at 0.20 threshold: TP=2, FP=3 (indices 0, 1, 2, 3, 4)
        # NB = (2 / 10) - (3 / 10) * 0.25 = 0.20 - 0.075 = 0.125
        self.assertAlmostEqual(res["net_benefit_model"][0], 0.125, places=5)

    def test_biopsy_reduction_formula(self):
        """Verifies that Net Reduction in Biopsies formula correctly equates to (TN/N) - (FN/N)*(1-pt)/pt."""
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.4, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.02])
        pt = 0.20
        w = pt / (1.0 - pt)

        res = calculate_net_benefit(y_true, y_prob, np.array([pt]))
        nb_model = res["net_benefit_model"][0]
        nb_all = res["net_benefit_treat_all"][0]

        # Biopsies reduced per 100 patients: (NB_model - NB_all) / w * 100
        reduction = (nb_model - nb_all) / w * 100
        # Here: (0.125 - 0.0) / 0.25 * 100 = 50.0
        self.assertAlmostEqual(reduction, 50.0, places=4)

    def test_partition_independence_and_no_leakage(self):
        """Confirms calibration and test partitions share 0% patients."""
        manifest_p = os.path.join(self.clean_dir, "clean_split_manifest.csv")
        self.assertTrue(os.path.exists(manifest_p), "Manifest missing")
        df = pd.read_csv(manifest_p)

        calib_pts = set(df[df["partition"] == "calib"]["patient_id"])
        test_pts = set(df[df["partition"] == "test"]["patient_id"])
        overlap = calib_pts & test_pts
        self.assertEqual(len(overlap), 0, "Patient leakage detected between calibration and test!")

    def test_gap8_outputs_exist_and_consistent(self):
        """Verifies that Gap 8 files exist, are non-empty, and hold valid data."""
        table_p = os.path.join(self.gap8_dir, "dca_threshold_table.csv")
        summary_p = os.path.join(self.gap8_dir, "gap8_dca_summary.json")
        ci_p = os.path.join(self.gap8_dir, "dca_bootstrap_ci.json")
        plot1_p = os.path.join(self.gap8_dir, "dca_net_benefit_curves.png")
        plot2_p = os.path.join(self.gap8_dir, "dca_biopsy_reduction_curves.png")

        for p in [table_p, summary_p, ci_p, plot1_p, plot2_p]:
            self.assertTrue(os.path.exists(p), f"Missing artifact: {p}")
            self.assertGreater(os.path.getsize(p), 0, f"Empty artifact: {p}")

        # Check summary numbers at pt = 0.02
        with open(summary_p, "r") as f:
            summary = json.load(f)

        m02 = summary["metrics_at_pt_002"]
        self.assertGreater(m02["net_benefit_model_platt"], 0.0)
        self.assertGreater(m02["net_benefit_model_platt"], m02["net_benefit_treat_all"])
        self.assertGreater(m02["net_biopsy_reduction_per_100"], 40.0)
        self.assertGreaterEqual(m02["sensitivity"], 0.90)

    def test_bootstrap_confidence_intervals_ordering(self):
        """Confirms that 95% bootstrap intervals satisfy lower <= mean <= upper."""
        ci_p = os.path.join(self.gap8_dir, "dca_bootstrap_ci.json")
        with open(ci_p, "r") as f:
            ci_data = json.load(f)

        for pt_str, entry in ci_data.items():
            for mode in ["image_level", "patient_clustered"]:
                sub = entry[mode]
                self.assertLessEqual(sub["nb_platt_ci_lower"], sub["nb_platt_mean"])
                self.assertLessEqual(sub["nb_platt_mean"], sub["nb_platt_ci_upper"])
                self.assertLessEqual(sub["nb_all_ci_lower"], sub["nb_all_mean"])
                self.assertLessEqual(sub["nb_all_mean"], sub["nb_all_ci_upper"])


if __name__ == "__main__":
    unittest.main()
