# Clinical Utility Evaluation — Results & Evidence Report

**Project:** MelanomaAI Clinical Decision Utility Research  
**Branch:** `research/clinical-utility`  
**Execution Date:** October 18, 2026  
**Status:** Code Pipeline Verified & Calibrated; Empirical Evaluation Blocked on Checkpoint/Data Ingestion  

---

## 1. Verified Codebase Corrections

| Component | Previous Issue | Applied Fix | Verification Status |
| :--- | :--- | :--- | :--- |
| **Risk Scoring (`risk_engine.py`)** | Penalized dark skin tones via `tone_pts * prob`. | Monotonic mapping from probability; explicit demographic uncertainty flag for $ITA \le 28$. | **PASSED** (`tests/test_risk_engine.py`) |
| **ABCD Scoring (`risk_engine.py`)** | Misleading linear proxy presented as morphological measurement. | Formally flagged with `is_synthetic_proxy = True` and disclaimer. | **PASSED** (`tests/test_risk_engine.py`) |
| **Data Partitioning (`split_dataset.py`)** | Random split allowed augmented duplicates into validation. | Grouped lesion splitting with zero overlap and synthetic image exclusion. | **PASSED** (`tests/test_split_dataset.py`) |
| **Model Selection (`evaluate_model.py`)** | Selected candidate models by file byte size. | Replaced with explicit priority list and state dictionary validation. | **PASSED** (code reviewed & updated) |
| **Resolution Alignment (`evaluate_model.py`)** | Mismatch ($224 \times 224$ vs $384 \times 384$). | Default aligned to $384 \times 384$ with configurable environment override. | **PASSED** (code reviewed & updated) |
| **Clinical Calibration & DCA (`clinical_decision_analysis.py`)** | Missing calibration and decision analysis engines. | Implemented Brier score, ECE, Platt scaling, Isotonic regression, and DCA with prevalence adjustment. | **PASSED** (`tests/test_clinical_decision_analysis.py`) |

---

## 2. Unit Test Results

Automated unit test suite executed on host environment:

```text
Ran 12 tests in 3 test suites:
- tests/test_risk_engine.py ................. [5/5 PASSED]
  * test_demographic_non_discrimination (verified dark skin never penalized)
  * test_probability_monotonicity (verified strict monotonicity)
  * test_tier_boundaries (verified standard anchor tiers)
  * test_abcd_synthetic_disclosure (verified synthetic disclosure flag)
  * test_batch_scoring_fairness (verified subgroup gap calculation)

- tests/test_clinical_decision_analysis.py .. [5/5 PASSED]
  * test_calibration_curve_and_ece (verified ECE & Brier score)
  * test_platt_scaling (verified logistic calibration on logits)
  * test_isotonic_regression (verified isotonic mapping)
  * test_decision_curve_analysis_net_benefit (verified DCA with & without prevalence adjustment)
  * test_bootstrap_uncertainty_intervals (verified 95% bootstrap CI calculation)

- tests/test_split_dataset.py ............... [2/2 PASSED]
  * test_grouped_partition_zero_overlap (verified 0% lesion overlap between train, calib, test)
  * test_filter_leak_free_images (verified aug_* files strictly excluded from evaluation)

Result: 12 passed in 0.45s (100% PASS RATE)
```

---

## 3. Empirical Model Evaluation Status

- **Status**: **BLOCKED (Awaiting Artifact Ingestion)**
- **Reason**: The model checkpoint (`models/melanoma_final.pth`) and dataset images (`data/`) were not checked into the Git repository and do not exist on the local system drive.
- **Scientific Integrity Protocol**: In accordance with the Master Prompt instructions, no fabricated predictions or simulated performance tables have been entered as real experimental results.

---

## 4. Methodological Conclusions Supported by Current Evidence

1. **Previous Claims Invalid for Clinical Use**: The previously cited metrics (92.8% Accuracy, 0.9770 AUC) cannot be claimed as validated clinical performance due to the documented train/validation leakage in `balance_dataset.py` and `auto_train.py`.
2. **Clinical Decision Utility Requires True Calibration**: Raw outputs from Focal Loss training must be recalibrated on an independent calibration split before calculating Net Benefit.
3. **Prevalence Adjustment is Essential**: Applying DCA directly to an enriched 1:1 or 3:1 case-control dataset produces grossly inflated net benefit curves. The newly implemented DCA engine will apply prevalence adjustment ($P = 2\% - 5\%$) once data is available.

---

## 5. Immediate Next Step

To generate empirical DCA curves, ECE plots, and calibrated metrics:
- Place the existing model checkpoint at `C:\DS_CP\models\melanoma_final.pth` and the test dataset at `C:\DS_CP\data\`.
- Execute:
  ```bash
  python evaluate_model.py
  ```
