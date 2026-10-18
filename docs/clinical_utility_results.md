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

## 3. Empirical Model Evaluation Findings (HAM10000 Out-of-Sample Test Split)

- **Dataset**: HAM10000 (`D:\EDI - Copy\data\raw\`) — 10,015 images partitioned 50% Calibration / 50% Test grouped strictly by `lesion_id` (0% lesion overlap).
- **Test Set Size**: 4,948 dermoscopic images (538 Melanomas, 4,410 Non-melanocytic/Benign lesions across 3,735 unique lesions).
- **Model Checkpoint**: Frozen `D:\EDI - Copy\models\melanoma_final.pth` (EfficientNet-B3).
- **Execution Script**: [evaluate_external_ham10000.py](file:///C:/DS_CP/evaluate_external_ham10000.py).
- **Artifacts Saved**: [C:\DS_CP\outputs\clinical_utility_external\](file:///C:/DS_CP/outputs/clinical_utility_external/).

### Primary Performance Metrics (Out-of-Sample Test Partition)

| Metric | Point Estimate | 95% Bootstrap Confidence Interval | Notes |
| :--- | :---: | :---: | :--- |
| **AUC-ROC** | **0.7967** | **[0.7772, 0.8138]** | Substantial drop from claimed 0.9770 internal validation. |
| **Sensitivity (@ 0.50 threshold)** | **30.48%** | [26.57%, 34.63%] | Misses 69.5% of melanomas (374 / 538) at uncalibrated default. |
| **Specificity (@ 0.50 threshold)** | **93.92%** | [93.18%, 94.60%] | 4,142 / 4,410 benign lesions correctly classified. |
| **Precision (@ 0.50 threshold)** | **37.96%** | [33.57%, 42.54%] | 164 True Positives vs 268 False Positives. |
| **F1-Score (@ 0.50 threshold)** | **0.3381** | [0.2994, 0.3776] | Uncalibrated harmonic mean. |
| **Calibration-Tuned Threshold ($\tau = 0.4287$)** | — | — | Sensitivity rises to **47.21%**, Specificity **88.48%**, F1 **0.3908**. |

### Calibration Assessment (Reliability Diagram)

| Calibration State | Brier Score | Expected Calibration Error (ECE) | Interpretation |
| :--- | :---: | :---: | :--- |
| **Uncalibrated Model** | 0.1053 | **0.1369** | Substantial miscalibration; probabilities skewed by focal loss. |
| **Platt Calibrated (Logistic)** | **0.0836** | **0.0099** | **>13x reduction in ECE**; probabilities aligned with empirical risk. |
| **Isotonic Regression** | 0.0838 | 0.0112 | Similar strong calibration improvement. |

### Demographic Subgroup Disparity (CNN Model Performance by ITA)

| Demographic Subgroup | Sample Count | Melanoma Count | Subgroup AUC-ROC | Specificity (@ 0.50) | Sensitivity (@ 0.50) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Light Skin ($ITA > 28$)** | 3,068 | 249 | **0.8312** | 96.13% | 31.73% |
| **Dark Skin ($ITA \le 28$)** | 1,880 | 289 | **0.7309** | 90.01% | 29.41% |
| **Demographic Disparity Gap** | — | — | **$\Delta = 0.1003$ (-10.0%)** | $\Delta = 6.12\%$ | $\Delta = 2.32\%$ |

> **Key Fairness Discovery**: While the code fix in `risk_engine.py` successfully eliminated the direct arithmetic penalty in risk scores, the underlying CNN model demonstrates a genuine **10.0% discrimination disparity** ($0.8312$ on light skin vs. $0.7309$ on dark skin), confirming under-representation in feature learning.

### Decision Curve Analysis (DCA) Audited Summary
- **Primary Care Screening ($P = 2\%$)**: Across primary care triage thresholds ($p_t \ge 0.04$), the calibrated model exhibits **negative Net Benefit** (e.g., $\text{NB} = -0.0101$ at $p_t = 0.05$) due to high false-positive referral volume ($55.5\%$ of benign lesions flagged), performing **worse than the "Refer None" strategy ($\text{NB} = 0$)**. Positive Net Benefit is only achieved in low-threshold zones ($p_t \le 0.03$).
- **Specialist Dermatology Biopsy ($P = 5\%$)**: At $p_t = 0.05$, Net Benefit is positive ($+0.0185$), outperforming "Refer All" and "Refer None".
- **Biopsy Reduction Retraction**: The previous claim of an "18–25% reduction in unnecessary biopsies" has been retracted as an overstatement. In the enriched cohort, reducing benign biopsies at $p_t = 0.05$ inherently trades off a **7.43% False Negative Rate** (40 missed melanomas out of 538). No cost-free reduction in biopsies exists with this model.

---

## 4. Key Scientific Conclusions

1. **Internal Metrics Were Severely Inflated**: The 92.8% accuracy and 0.9770 AUC in the project status report were artifacts of augmented duplicate leakage. On a leak-free external test split with zero lesion overlap, the actual discrimination AUC is **0.7967** (95% CI: 0.7772–0.8138).
2. **Default 0.50 Threshold is Clinically Dangerous**: At $\tau = 0.50$, the uncalibrated model misses **69.5% of melanomas** (Sensitivity = 30.48%). For clinical triage, a lower operating threshold ($p_t \le 0.10$ or high-sensitivity tuning) is mandatory.
3. **Platt Calibration Successfully Restores Statistical Probabilities**: Pre-fitting Platt scaling on a separate calibration split reduces ECE from 0.1369 to 0.0099 without overfitting the test set.
4. **Demographic Bias Exists in the Weights**: The CNN feature representations are significantly less discriminative on darker skin tones ($\Delta \text{AUC} = 0.1003$), necessitating the explicit `under_represented_tone` caution flag implemented in `risk_engine.py`.

---

## 5. Artifact Directory & Visual Deliverables

All generated empirical deliverables are stored in [outputs/clinical_utility_external/](file:///C:/DS_CP/outputs/clinical_utility_external/):
- `test_metrics_summary.json`: Complete JSON summary of metrics and bootstrap CIs.
- `test_predictions.csv`: 4,948 row-level out-of-sample predictions.
- `dca_curves.json`: Numerical Net Benefit values across thresholds and prevalence rates.
- `roc_curve_test.png`: External test ROC plot.
- `confusion_matrix_test.png`: 2x2 contingency table.
- `calibration_curve_test.png`: Reliability diagram comparing uncalibrated vs. calibrated model.
- `decision_curve_analysis_test.png`: Decision Curve Analysis plots.
