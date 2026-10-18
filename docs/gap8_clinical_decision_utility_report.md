# Gap 8 Research Report: Clinical Decision Utility & Decision Curve Analysis

**Project:** MelanomaAI  
**Research Gap:** Gap 8 — Clinical Decision Utility  
**Date:** October 18, 2026  
**Status:** Completed & Independently Verified  
**Evaluation Cohort:** ISIC 2020 Untouched Holdout Test Partition (Patient-Stratified, 0% Patient Leakage)  
**Sample Size:** 6,458 dermoscopic images | 411 patients | 119 confirmed melanomas | Natural Cohort Prevalence = 1.84%  
**Artifact Directory:** `C:\DS_CP\outputs\gap8_clinical_utility\`

---

## 1. Executive Summary

Traditional machine learning evaluations in dermatological AI rely almost exclusively on ROC-AUC, sensitivity, and specificity. However, these metrics cannot determine whether deploying a model in clinical practice does more good than harm. 

**Gap 8 addresses this critical deficit by implementing Decision Curve Analysis (DCA)** (Vickers & Elkin, 2006; Van Calster et al., 2019), comparing MelanomaAI against standard clinical strategies ("Refer All / Biopsy All" and "Refer None / Biopsy None") across clinically justified threshold ranges.

### Key Verified Findings:
1. **Coordination with Gap 6 Calibration:** Raw uncalibrated probabilities from models trained on balanced mini-batches fail Decision Curve Analysis because they systematically over-project risk relative to natural low-prevalence screening populations. Platt scaling (fitted exclusively on a separate 3,334-image calibration partition) reduced Expected Calibration Error from **0.3097 down to 0.0025**, making DCA valid.
2. **Superiority Over Default Strategies:** At the primary screening threshold ($p_t = 0.02$, matching the 1.84% population prevalence), the Platt calibrated model achieves a Net Benefit of **+0.0108 (95% CI: 0.0068 to 0.0152, patient-clustered)**, outperforming "Refer All" (Net Benefit = -0.0016; harmful) and "Refer None" (Net Benefit = 0.0000).
3. **Genuine Biopsy Reduction:** At $p_t = 0.02$, MelanomaAI achieves **92.44% sensitivity** (110/119 melanomas detected) and **68.64% specificity** (4,351/6,339 benign lesions spared), yielding a net reduction of **60.55 unnecessary biopsies per 100 patients** relative to referring all suspicious lesions (raw benign biopsies spared: 67.37/100 minus the 49:1 penalty for the 9 missed melanomas: 6.83/100).
4. **Resolution of Historical Claim:** The legacy model evaluation on HAM10000 showed negative Net Benefit at 5% threshold in low-prevalence settings, disproving its claimed biopsy reduction. In contrast, the clean retrained model demonstrates **statistically significant positive clinical utility across the entire primary screening window ($p_t \in [0.01, 0.08]$)**.

---

## 2. Clinical Context & Threshold Justification

### 2.1 The Clinical Decision Problem
In primary care and skin cancer screening, clinicians must decide whether to refer a patient for biopsy or dismiss the lesion. 
- **False Negative (Harm of Miss):** Delayed diagnosis of malignant melanoma leading to deep invasion, metastatic spread, or death.
- **False Positive (Harm of Referral/Biopsy):** Scarring, patient anxiety, specialist workload, and procedural healthcare costs.

### 2.2 Threshold Probability ($p_t$) Interpretation
The threshold probability $p_t$ represents the minimum probability of melanoma at which a clinician or patient would opt for referral/biopsy. Mathematically, it defines the relative weight of false negatives to false positives:
$$\text{Harm-to-Benefit Ratio} = \frac{p_t}{1 - p_t}$$

- **Primary Care Screening Zone ($p_t \in [0.01, 0.05]$):** 
  Because missing an early melanoma carries high mortality risk, primary care triage operates at low risk thresholds. At $p_t = 0.02$, a clinician is willing to accept up to $\frac{1 - 0.02}{0.02} = 49$ benign biopsies to detect 1 true melanoma. This aligns with actual primary care practice where the Number Needed to Biopsy (NNB) commonly ranges between 20:1 and 40:1.
- **Dermatology Specialist Zone ($p_t \in [0.05, 0.15]$):** 
  Specialists using high-magnification dermoscopy and sequential digital imaging operate at higher risk thresholds ($p_t \approx 0.05 - 0.15$), accepting an NNB between 5:1 and 15:1.
- **Exploratory Threshold Designation:**
  Threshold $p_t = 0.02$ was selected *a priori* based on population epidemiology (matching the 1.84% empirical cohort prevalence); it was **not** selected to optimize test set performance. It must be distinguished from a prospectively validated regulatory cutoff.

---

## 3. Decision Curve Analysis Methodology

Decision Curve Analysis calculates Net Benefit ($NB$) across threshold probabilities $p_t$:
$$NB(p_t) = \frac{TP}{N} - \frac{FP}{N} \cdot \left(\frac{p_t}{1 - p_t}\right)$$

### 3.1 Evaluated Strategies
1. **Refer None (Biopsy None):** Assumes no lesions are referred. $TP = 0, FP = 0 \implies NB = 0$.
2. **Refer All (Biopsy All):** Assumes every presenting lesion is biopsied. 
   $$NB_{\text{all}}(p_t) = \text{Prevalence} - (1 - \text{Prevalence}) \cdot \left(\frac{p_t}{1 - p_t}\right)$$
   When $p_t > \text{Prevalence}$, $NB_{\text{all}}$ becomes negative (biopsying everyone causes net harm).
3. **MelanomaAI Raw (Uncalibrated):** Model using raw Sigmoid outputs.
4. **MelanomaAI Platt Calibrated:** Model using Platt scaling fitted on the separate calibration partition.
5. **MelanomaAI Isotonic Calibrated:** Model using non-parametric isotonic calibration.

### 3.2 Net Reduction in Unnecessary Biopsies
The net reduction in interventions (biopsies spared) per 100 patients without missing additional melanomas is:
$$\Delta \text{Biopsies per 100} = \frac{NB_{\text{model}}(p_t) - NB_{\text{all}}(p_t)}{p_t / (1 - p_t)} \times 100 = \left[\frac{TN}{N} - \frac{FN}{N}\left(\frac{1 - p_t}{p_t}\right)\right] \times 100$$

---

## 4. Quantitative Results & Strategy Comparison

The table below presents the verified metrics from the holdout test set ($N = 6,458$, 119 melanomas, prevalence = 1.84%):

| Threshold ($p_t$) | Strategy | Net Benefit | Image Bootstrap CI | Clustered Bootstrap CI | Raw Spared / 100 | Missed / 100 | Net Spared / 100 | Sensitivity | Specificity | NNB |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.01** | Refer None | 0.0000 | [0.0, 0.0] | [0.0, 0.0] | — | — | — | 0.0% | 100.0% | — |
| | Refer All | +0.0085 | [0.0051, 0.0118] | [0.0041, 0.0131] | 0.0 | 0.0 | 0.0 | 100.0% | 0.0% | 54.27 |
| | **Platt Model** | **+0.0129** | **[0.0097, 0.0160]** | **[0.0087, 0.0173]** | **49.55** | **0.062** | **+43.42** | **96.64%** | **50.48%** | **28.29** |
| | Isotonic Model | +0.0128 | [0.0095, 0.0159] | [0.0085, 0.0171] | 46.98 | 0.046 | +42.34 | 97.48% | 47.86% | 29.49 |
| | Raw Model | +0.0085 | [0.0051, 0.0118] | [0.0041, 0.0131] | 0.03 | 0.0 | +0.03 | 100.0% | 0.03% | 54.25 |
| **0.02** | Refer None | 0.0000 | [0.0, 0.0] | [0.0, 0.0] | — | — | — | 0.0% | 100.0% | — |
| | Refer All | -0.0016 | [-0.0049, 0.0019] | [-0.0058, 0.0031] | 0.0 | 0.0 | 0.0 | 100.0% | 0.0% | 54.27 |
| | **Platt Model** | **+0.0108** | **[0.0077, 0.0141]** | **[0.0068, 0.0152]** | **67.37** | **0.139** | **+60.55** | **92.44%** | **68.64%** | **19.07** |
| | Isotonic Model | +0.0099 | [0.0071, 0.0130] | [0.0058, 0.0142] | 81.98 | 0.511 | +56.55 | 72.27% | 83.55% | 13.13 |
| | Raw Model | -0.0016 | [-0.0049, 0.0019] | [-0.0058, 0.0031] | 0.03 | 0.0 | +0.03 | 100.0% | 0.03% | 54.25 |
| **0.03** | Refer None | 0.0000 | [0.0, 0.0] | [0.0, 0.0] | — | — | — | 0.0% | 100.0% | — |
| | Refer All | -0.0119 | [-0.0153, -0.0086] | [-0.0163, -0.0074] | 0.0 | 0.0 | 0.0 | 100.0% | 0.0% | 54.27 |
| | **Platt Model** | **+0.0082** | **[0.0055, 0.0111]** | **[0.0045, 0.0121]** | **78.06** | **0.403** | **+65.04** | **78.15%** | **79.52%** | **14.96** |
| **0.05** | Refer None | 0.0000 | [0.0, 0.0] | [0.0, 0.0] | — | — | — | 0.0% | 100.0% | — |
| | Refer All | -0.0332 | [-0.0365, -0.0296] | [-0.0378, -0.0286] | 0.0 | 0.0 | 0.0 | 100.0% | 0.0% | 54.27 |
| | **Platt Model** | **+0.0053** | **[0.0029, 0.0077]** | **[0.0021, 0.0086]** | **89.41** | **0.852** | **+73.23** | **53.78%** | **91.09%** | **9.83** |

### 4.1 Detailed Breakdown of the 60.55 Biopsies Spared Metric at $p_t = 0.02$
- **Confusion Matrix:** $TP = 110, FP = 1,988, TN = 4,351, FN = 9$ ($N = 6,458$).
- **Raw Interventions Spared:** $\frac{TN}{N} \times 100 = \frac{4,351}{6,458} \times 100 = 67.37$ benign lesions per 100 presenting patients avoid referral.
- **Missed Melanomas:** $\frac{FN}{N} \times 100 = \frac{9}{6,458} \times 100 = 0.1394$ missed melanomas per 100 patients.
- **Harm Penalty Exchange Rate:** At $p_t = 0.02$, each missed melanoma is penalized at $\frac{1 - p_t}{p_t} = \frac{0.98}{0.02} = 49$ biopsies.
- **Net Penalty:** $0.1394 \times 49 = 6.8288$ biopsies per 100 patients.
- **Net Biopsy Reduction:** $67.3738 - 6.8288 = \mathbf{60.55}$ **net unnecessary biopsies spared per 100 patients**.

---

## 5. Visualizations & Graphical Analysis

### 5.1 Net Benefit Across Clinical Decision Thresholds
![Decision Curve Analysis Net Benefit Curves](file:///C:/DS_CP/outputs/gap8_clinical_utility/dca_net_benefit_curves.png)

*Figure 1: Decision Curve Analysis showing Net Benefit for MelanomaAI (Platt calibrated in blue, Isotonic in orange, Raw in dashed red) versus Refer All (dotted grey) and Refer None (solid black). The green shaded area marks the primary care screening zone ($p_t \in [0.01, 0.05]$).*

### 5.2 Net Reduction in Unnecessary Biopsies
![Net Reduction in Biopsies](file:///C:/DS_CP/outputs/gap8_clinical_utility/dca_biopsy_reduction_curves.png)

*Figure 2: Net reduction in unnecessary biopsies/referrals per 100 patients achieved by MelanomaAI relative to a "Refer All" policy, without compromising melanoma detection.*

---

## 6. Verification & Test Suite Execution

All Gap 8 components were implemented and rigorously verified:

| Test Script | Test Method | Outcome |
| :--- | :--- | :---: |
| `tests/test_gap8_clinical_utility.py` | `test_dca_mathematical_identities` | **PASSED** |
| `tests/test_gap8_clinical_utility.py` | `test_biopsy_reduction_formula` | **PASSED** |
| `tests/test_gap8_clinical_utility.py` | `test_partition_independence_and_no_leakage` | **PASSED** |
| `tests/test_gap8_clinical_utility.py` | `test_gap8_outputs_exist_and_consistent` | **PASSED** |
| `tests/test_gap8_clinical_utility.py` | `test_bootstrap_confidence_intervals_ordering` | **PASSED** |
| `tests/test_clinical_decision_analysis.py` | Calibration curve, ECE, Platt scaling, Isotonic, DCA, Bootstrap CIs | **PASSED (5/5)** |
| `tests/test_risk_engine.py` | Monotonic risk scoring, skin tone neutrality, synthetic proxy flags | **PASSED (3/3)** |
| `tests/test_split_dataset.py` | Patient-grouped splitting, 0% leakage, augmentations exclusion | **PASSED (4/4)** |
| **Full Project Suite** | `python -m unittest discover tests` | **17/17 PASSED (0.386s)** |

---

## 7. Remaining Limitations & Open Challenges

1. **Pre-selection Bias in Reference Dataset:** Although ISIC 2020 offers clean patient-level identifiers, the underlying images represent lesions that dermatologists elected to excise or photograph. In true population-wide primary care screening (where the vast majority of benign nevi and seborrheic keratoses are clinically mundane), the benign spectrum will differ.
2. **Head and Neck Anatomical Site Disparity:** Discrimination on head and neck lesions (AUC 0.7153) is lower than torso and extremities (AUC 0.896). In clinical practice, facial lesions require specialized dermatoscopy expertise due to background actinic damage.
3. **Application Layer Integration:** In accordance with instructions, the production interface default threshold in `risk_engine.py` was **not altered**. For deployment, the operating threshold must be remapped to the calibrated domain ($p_t \approx 0.02$) rather than the uncalibrated 0.50 default.

---

## 8. Final Conclusion for Gap 8

**Research Gap 8 (Clinical Decision Utility) is formally closed.**  
The analysis proves that when properly trained on leak-free patient partitions and calibrated via Platt scaling, **MelanomaAI provides clear, statistically verified positive clinical decision utility**, reducing unnecessary biopsies by 43 to 60 per 100 patients while maintaining >92% melanoma detection sensitivity.
