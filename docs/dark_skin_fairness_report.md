# MelanomaAI — Full-Cohort Dark-Skin Fairness & CLAHE Ablation Report

**Date:** October 18, 2026  
**Evaluation Scope:** Complete ISIC 2020 Holdout Test Cohort ($N = 6,458$ images across 411 patients)  
**Evaluated Checkpoint:** `clean_model_best.pth`  
**Execution Script:** `C:\DS_CP\run_dark_skin_fairness_evaluation.py`  
**Artifact Directory:** `C:\DS_CP\outputs\dark_skin_fairness\`  

---

## 1. Executive Summary & Core Scientific Findings

A comprehensive, full-cohort demographic fairness evaluation was executed across all 6,458 held-out test images of the ISIC 2020 dataset, eliminating the sampling noise of the earlier 500-sample preliminary check.

### Key Conclusions:
1. **The Fairness Gap is NOT Solved:**
   * Full-cohort evaluation reveals a statistically significant **$8.03\%$ AUROC disparity** favoring lighter skin:
     * **Light Skin ($\text{ITA} > 28$):** $\text{AUROC} = \mathbf{0.9013}$ ($95\%\text{ CI: } 0.8667 - 0.9322$)
     * **Dark Skin ($\text{ITA} \le 28$):** $\text{AUROC} = \mathbf{0.8210}$ ($95\%\text{ CI: } 0.7712 - 0.8638$)
   * The earlier report claiming Dark Skin AUROC of $0.9238$ was an artifact of extreme small-sample noise ($N=157$, only 6 melanomas). Full evaluation on 2,120 dark-skin images (58 melanomas) demonstrates lower baseline discrimination.
2. **CLAHE Does NOT Improve Feature Discrimination:**
   * A head-to-head ablation on all 2,120 dark-skin images demonstrates that CLAHE provides **zero AUROC improvement**:
     * $\text{AUROC with CLAHE} = \mathbf{0.8210}$
     * $\text{AUROC without CLAHE} = \mathbf{0.8226}$
     * $\Delta\text{AUROC} = \mathbf{-0.0016}$
   * The long-standing architectural claim that *"CLAHE is the core of the robustness layer that separates this system from standard melanoma AI tools"* is **empirically disproven** on the clean model. CLAHE merely shifts predictions toward higher probabilities (increasing sensitivity at the expense of specificity), identical to a threshold adjustment.
3. **Progress Over Legacy Model:**
   * While a disparity remains ($0.9013$ vs $0.8210$), the clean-trained model achieves substantially higher dark-skin AUROC than the legacy model ($0.7309$ on external HAM10000 data).
   * At the exploratory calibrated screening threshold ($p_t = 0.02$), dark-skin sensitivity reaches **$98.28\%$** ($57/58$ melanomas caught, $1.72\%$ FNR), though with a reduced specificity of **$45.88\%$** ($1,116$ false positives).

---

## 2. Cohort Demographics & Stratification

Individual Typology Angle (ITA) was calculated across all 6,458 holdout test images. Following standard dermoscopic dermatological literature (Chardon et al., Del Bino et al.), skin tone was partitioned at $\text{ITA} = 28^\circ$:
* **Light Skin ($\text{ITA} > 28^\circ$):** Very Light, Light, and Intermediate tones.
* **Dark Skin ($\text{ITA} \le 28^\circ$):** Tan, Brown, and Dark tones.

| Metric | Full Test Cohort | Light Skin ($\text{ITA} > 28$) | Dark Skin ($\text{ITA} \le 28$) |
| :--- | :---: | :---: | :---: |
| **Total Images** | 6,458 | 4,338 (67.17%) | 2,120 (32.83%) |
| **Unique Patients** | 411 | 381 | 364 |
| **Melanoma Cases** | 119 | 61 (51.26%) | 58 (48.74%) |
| **Benign Cases** | 6,339 | 4,277 | 2,062 |
| **Prevalence** | 1.84% | 1.41% | 2.74% |

*(Note: Multi-lesion patients can contribute lesions across different anatomical regions with differing local skin tones, explaining why sum of subgroup patients exceeds total unique cohort patients).*

---

## 3. Subgroup Performance: Standard Pipeline (WITH CLAHE)

All operating points were evaluated at pre-specified *a priori* thresholds:
* Raw default threshold: $\tau = 0.50$
* Raw sensitivity threshold: $\tau = 0.40$
* Calibrated exploratory screening threshold: $p_t = 0.02$
* Calibrated triage threshold: $p_t = 0.05$

### 3.1 Operating Performance Table

| Subgroup | AUROC [95% CI] | Scale & Threshold | TP | FP | TN | FN | Sensitivity | Specificity | FNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Light Skin** | **0.9013** | Raw $\tau = 0.50$ | 37 | 352 | 3,925 | 24 | 60.66% | 91.77% | 39.34% |
| ($N = 4,338$) | $[0.8667, 0.9322]$ | Raw $\tau = 0.40$ | 53 | 929 | 3,348 | 8 | 86.89% | 78.28% | 13.11% |
| | | Platt $p_t = 0.02$ | 53 | 872 | 3,405 | 8 | 86.89% | 79.61% | 13.11% |
| | | Platt $p_t = 0.05$ | 31 | 229 | 4,048 | 30 | 50.82% | 94.65% | 49.18% |
| **Dark Skin** | **0.8210** | Raw $\tau = 0.50$ | 40 | 524 | 1,538 | 18 | 68.97% | 74.59% | 31.03% |
| ($N = 2,120$) | $[0.7712, 0.8638]$ | Raw $\tau = 0.40$ | 57 | 1,175 | 887 | 1 | 98.28% | 43.02% | 1.72% |
| | | Platt $p_t = 0.02$ | 57 | 1,116 | 946 | 1 | 98.28% | 45.88% | 1.72% |
| | | Platt $p_t = 0.05$ | 33 | 336 | 1,726 | 25 | 56.90% | 83.71% | 43.10% |

### 3.2 Patient-Clustered Bootstrap Analysis ($\tau = 0.50$)
* **Light Skin:**
  * Sensitivity: Mean = $60.48\%$, $95\%\text{ CI: } [47.62\%, 73.69\%]$
  * Specificity: Mean = $91.76\%$, $95\%\text{ CI: } [90.04\%, 93.29\%]$
  * FNR: Mean = $39.52\%$, $95\%\text{ CI: } [26.31\%, 52.38\%]$
* **Dark Skin:**
  * Sensitivity: Mean = $68.62\%$, $95\%\text{ CI: } [55.55\%, 79.63\%]$
  * Specificity: Mean = $74.50\%$, $95\%\text{ CI: } [70.68\%, 77.75\%]$
  * FNR: Mean = $31.38\%$, $95\%\text{ CI: } [20.37\%, 44.45\%]$

---

## 4. CLAHE Ablation Study: Head-to-Head Comparison

To determine whether CLAHE provides an authentic discriminative benefit, inference was rerun on the exact same checkpoint and images with CLAHE disabled (`no_clahe=True`).

### 4.1 Paired Performance on Dark-Skin Images ($N = 2,120$, 58 Melanomas)

| Metric | With CLAHE (Standard) | Without CLAHE (Ablation) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **AUROC** | **0.8210** | **0.8226** | **-0.0016** |
| **95% Bootstrap CI** | $[0.7712, 0.8638]$ | $[0.7694, 0.8685]$ | Complete overlap |
| **Raw $\tau = 0.50$ Sensitivity** | 68.97% (40/58) | 60.34% (35/58) | +8.63% |
| **Raw $\tau = 0.50$ Specificity** | 74.59% (1,538/2,062) | 84.24% (1,737/2,062) | **-9.65%** |
| **Platt $p_t = 0.02$ Sensitivity** | 98.28% (57/58) | 91.38% (53/58) | +6.90% |
| **Platt $p_t = 0.02$ Specificity** | 45.88% (946/2,062) | 63.09% (1,301/2,062) | **-17.21%** |
| **Mean Probability Shift** | $+0.0469$ higher probability with CLAHE across dark skin |

### 4.2 Mechanistic Interpretation
1. **Zero Discriminative Gain:** Because AUROC is identical ($-0.0016$), CLAHE does not make melanoma lesions more distinguishable from benign lesions on dark skin.
2. **Probability Inversion / Threshold Artifact:** Applying CLAHE uniformly increases output logits by $\sim 0.047$. This produces higher sensitivity at raw $\tau = 0.50$ solely by trading off specificity (causing 199 additional false positives at $\tau=0.50$ and 355 additional false positives at $p_t=0.02$).
3. **Conclusion:** CLAHE does not act as a domain-robustness enhancer. Its effect is functionally identical to lowering the operating threshold.

---

## 5. Summary of Evidence, Missing Data & Remaining Limitations

1. **Unsupported Claims Refuted:**
   * Claim: *"CLAHE robustness layer eliminates bias at inference time."* — **REFUTED.** AUROC is unchanged with or without CLAHE ($0.8210$ vs $0.8226$).
   * Claim: *"Dark skin achieves higher AUC than light skin ($0.9238$ vs $0.8704$)."* — **REFUTED.** That finding was an artifact of sampling only 500 images. On the full cohort, Light Skin is $0.9013$ and Dark Skin is $0.8210$.
2. **Remaining Research Gaps:**
   * **Underlying Representation Bias:** Even with clean training, EfficientNet-B3 achieves $8.03\%$ lower AUROC on darker skin tones ($0.8210$ vs $0.9013$). This gap reflects remaining representation disparity in public web/clinical dermoscopy datasets.
   * **Specificity Penalty:** While high sensitivity ($98.28\%$) can be achieved on dark skin at $p_t=0.02$, it requires accepting a referral rate of $55.33\%$ (specificity $45.88\%$).
   * **Prospective Clinical Evidence:** All findings remain retrospective; clinical deployment requires prospective evaluation across diverse clinical sites.
