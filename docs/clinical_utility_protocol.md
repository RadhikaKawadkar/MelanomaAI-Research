# Clinical Utility & Decision Analysis Protocol

**Project:** MelanomaAI Clinical Decision Utility Evaluation  
**Version:** 1.0  
**Date:** October 18, 2026  

---

## 1. Clinical Context & Decision-Making Scenario

- **Target Clinical Scenario**: Outpatient dermatology referral and primary-care dermoscopic screening.
- **Target Outcome ($Y=1$)**: Histologically confirmed melanoma requiring excision.
- **Primary Clinical Decision**: **Immediate Specialist Referral / Biopsy** vs. **Routine Monitoring / Clinical Reassurance**.
- **Clinical Cost Asymmetry**: A False Negative (missed melanoma) carries severe morbidity/mortality risk, whereas a False Positive (benign biopsy) carries minimal surgical and financial costs. Therefore, clinicians operate at low threshold probabilities ($p_t$).

---

## 2. Decision Threshold Range Justification & Critical Review

### Methodological Justification
The decision threshold probability ($p_t$) represents the specific clinical cut-point where the expected utility of intervention (referral or biopsy) equals the expected utility of observation:
$$p_t = \frac{\text{Harm}_{\text{FP}}}{\text{Benefit}_{\text{TP}} + \text{Harm}_{\text{FP}}} = \frac{1}{1 + \frac{\text{Benefit}_{\text{TP}}}{\text{Harm}_{\text{FP}}}}$$

In melanoma detection, the harm of a False Negative (delayed diagnosis of invasive melanoma with risk of metastasis and mortality) far outweighs the harm of a False Positive (minor anxiety and a superficial diagnostic punch or shave biopsy). As a consequence, clinicians operate at low threshold probabilities ($p_t \ll 0.50$).

### Stratification by Clinical Setting
A single static range of 5%–35% is methodologically inadequate without distinguishing the clinical setting and user persona:

1. **Setting A — Primary Care Screening & Triage (PCP to Specialist Referral)**:
   - **User**: General practitioner / primary care physician.
   - **Clinical Decision**: Routine reassurance vs. Urgent dermatology referral.
   - **Cost/Harm Ratio**: Missing a melanoma is catastrophic; a referral carries low physical harm.
   - **Justified Threshold Range**: **$p_t \in [0.02, 0.08]$** (corresponding to willingness to refer 12 to 50 benign lesions per detected melanoma).
2. **Setting B — Secondary Care Dermatology (Excisional Biopsy Decision)**:
   - **User**: Board-certified dermatologist with dermoscopy expertise.
   - **Clinical Decision**: Clinical / digital monitoring vs. Surgical excision/biopsy.
   - **Cost/Harm Ratio**: Dermatologist NNB benchmarks typically range from 5:1 to 15:1.
   - **Justified Threshold Range**: **$p_t \in [0.08, 0.20]$** (corresponding to willingness to biopsy 5 to 12 benign lesions per melanoma).
3. **Setting C — Cosmetic / High-Prevalence Sensitive Zone**:
   - **User**: Specialist evaluating lesions in cosmetically sensitive areas (e.g., face) or multiple dysplastic nevi.
   - **Justified Threshold Range**: **$p_t \in [0.20, 0.35]$** (higher threshold combined with short-term sequential digital dermoscopy monitoring).

### Evidence Required to Finalize Operational Thresholds
Before deploying or claiming clinical utility, the following empirical evidence must be established:
1. **Intended Deployment Tier**: Explicitly declaring whether MelanomaAI operates as a primary care triage filter or a specialist biopsy aid.
2. **Formal Harm Utility Assessment**: Clinical elicitation of the local harm ratio ($\text{Harm}_{\text{FP}} / \text{Benefit}_{\text{TP}}$) accounting for biopsy morbidity, scarring, and healthcare costs.
3. **Multi-Threshold Sensitivity Analysis**: Evaluating Net Benefit across all three sub-ranges ($[0.02, 0.08]$, $[0.08, 0.20]$, $[0.20, 0.35]$) rather than relying on an unverified single bracket.

---

## 3. Probability Calibration Protocol

Raw neural network outputs (especially those trained with Focal Loss or class-weighted cross-entropy) cannot be interpreted as true probabilities without calibration:

1. **Metrics**:
   - **Brier Score**: $\frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$.
   - **Expected Calibration Error (ECE)**: Weighted average difference between bin accuracy and confidence across 10 equal-frequency or equal-width bins.
2. **Methods**:
   - **Platt Scaling**: Logistic regression fitted on the log-odds of a dedicated, held-out calibration split.
   - **Isotonic Regression**: Non-parametric isotonic mapping on the calibration split.
3. **Partitioning**: Calibration parameters must NEVER be fitted on the test partition.

---

## 4. Decision Curve Analysis (DCA) Protocol

DCA quantifies the clinical value of the model by calculating the **Net Benefit** ($NB$) across threshold probabilities:

$$\text{Net Benefit}(p_t) = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \left(\frac{p_t}{1 - p_t}\right)$$

### Benchmark Strategies:
1. **Refer All**:
   $$\text{NB}_{\text{all}} = \text{Prevalence} - (1 - \text{Prevalence}) \times \left(\frac{p_t}{1 - p_t}\right)$$
2. **Refer None**:
   $$\text{NB}_{\text{none}} = 0$$

### Prevalence Adjustment:
Public benchmark datasets (e.g. HAM10000) have an artificial case-control sampling prevalence (~20%–50%). In primary screening, true melanoma prevalence is typically 1% to 5%. Calculating DCA on an unadjusted enriched cohort exaggerates net benefit.

When evaluating on enriched datasets, Net Benefit must be adjusted using Bayes' theorem:
$$\text{TP}_{\text{adj}} = \text{Sensitivity}(p_t) \times P_{\text{target}}$$
$$\text{FP}_{\text{adj}} = (1 - \text{Specificity}(p_t)) \times (1 - P_{\text{target}})$$
$$\text{NB}_{\text{adj}}(p_t) = \text{TP}_{\text{adj}} - \text{FP}_{\text{adj}} \times \left(\frac{p_t}{1 - p_t}\right)$$

---

## 5. Statistical Rigor & Uncertainty

- All primary performance metrics (AUC, Sensitivity, Specificity, Brier Score, ECE, Net Benefit) must be reported with **95% Bootstrap Confidence Intervals** generated from 1,000 percentile resamples.
- Test sets must strictly enforce **zero lesion overlap** and **zero synthetic image inclusion**.
