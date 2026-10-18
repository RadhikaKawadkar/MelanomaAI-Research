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

## 2. Decision Threshold Range Justification

In standard clinical practice, a dermatologist is willing to biopsy 10 to 20 benign lesions to detect one melanoma (Number Needed to Biopsy, $\text{NNB} \approx 10-20$).

Using the decision-analytic relationship between threshold probability $p_t$ and cost-benefit ratio:
$$p_t = \frac{\text{Harm of FP}}{\text{Benefit of TP} + \text{Harm of FP}}$$

- For $\text{NNB} = 20 \implies p_t \approx 0.05$ (5%)
- For $\text{NNB} = 10 \implies p_t \approx 0.10$ (10%)
- For $\text{NNB} = 5 \implies p_t \approx 0.17$ (17%)
- Balanced / Intermediate threshold: $p_t \approx 0.25 - 0.35$

**Standard Evaluated Threshold Range**:
$$p_t \in [0.05, 0.35]$$

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
