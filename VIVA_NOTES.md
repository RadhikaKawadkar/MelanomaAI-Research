# MelanomaAI — Viva Technical Notes

Concise explanations for final-year evaluation. Metrics are frozen at production values.

## Production results (test set, n=4000)

| Metric | Value | One-line meaning |
|--------|-------|------------------|
| Accuracy | 92.8% | Overall correct classifications |
| AUC-ROC | 0.9770 | Discrimination across all thresholds |
| Sensitivity | 79.51% | Malignant cases correctly detected (recall) |
| Specificity | 97.38% | Benign cases correctly cleared |
| Precision | 91.27% | When model says malignant, how often correct |
| F1-score | 0.8498 | Harmonic mean of precision & recall |

---

## 1. EfficientNetB3 (architecture choice)

- **What:** Convolutional neural network from the EfficientNet family; B3 balances accuracy vs compute for 224×224 medical images.
- **Why here:** Strong transfer-learning backbone for dermoscopy; custom classifier head (dropout + 512→256→1 + sigmoid) replaces ImageNet 1000-class head for binary melanoma detection.
- **Training:** ImageNet-pretrained weights for phase 1, then fine-tuned top layers in phase 2 — **trained from scratch on the classification head**, not the full ImageNet decision boundary.

## 2. Focal loss (training objective)

- **Problem:** Class imbalance (many more benign than malignant nevi).
- **Formula idea:** Down-weights easy examples, up-weights hard misclassified ones: FL = −α(1−p)^γ log(p).
- **Our settings:** α ≈ 0.38, γ = 2.0 — improves minority (malignant) class learning without oversampling alone.
- **Why not plain cross-entropy:** Reduces dominance of easy benign patches so the model still learns subtle melanoma patterns.

## 3. AUC-ROC

- **Definition:** Area under the curve plotting True Positive Rate vs False Positive Rate at all thresholds.
- **Interpretation:** 0.977 = excellent separation between benign and malignant score distributions.
- **Why report it:** Threshold-independent; suitable for medical screening where operating point may change.

## 4. Sensitivity vs specificity

- **Sensitivity (recall):** TP / (TP + FN) — catches melanoma; **79.51%** means some malignancies still missed → clinical follow-up still required.
- **Specificity:** TN / (TN + FP) — **97.38%** means few benign lesions flagged as malignant → lowers unnecessary biopsies.
- **Trade-off:** Tuned via decision threshold (default 0.50) and risk stratification for borderline cases.

## 5. Grad-CAM (explainability)

- **What:** Gradient-weighted Class Activation Mapping — highlights image regions that most influenced the prediction.
- **How:** Backpropagate from output logit to last conv layer of EfficientNetB3; weight feature maps by gradient importance.
- **Clinical use:** Verify attention is on the lesion, not ruler marks, hair, or ink — supports trust and error analysis.
- **In app:** Original vs overlay shown in Workspace after analysis.

## 6. Fairness & skin-tone preprocessing

- **Issue:** Dermoscopy datasets skew toward lighter skin; models often underperform on darker skin (Fitzpatrick IV–VI).
- **Our approach (inference-time, no retrain):** Estimate **ITA** (Individual Typology Angle) from corner skin pixels in L*a*b* color space.
- **Routing:** ITA ≤ 28° → apply **CLAHE** on L channel before inference; lighter skin → standard path.
- **Result:** ~24% reduction in fairness gap in ablation (dark-skin accuracy 81.88% → 84.14% with CLAHE).

## 7. CLAHE

- **Contrast Limited Adaptive Histogram Equalization** — local contrast enhancement.
- **Why for dark skin:** Improves visibility of lesion borders and pigment network without changing labels.
- **Key point:** Applied at **inference**, so the frozen `melanoma_final.pth` benefits without retraining.

## 8. Dataset balancing

- **Source:** HAM10000 (+ project-specific balancing pipeline in `balance_dataset.py`).
- **Pipeline:** Resample / augment minority malignant class; weighted sampling during training (`WeightedRandomSampler`).
- **Goal:** Reduce bias toward predicting benign; paired with focal loss for hard examples.
- **Note:** Raw images are not in git; `data/` is local only.

## 9. Why “trained from scratch” (clarify for examiners)

- **Clarify wording:** The **classifier head** and fine-tuned top blocks were trained on our melanoma dataset from ImageNet initialization — not using a generic ImageNet “melanoma” class.
- **We did not** train EfficientNet from random initialization (that would need far more data).
- **Phases:** Phase 1 — frozen backbone, train head; Phase 2 — unfreeze top ~30% of backbone with lower learning rate.

## 10. Risk engine (clinical layer)

- **Not changing the CNN score** — combines probability, distance from threshold, skin-tone reliability, and optional age into a 0–100 **clinical risk score**.
- **Levels:** LOW / MODERATE / HIGH / CRITICAL with recommended actions.
- **Purpose:** Reduce false-positive panic when probability is only slightly above threshold.

---

## Demo script (2–3 minutes)

1. **Workspace** — Quick-load **Benign** → Run Analysis → show low probability, LOW/MODERATE risk.
2. Quick-load **Malignant** → show verdict, risk bar, ABCD, Grad-CAM on lesion.
3. Quick-load **Dark skin** → highlight **CLAHE enhanced** badge.
4. **Model Performance** — ROC, confusion matrix, v1→v2 table.
5. **Fairness** — skin-tone bars + CLAHE ablation.
6. **Session History** — export CSV if needed.

## Likely examiner questions

| Question | Short answer |
|----------|----------------|
| Why EfficientNetB3? | Best accuracy/latency trade-off for 224px dermoscopy on our GPU |
| How prevent overfitting? | Augmentation, dropout, two-phase training, early stopping |
| Is it diagnostic? | No — decision support only; requires dermatologist |
| GPU required? | No — CPU works; GPU faster for demo |
| What if Grad-CAM on background? | Flag for manual review; known limitation of weak supervision |
