# Clinical Utility & Scientific Integrity Audit

**Repository:** `https://github.com/RadhikaKawadkar/MelanomaAI-Research.git`  
**Workspace:** `C:\DS_CP`  
**Date:** October 18, 2026  
**Auditor:** AntiGravity AI Research Agent  
**Branch:** `research/clinical-utility`  

---

## 1. Executive Summary

This audit evaluated whether the existing MelanomaAI codebase provides a scientifically defensible foundation for clinical decision support. The investigation uncovered three critical methodological blockers:
1. **Severe Data Leakage (Train/Validation Contamination)**: Augmented copies of malignant images crossed the training/validation split, and HAM10000 images were not grouped by lesion or patient ID.
2. **Demographic Equity Violation in Heuristic Risk Scoring**: `compute_risk_score` in [risk_engine.py](file:///C:/DS_CP/risk_engine.py) awarded higher points to lighter skin tones, causing dark-skinned patients to receive lower risk tiers for the exact same malignancy probability.
3. **Missing Checkpoint & Dataset Artifacts**: Model weights (`models/melanoma_final.pth`) and datasets (`data/`) were gitignored and not present on disk.

---

## 2. Detailed Findings by Component

### A. Model Prediction Architecture
- **Backbone**: EfficientNet-B3 with custom 2-stage MLP classification head (`512 -> 256 -> 1 -> Sigmoid`).
- **Loss**: Focal Loss ($\gamma = 2.0, \alpha = 0.35$).
- **Output**: Uncalibrated Sigmoid probability $\in [0, 1]$.
- **Locations**: [backend/model.py](file:///C:/DS_CP/backend/model.py), [auto_train.py](file:///C:/DS_CP/auto_train.py), [evaluate_model.py](file:///C:/DS_CP/evaluate_model.py).

### B. Preprocessing & Resolution Discrepancy
- **Training Resolution**: $384 \times 384$ in [auto_train.py:39](file:///C:/DS_CP/auto_train.py#L39).
- **Inference & Previous Eval Resolution**: $224 \times 224$ in [backend/model.py:69](file:///C:/DS_CP/backend/model.py#L69) and [evaluate_model.py:30](file:///C:/DS_CP/evaluate_model.py#L30).
- **Skin Tone Routing**: Individual Typology Angle ($ITA$) calculated from image corners in LAB color space. If $ITA \le 28$ (Tan, Brown, Dark), inference-time CLAHE is applied to the L-channel.

### C. Risk Tier Calculation Flaw & Correction
- **Original Heuristic**:
  $$\text{score} = \min(100, \text{prob} \times 65 + \text{conf} + \text{tone\_pts} \times \text{prob} + \text{age\_adj})$$
  Where `tone_pts` was:
  - Very Light: 15
  - Light: 13
  - Intermediate: 10
  - Tan: 6
  - Brown: 3
  - Dark: 1
- **Disparity Inversion**: For identical probability (e.g. $P=0.60$), very light skin received $+9.0$ points while dark skin received $+0.6$ points. This lowered triage priority for darker skin tones.
- **Correction Applied**: Removed `skin_contrib = tone_pts * prob`. Probability now monotonically maps directly to the base score ($P \times 100$). An explicit `under_represented_tone` flag and demographic uncertainty notice are raised for $ITA \le 28$.

### D. Synthetic ABCD Scoring Disclosure
- In [risk_engine.py:249-264](file:///C:/DS_CP/risk_engine.py#L249-L264), ABCD dermoscopy metrics (Asymmetry, Border, Color, Diameter) were linearly computed from $P$ (`asymmetry = prob * 0.89 + 0.05`, `diameter = 5.0 + prob * 8.2`).
- **Correction Applied**: Marked explicitly with `is_synthetic_proxy = True` and documented as UI prototype aids, not true clinical lesion morphology.

### E. Data Leakage Mechanism
1. [balance_dataset.py](file:///C:/DS_CP/balance_dataset.py) augmented malignant images offline into `data/v2/balanced/malignant/aug_*`.
2. [auto_train.py](file:///C:/DS_CP/auto_train.py) applied a random permutation across all files in that folder.
3. Augmented copies of training lesions entered the validation set.
4. Images were not grouped by `lesion_id` or `patient_id` (a known characteristic of HAM10000).
5. The previously claimed 92.8% accuracy and 0.9770 AUC cannot be considered valid for prospective clinical deployment without leak-free re-evaluation.

### F. Checkpoint Selection Bug
- [evaluate_model.py:67](file:///C:/DS_CP/evaluate_model.py#L67) previously sorted candidate checkpoints by file size (`os.path.getsize(p)`).
- **Correction Applied**: Replaced with explicit priority ranking and checkpoint loadability validation.

---

## 3. Current Artifact Status

| Artifact | Expected Path | Present on Disk? | Status |
| :--- | :--- | :--- | :--- |
| Model Checkpoint | `models/melanoma_final.pth` | No | Blocked pending checkpoint recovery |
| Dataset Directory | `data/` (`raw`, `processed`) | No | Blocked pending dataset recovery |
| Output Metrics | `outputs/metrics.json` | No | Clean slate |
| Python Environment | Python 3.10 & 3.14 (CUDA 12.8) | Yes | PyTorch 2.10.0+cu128 verified working |
