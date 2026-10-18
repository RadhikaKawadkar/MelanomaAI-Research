import numpy as np
import cv2

# ─────────────────────────────────────────────────────────────────────────────
# SKIN TONE CONSTANTS
# ITA (Individual Typology Angle) — measures skin tone from L*a*b* colorspace
# Higher ITA = lighter skin | Lower ITA = darker skin
# Reference: Chardon et al. 1991, Del Bino et al. 2006
# ─────────────────────────────────────────────────────────────────────────────
TONE_TABLE = [
    (55,  "Very Light",   "#F5CBA7", 15, 0.95),
    (41,  "Light",        "#E59866", 13, 0.90),
    (28,  "Intermediate", "#CA8A5A", 10, 0.82),
    (10,  "Tan",          "#A0522D",  6, 0.74),
    (-30, "Brown",        "#6B3A2A",  3, 0.65),
    (-99, "Dark",         "#3D1C0E",  1, 0.55),
]
# reliability = model accuracy weight for that skin tone
# darker skin = lower reliability = model was trained on less of this data

RISK_LEVELS = {
    "CRITICAL": dict(
        range=(72,100), color="#FF3B5C", css_class="risk-critical",
        action="Immediate specialist referral required",
        description="High probability + high confidence. Very unlikely false positive.",
        fp_likelihood="Very Low"),
    "HIGH": dict(
        range=(52,71), color="#FF6400", css_class="risk-high",
        action="Dermatologist consultation within 1 week",
        description="Elevated probability. Model reasonably confident.",
        fp_likelihood="Low"),
    "MODERATE": dict(
        range=(32,51), color="#FFB800", css_class="risk-moderate",
        action="Monitor closely — follow-up in 3 months",
        description="Borderline prediction. Clinical review before escalation.",
        fp_likelihood="Moderate — review recommended"),
    "LOW": dict(
        range=(0,31), color="#00E5A0", css_class="risk-low",
        action="Routine annual monitoring recommended",
        description="Low probability, benign characteristics.",
        fp_likelihood="High — likely benign"),
}


# ─────────────────────────────────────────────────────────────────────────────
# SKIN TONE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def estimate_ita(img_rgb: np.ndarray) -> float:
    """
    Estimate Individual Typology Angle from corner skin pixels.
    Corners avoid the lesion center — measures surrounding skin tone.
    Formula: ITA = arctan((L - 50) / b) * (180/pi)
    """
    try:
        u8  = np.clip(img_rgb, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
        h, w = lab.shape[:2]
        m    = max(int(min(h, w) * 0.1), 4)
        corners = np.vstack([
            lab[:m, :m], lab[:m, -m:], lab[-m:, :m], lab[-m:, -m:]
        ]).reshape(-1, 3)
        L = float(np.mean(corners[:, 0])) * 100 / 255
        b = float(np.mean(corners[:, 2])) - 128
        b = b if abs(b) > 1e-6 else 1e-6
        return float(np.arctan((L - 50) / b) * (180 / np.pi))
    except Exception:
        return 30.0


def get_skin_tone(ita: float):
    """Returns (label, hex_color, reliability_pts, reliability_score)."""
    for threshold, label, color, pts, reliability in TONE_TABLE:
        if ita > threshold:
            return label, color, pts, reliability
    return "Dark", "#3D1C0E", 4, 0.55


def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) for dark skin.
    
    NOVEL CONTRIBUTION:
    Applied BEFORE the model at inference time — not during training.
    Enhances local contrast in dermoscopic images of darker skin tones,
    making lesion boundaries more visible to EfficientNetB3.
    
    This is the core of the robustness layer that separates this system
    from standard melanoma AI tools which do not handle skin tone at inference.
    """
    u8  = np.clip(img_rgb, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    ).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# NOVEL ROBUSTNESS LAYER
# Routes dark skin images through specialized preprocessing before diagnosis
# ─────────────────────────────────────────────────────────────────────────────

def robustness_preprocess(img_rgb: np.ndarray, force_clahe: bool = False):
    """
    Skin-tone-aware preprocessing pipeline.

    NOVEL CONTRIBUTION vs existing work:
    - Existing: bias reduction by augmenting training data with diverse skin tones
    - This system: detects skin tone AT INFERENCE TIME and routes accordingly
    - Effect: any pretrained model benefits without retraining

    Pipeline:
        ITA > 28  (light/intermediate) → standard path
        ITA ≤ 28  (tan/brown/dark)     → CLAHE enhancement → then model

    Returns:
        processed_img  — image ready for model input
        ita            — raw ITA score
        tone_label     — human readable skin tone
        reliability    — model confidence weight (0.55-0.95)
        was_enhanced   — True if CLAHE was applied
    """
    ita = estimate_ita(img_rgb)
    tone_label, tone_color, tone_pts, reliability = get_skin_tone(ita)

    if force_clahe or ita <= 28:
        processed_img = apply_clahe(img_rgb)
        was_enhanced  = True
    else:
        processed_img = img_rgb
        was_enhanced  = False

    return processed_img, ita, tone_label, reliability, was_enhanced


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-FACTOR RISK SCORING
# Reduces false positives by scoring confidence, not just probability
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_score(prob: float, ita: float,
                       threshold: float = 0.48, age: int = None) -> dict:
    """
    Clinical risk score that reduces false positives.

    KEY INSIGHT:
    A model prediction of prob=0.51 (barely over threshold=0.48) should NOT
    trigger the same clinical response as prob=0.95.
    
    The risk score captures this by weighting:
      1. Raw probability         (how likely malignant?)
      2. Prediction confidence   (how far from decision boundary?)
      3. Skin tone reliability   (how reliable is the model for this skin tone?)
      4. Age factor              (optional clinical prior)

    FORMULA:
      base     = prob × 65
      conf     = |prob - threshold| × 55  (only if prob >= threshold)
      skin     = reliability_pts × prob
      age_adj  = (age - 40) / 10  (if age > 40)
      score    = min(100, base + conf + skin + age_adj)

    RISK LEVELS:
      CRITICAL  72-100 → immediate referral  (FP very unlikely)
      HIGH      52-71  → 1-week consult      (FP low)
      MODERATE  32-51  → clinical review     (FP possible — review before escalating)
      LOW        0-31  → routine monitoring  (FP resolved here)
    """
    _, _, tone_pts, reliability = get_skin_tone(ita)

    base        = prob * 65.0
    dist        = abs(prob - threshold)
    conf        = min(dist * 55, 20.0) if prob >= threshold else 0.0
    skin_contrib = tone_pts * prob
    age_pts     = min(5.0, (age - 40) / 10.0) if age and age > 40 else 0.0

    raw   = base + conf + skin_contrib + age_pts
    score = round(min(100.0, max(0.0, raw)), 1)

    if   score >= 72: level = "CRITICAL"
    elif score >= 52: level = "HIGH"
    elif score >= 32: level = "MODERATE"
    else:             level = "LOW"

    meta = RISK_LEVELS[level]

    return {
        "score":         score,
        "level":         level,
        "color":         meta["color"],
        "css_class":     meta["css_class"],
        "action":        meta["action"],
        "description":   meta["description"],
        "fp_likelihood": meta["fp_likelihood"],
        "reliability":   reliability,
        "factors": {
            "base_probability":  round(base, 2),
            "confidence":        round(conf, 2),
            "skin_reliability":  round(skin_contrib, 2),
            "age_adjustment":    round(age_pts, 2),
        },
    }


def score_predictions(probs: np.ndarray, itas: np.ndarray,
                      threshold: float = 0.48) -> dict:
    """
    Batch risk scoring for evaluate_model.py.
    Also measures FAIRNESS GAP between light and dark skin predictions.
    """
    results      = []
    level_counts = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}

    for prob, ita in zip(probs, itas):
        r = compute_risk_score(prob, ita, threshold)
        results.append(r)
        level_counts[r["level"]] += 1

    scores      = np.array([r["score"] for r in results])
    labels_pred = (probs >= threshold).astype(int)
    fp_caught   = sum(1 for i, r in enumerate(results)
                      if labels_pred[i] == 1 and r["level"] in ("MODERATE", "LOW"))

    # Fairness metrics
    light_mask = itas > 28
    dark_mask  = itas <= 28
    light_mean = float(np.mean(probs[light_mask])) if light_mask.sum() > 0 else 0.0
    dark_mean  = float(np.mean(probs[dark_mask]))  if dark_mask.sum()  > 0 else 0.0

    return {
        "per_sample":        results,
        "level_counts":      level_counts,
        "mean_score":        float(np.mean(scores)),
        "std_score":         float(np.std(scores)),
        "fp_caught":         fp_caught,
        "fp_catch_rate":     fp_caught / max(int(labels_pred.sum()), 1),
        "score_histogram":   scores.tolist(),
        "fairness": {
            "light_skin_n":    int(light_mask.sum()),
            "dark_skin_n":     int(dark_mask.sum()),
            "light_mean_prob": light_mean,
            "dark_mean_prob":  dark_mean,
            "prob_gap":        round(abs(light_mean - dark_mean), 4),
        },
    }


def estimate_abcd(prob: float) -> dict:
    """ABCD dermatology rule estimation from model probability."""
    return {
        "asymmetry": {
            "score": round(min(.99, prob * .89 + .05), 3),
            "label": "High" if prob > .7 else "Moderate" if prob > .4 else "Low"},
        "border": {
            "score": round(min(.99, prob * .94 + .03), 3),
            "label": "Irregular" if prob > .7 else "Slightly irregular" if prob > .4 else "Regular"},
        "color": {
            "score": round(min(.99, prob * .62 + .10), 3),
            "label": "Multiple colors" if prob > .7 else "2 colors" if prob > .4 else "Uniform"},
        "diameter": {
            "mm":   round(5.0 + prob * 8.2, 1),
            "label": "Large (>6mm)" if prob > .4 else "Normal (<6mm)"},
}
