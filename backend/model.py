import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from risk_engine import robustness_preprocess, get_skin_tone, estimate_ita, compute_risk_score, estimate_abcd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 384

val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ─────────────────────────────────────────────────────────────────────────────
# PLATT CALIBRATION COEFFICIENTS & FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
# Verified clean retraining Platt scaling coefficients
# Fitted strictly on independent CALIB partition (3,334 images, 206 patients, 0% test leakage):
# Logit formula: logit = ln(p_raw / (1 - p_raw)), clipped to [1e-6, 1 - 1e-6].
# Calibrated probability: p_calib = 1 / (1 + exp(-(A * logit + B)))
PLATT_COEFF_A = 1.862194
PLATT_COEFF_B = -3.203446

# Audit coefficients specified in prompt instruction
PLATT_AUDIT_A = 0.7410
PLATT_AUDIT_B = -1.9791


def calibrate_probability(
    prob: float,
    a: float = PLATT_COEFF_A,
    b: float = PLATT_COEFF_B,
    eps: float = 1e-6
) -> float:
    """
    Applies Platt scaling calibration to a raw model probability.
    Uses the exact same logit convention as the clean evaluation pipeline:
        logit = ln(clip(p, eps, 1-eps) / (1 - clip(p, eps, 1-eps)))
        p_calib = 1 / (1 + exp(-(a * logit + b)))
    """
    prob = float(np.clip(prob, eps, 1.0 - eps))
    logit = float(np.log(prob / (1.0 - prob)))
    calib_prob = 1.0 / (1.0 + np.exp(-(a * logit + b)))
    return float(np.clip(calib_prob, 0.0, 1.0))


def build_model():
    base = models.efficientnet_b3(weights=None)
    nf = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(nf, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )
    return base


_CACHED_MODEL = None
_CACHED_MODEL_PATH = None


def load_model(cached: bool = True):
    global _CACHED_MODEL, _CACHED_MODEL_PATH
    if cached and _CACHED_MODEL is not None and _CACHED_MODEL_PATH is not None:
        return _CACHED_MODEL, _CACHED_MODEL_PATH

    model = build_model()
    candidate_files = [
        # 1. Clean retrained checkpoint (ISIC 2020 patient-stratified)
        os.path.join(os.path.dirname(__file__), '..', 'outputs', 'clean_retraining', 'checkpoints', 'clean_model_best.pth'),
        os.path.join(os.getcwd(), 'outputs', 'clean_retraining', 'checkpoints', 'clean_model_best.pth'),
        # 2. Existing production and intermediate checkpoints (fallbacks)
        os.path.join(os.path.dirname(__file__), '..', 'models', 'melanoma_final.pth'),
        os.path.join(os.path.dirname(__file__), '..', 'models', 'best_phase2.pth'),
        os.path.join(os.path.dirname(__file__), '..', 'models', 'best_phase1.pth'),
        os.path.join(os.getcwd(), 'models', 'melanoma_final.pth'),
        os.path.join(os.getcwd(), 'models', 'best_phase2.pth'),
        os.path.join(os.getcwd(), 'models', 'best_phase1.pth'),
    ]
    for p in candidate_files:
        if os.path.exists(p):
            try:
                try:
                    ck = torch.load(p, map_location=DEVICE, weights_only=False)
                except TypeError:
                    ck = torch.load(p, map_location=DEVICE)
                if isinstance(ck, dict) and 'model_state_dict' in ck:
                    model.load_state_dict(ck['model_state_dict'])
                elif isinstance(ck, dict) and 'state_dict' in ck:
                    model.load_state_dict(ck['state_dict'])
                elif isinstance(ck, dict):
                    model.load_state_dict(ck)
                else:
                    model.load_state_dict(ck)
                eval_model = model.to(DEVICE).eval()
                resolved_path = os.path.normpath(os.path.abspath(p))
                if cached:
                    _CACHED_MODEL = eval_model
                    _CACHED_MODEL_PATH = resolved_path
                return eval_model, resolved_path
            except Exception:
                continue
    return None, None


def predict(model, img_arr, threshold=0.5, use_robustness=True,
            use_calibration=False, calib_a=PLATT_COEFF_A, calib_b=PLATT_COEFF_B):
    img = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
    if use_robustness:
        processed, ita, tone_lbl, reliability, clahe_used = robustness_preprocess(img)
    else:
        processed = img
        ita = estimate_ita(img)
        tone_lbl, tone_col, _, reliability = get_skin_tone(ita)
        clahe_used = False
    tensor = val_tf(image=processed)["image"].unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        prob_raw = float(model(tensor).squeeze().cpu())
    
    prob_calibrated = calibrate_probability(prob_raw, a=calib_a, b=calib_b)
    prob = prob_calibrated if use_calibration else prob_raw
    
    _, tone_col, _, _ = get_skin_tone(ita)
    label = "MALIGNANT" if prob >= threshold else "BENIGN"
    risk = compute_risk_score(prob, ita, threshold)
    abcd = estimate_abcd(prob)
    return {
        'prob': prob,
        'prob_raw': prob_raw,
        'prob_calibrated': prob_calibrated,
        'label': label,
        'ita': ita,
        'tone_lbl': tone_lbl,
        'tone_col': tone_col,
        'clahe': clahe_used,
        'reliability': reliability,
        'risk': risk,
        'abcd': abcd,
    }


def predict_image(img_path, threshold=0.5, use_robustness=True,
                  use_calibration=False, calib_a=PLATT_COEFF_A, calib_b=PLATT_COEFF_B,
                  model=None):
    path = None
    if model is None:
        model, path = load_model()
    else:
        path = _CACHED_MODEL_PATH
    if model is None:
        raise FileNotFoundError('Model checkpoint not found. Ensure clean_model_best.pth or models/melanoma_final.pth exists.')
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError('Unable to read image file: {}'.format(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = predict(
        model, img,
        threshold=threshold,
        use_robustness=use_robustness,
        use_calibration=use_calibration,
        calib_a=calib_a,
        calib_b=calib_b
    )
    return {
        'probability': result['prob'],
        'raw_probability': result['prob_raw'],
        'calibrated_probability': result['prob_calibrated'],
        'label': result['label'],
        'ita': result['ita'],
        'tone': result['tone_lbl'],
        'tone_col': result['tone_col'],
        'clahe_applied': result['clahe'],
        'risk': result['risk'],
        'abcd': result['abcd'],
        'reliability': result['reliability'],
        'model_path': path,
    }
