import os
import sys
import cv2
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from risk_engine import robustness_preprocess, get_skin_tone, estimate_ita, compute_risk_score, estimate_abcd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_tf = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


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


def load_model():
    model = build_model()
    candidate_files = [
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
                ck = torch.load(p, map_location=DEVICE)
                if isinstance(ck, dict) and 'model_state_dict' in ck:
                    model.load_state_dict(ck['model_state_dict'])
                elif isinstance(ck, dict) and 'state_dict' in ck:
                    model.load_state_dict(ck['state_dict'])
                elif isinstance(ck, dict):
                    model.load_state_dict(ck)
                else:
                    model.load_state_dict(ck)
                return model.to(DEVICE).eval(), p
            except Exception:
                continue
    return None, None


def predict(model, img_arr, threshold=0.5, use_robustness=True):
    img = cv2.resize(img_arr, (224, 224))
    if use_robustness:
        processed, ita, tone_lbl, reliability, clahe_used = robustness_preprocess(img)
    else:
        processed = img
        ita = estimate_ita(img)
        tone_lbl, tone_col, _, reliability = get_skin_tone(ita)
        clahe_used = False
    tensor = val_tf(image=processed)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = float(model(tensor).squeeze().cpu())
    _, tone_col, _, _ = get_skin_tone(ita)
    label = "MALIGNANT" if prob >= threshold else "BENIGN"
    risk = compute_risk_score(prob, ita, threshold)
    abcd = estimate_abcd(prob)
    return {
        'prob': prob,
        'label': label,
        'ita': ita,
        'tone_lbl': tone_lbl,
        'tone_col': tone_col,
        'clahe': clahe_used,
        'reliability': reliability,
        'risk': risk,
        'abcd': abcd,
    }


def predict_image(img_path, threshold=0.5, use_robustness=True):
    model, path = load_model()
    if model is None:
        raise FileNotFoundError('Model checkpoint not found. Ensure models/melanoma_final.pth exists.')
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError('Unable to read image file: {}'.format(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = predict(model, img, threshold=threshold, use_robustness=use_robustness)
    return {
        'probability': result['prob'],
        'label': result['label'],
        'ita': result['ita'],
        'tone': result['tone_lbl'],
        'tone_col': result['tone_col'],
        'clahe_applied': result['clahe'],
        'risk': result['risk'],
        'abcd': result['abcd'],
        'reliability': result['reliability'],
    }
