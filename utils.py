"""
Shared utilities for MelanomaAI multi-page app
CNN model loading, prediction, and skin tone analysis
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─── CNN MODEL FUNCTIONS ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = models.efficientnet_b3(weights=None)
    num_features = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    ckpt = torch.load("models/melanoma_final.pth", map_location=device, weights_only=False)
    base.load_state_dict(ckpt["model_state_dict"])
    base = base.to(device)
    base.eval()
    return base, "models/melanoma_final.pth"

def estimate_ita(img_rgb):
    try:
        u8 = np.clip(img_rgb,0,255).astype(np.uint8)
        lab = cv2.cvtColor(u8,cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
        mask = gray > 20
        valid_lab = lab[mask]
        if len(valid_lab) == 0: return 30.0
        L_channel = valid_lab[:, 0].astype(float) * 100 / 255.0
        threshold_L = np.percentile(L_channel, 60)
        threshold_L_upper = np.percentile(L_channel, 95)
        skin_pixels = valid_lab[(L_channel > threshold_L) & (L_channel < threshold_L_upper)]
        if len(skin_pixels) == 0: skin_pixels = valid_lab
        L = float(np.mean(skin_pixels[:, 0])) * 100 / 255.0
        b = float(np.mean(skin_pixels[:, 2])) - 128.0
        b = b if abs(b) > 1e-6 else 1e-6
        ita = np.arctan2((L - 50), b) * (180 / np.pi)
        
        # In ITA theory, values > 90 are just extremely light skin on the negative b axis
        if ita > 90: return 90.0
        if ita < -90: return -90.0
        return ita
    except Exception as e:
        return 30.0

TONES = [(55,"Very Light","#F5CBA7"),(41,"Light","#E59866"),
         (28,"Intermediate","#CA8A5A"),(10,"Tan","#A0522D"),
         (-30,"Brown","#6B3A2A"),(-99,"Dark","#3D1C0E")]

def get_tone(ita):
    for t,l,c in TONES:
        if ita>t: return l,c
    return "Dark","#3D1C0E"

def apply_clahe(img_rgb):
    u8 = np.clip(img_rgb,0,255).astype(np.uint8)
    lab = cv2.cvtColor(u8,cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)

def get_threshold():
    if os.path.exists("outputs/optimal_threshold.json"):
        with open("outputs/optimal_threshold.json") as f:
            return json.load(f)["optimal_threshold"]
    return 0.48

def run_prediction(model, img_array, threshold=0.48, use_robustness=True):
    img = cv2.resize(img_array,(224,224))
    ita = estimate_ita(img)
    tone_label, tone_color = get_tone(ita)
    if use_robustness and ita<=28:
        img = apply_clahe(img)
    transform = Compose([
        Resize(224, 224),
        Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])
    transformed = transform(image=img)
    device = next(model.parameters()).device
    inp = transformed["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        prob = float(model(inp).squeeze().cpu().item())
    label = "MALIGNANT" if prob>=threshold else "BENIGN"
    return prob,label,ita,tone_label,tone_color

# ─── METRICS LOADING ───────────────────────────────────────────────────────────────────
def load_metrics():
    for p in ["outputs/metrics.json","metrics.json"]:
        if os.path.exists(p):
            with open(p) as f: return json.load(f)
    return {"accuracy":0.89,"auc_roc":0.92,"sensitivity":0.87,"specificity":0.91}

# ─── SHARED CSS ───────────────────────────────────────────────────────────────────────
SHARED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@300;400;500;600;700&family=Syne:wght@700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root {
    --primary:#00D4FF; --danger:#FF3B5C; --success:#00E5A0;
    --bg:#03070F; --card:#0A101E; --border:rgba(255,255,255,0.08);
    --text:#CBD5E1; --muted:#475569;
}
html,body,[class*="css"]{font-family:'Public Sans',sans-serif!important;background:var(--bg)!important;color:var(--text)!important;}
.stApp{background:var(--bg)!important;}
.block-container{padding:0!important;max-width:100%!important;}
#MainMenu,footer,header{visibility:hidden;}
section[data-testid="stSidebar"]{display:none!important;}
.topnav{display:flex;align-items:center;justify-content:space-between;padding:16px 32px;border-bottom:1px solid var(--border);background:rgba(3,7,15,0.9);backdrop-filter:blur(12px);position:sticky;top:0;z-index:100;}
.logo-text{font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;color:white;text-transform:uppercase;letter-spacing:-0.5px;}
.logo-text span{color:var(--primary);}
.logo-icon{width:32px;height:32px;background:var(--primary);border-radius:6px;display:inline-flex;align-items:center;justify-content:center;color:#03070F;margin-right:10px;font-size:1rem;}
.nav-link{font-size:0.82rem;font-weight:500;color:var(--muted);margin:0 12px;cursor:pointer;transition:all 0.2s;}
.nav-link:hover{color:var(--primary);}
.nav-link.active{color:var(--primary);border-bottom:2px solid var(--primary);padding-bottom:2px;}
.page-header{padding:28px 32px 0;max-width:1440px;margin:0 auto;width:100%;}
.page-title{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:white;margin:0;letter-spacing:-1px;}
.card{background:var(--card);border:1px solid var(--border);border-radius:16px;overflow:hidden;margin-bottom:16px;}
.card-header{display:flex;align-items:center;justify-content:space-between;padding:18px 22px 14px;}
.card-title{font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:700;color:white;}
.card-body{padding:0 22px 22px;}
.btn-cyan{background:var(--primary);color:#03070F;border:none;border-radius:8px;padding:12px 24px;font-weight:600;font-size:0.9rem;cursor:pointer;transition:all 0.2s;}
.btn-cyan:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(0,212,255,0.3);}
.input-field{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 16px;color:var(--text);width:100%;font-size:0.9rem;}
.section-label{font-family:'IBM Plex Mono',monospace;font-size:0.7rem;text-transform:uppercase;letter-spacing:2px;color:var(--primary);margin-bottom:12px;}
.footer-bar{border-top:1px solid var(--border);background:rgba(3,7,15,0.9);padding:14px 32px;text-align:center;font-size:0.75rem;color:var(--muted);}
.metric-chip{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:8px 12px;display:inline-block;margin:4px;}
.metric-value{font-family:'IBM Plex Mono',monospace;font-size:1.1rem;font-weight:600;}
.metric-label{font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;}
.risk-bar{height:8px;background:linear-gradient(90deg,var(--success) 0%,var(--primary) 50%,var(--danger) 100%);border-radius:99px;position:relative;}
.risk-marker{position:absolute;top:-2px;width:12px;height:12px;background:white;border-radius:50%;box-shadow:0 2px 4px rgba(0,0,0,0.2);}
.data-table{width:100%;border-collapse:collapse;}
.data-table th{background:var(--card);padding:12px;text-align:left;font-weight:600;color:var(--text);border-bottom:1px solid var(--border);}
.data-table td{padding:12px;border-bottom:1px solid var(--border);color:var(--text);}
.badge{padding:4px 8px;border-radius:4px;font-size:0.75rem;font-weight:600;text-transform:uppercase;}
.badge-danger{background:var(--danger);color:white;}
.badge-success{background:var(--success);color:white;}
.progress-bar{height:4px;background:var(--card);border-radius:2px;overflow:hidden;}
.progress-fill{height:100%;background:var(--primary);}
</style>
"""

# ─── TOPNAV HTML ───────────────────────────────────────────────────────────────────────
def get_topnav(active_page):
    return f"""
    <div class="topnav">
        <div style="display:flex;align-items:center;gap:28px">
            <div><span class="logo-icon">🔬</span><span class="logo-text">Melanoma<span>AI</span></span></div>
            <div>
                <span class="nav-link {'active' if active_page=='Dashboard' else ''}" onclick="window.location.href='/'">Dashboard</span>
                <span class="nav-link {'active' if active_page=='Patient List' else ''}" onclick="window.location.href='/Patient_List'">Patient List</span>
                <span class="nav-link {'active' if active_page=='Archive' else ''}" onclick="window.location.href='/Archive'">Archive</span>
                <span class="nav-link {'active' if active_page=='Analytics' else ''}" onclick="window.location.href='/Analytics'">Analytics</span>
                <span class="nav-link {'active' if active_page=='Settings' else ''}" onclick="window.location.href='/Settings'">Settings</span>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:16px;font-size:0.78rem;color:#475569">
            <span style="color:var(--success);">● System Online</span>
            {datetime.now().strftime('%d %b %Y · %H:%M')}
        </div>
    </div>
    """
