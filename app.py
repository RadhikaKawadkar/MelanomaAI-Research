"""
MelanomaAI v2 — Production Streamlit Application (UI)

This file provides the Streamlit frontend only.
Backend + model remain unchanged and are imported from:
    backend/model.py, backend/gradcam.py, backend/metrics.py, risk_engine.py

Run:
    streamlit run app.py --server.port 8502
"""

# flake8: noqa

from __future__ import annotations

import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import streamlit as st
import torch
from PIL import Image

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── MUST BE FIRST ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MelanomaAI | Clinical Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Stitch-aligned, #03070F base, #00D4FF cyan
# ══════════════════════════════════════════════════════════════════════════════
CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=IBM+Plex+Mono:wght@400;500;600&family=Public+Sans:wght@300;400;500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@400,0&display=swap');

:root{
  --bg:        #03070F;
  --bg2:       #0A101E;
  --bg3:       #0D1526;
  --cyan:      #00D4FF;
  --cyan-dim:  rgba(0,212,255,0.12);
  --cyan-glow: 0 0 24px rgba(0,212,255,0.25);
  --red:       #FF3B5C;
  --red-dim:   rgba(255,59,92,0.12);
  --green:     #00E5A0;
  --green-dim: rgba(0,229,160,0.12);
  --orange:    #FF8C42;
  --yellow:    #FFD166;
  --border:    rgba(255,255,255,0.07);
  --border2:   rgba(0,212,255,0.2);
  --text:      #F1F5F9;
  --muted:     #64748B;
  --card:      rgba(10,16,30,0.8);
}

html,body,.stApp{
  background-color:#03070F !important;
  background-image:
    linear-gradient(rgba(0,212,255,0.025) 1px,transparent 1px),
    linear-gradient(90deg,rgba(0,212,255,0.025) 1px,transparent 1px);
  background-size:40px 40px;
  font-family:'Public Sans',sans-serif !important;
  color:#F1F5F9 !important;
}

#MainMenu,footer,header,.stDeployButton{display:none !important;}
.block-container{padding:0 !important;max-width:100% !important;}
section[data-testid="stSidebar"]{display:none !important;}

::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-thumb{background:#00D4FF;border-radius:2px;}

.t-heading{font-family:'Syne',sans-serif !important;font-weight:800;}
.t-mono{font-family:'IBM Plex Mono',monospace !important;}

.topnav{
  position:sticky;top:0;z-index:100;
  background:rgba(3,7,15,0.85);
  backdrop-filter:blur(16px);
  border-bottom:1px solid var(--border);
  padding:0 32px;
  height:60px;
  display:flex;align-items:center;justify-content:space-between;
}
.brand{display:flex;align-items:center;gap:10px;}
.brand-icon{
  width:34px;height:34px;border-radius:6px;
  background:var(--cyan);
  display:flex;align-items:center;justify-content:center;
  color:#03070F;font-size:18px;font-weight:900;
}
.brand-name{
  font-family:'Syne',sans-serif;font-size:20px;
  font-weight:800;letter-spacing:-0.02em;
  text-transform:uppercase;color:#F1F5F9;
}
.brand-name span{color:var(--cyan);}
.nav-pills{display:flex;gap:4px;}
.nav-pill{
  padding:6px 16px;border-radius:999px;font-size:13px;
  font-weight:500;cursor:pointer;transition:all 0.15s;
  border:1px solid transparent;
}
.nav-pill.active{
  background:var(--cyan);color:#03070F;font-weight:700;
}
.nav-pill:not(.active){
  border-color:var(--border);color:var(--muted);
}
.nav-pill:not(.active):hover{background:rgba(255,255,255,0.04);color:#F1F5F9;}
.nav-right{display:flex;align-items:center;gap:12px;}
.status-dot{width:8px;height:8px;border-radius:50%;background:var(--green);
  box-shadow:0 0 8px var(--green);animation:pulse-dot 2s infinite;}
@keyframes pulse-dot{0%,100%{opacity:1;}50%{opacity:0.4;}}
.doctor-chip{
  display:flex;align-items:center;gap:8px;
  padding:6px 14px;border-radius:999px;
  background:var(--bg2);border:1px solid var(--border);
}
.avatar{
  width:28px;height:28px;border-radius:50%;
  background:linear-gradient(135deg,var(--cyan),#0099bb);
  display:flex;align-items:center;justify-content:center;
  font-size:11px;font-weight:700;color:#03070F;
}

.kpi-strip{
  display:grid;grid-template-columns:repeat(6,1fr);
  gap:0;border-bottom:1px solid var(--border);
}
.kpi-item{
  padding:16px 20px;border-right:1px solid var(--border);
  transition:background 0.15s;
}
.kpi-item:last-child{border-right:none;}
.kpi-item:hover{background:rgba(0,212,255,0.03);}
.kpi-key{font-size:10px;color:var(--muted);text-transform:uppercase;
  letter-spacing:0.1em;font-family:'IBM Plex Mono',monospace;margin-bottom:4px;}
.kpi-val{font-family:'IBM Plex Mono',monospace;font-size:22px;
  font-weight:600;color:var(--cyan);}
.kpi-delta{font-size:10px;margin-top:2px;}

.glass{
  background:var(--card);
  backdrop-filter:blur(12px);
  border:1px solid var(--border);
  border-radius:12px;
}
.glass-cyan{border-color:var(--border2);}

.section-label{
  font-family:'IBM Plex Mono',monospace;
  font-size:10px;color:var(--muted);
  text-transform:uppercase;letter-spacing:0.15em;
  margin-bottom:12px;display:flex;align-items:center;gap:8px;
}
.section-label::before{
  content:'';display:inline-block;
  width:16px;height:1px;background:var(--cyan);
}

.verdict-wrap{
  text-align:center;padding:28px 20px;position:relative;overflow:hidden;
}
.verdict-mal{
  background:radial-gradient(ellipse at center,rgba(255,59,92,0.08) 0%,transparent 70%);
}
.verdict-ben{
  background:radial-gradient(ellipse at center,rgba(0,229,160,0.08) 0%,transparent 70%);
}
.verdict-label{
  font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:var(--muted);text-transform:uppercase;letter-spacing:0.3em;
}
.verdict-text{
  font-family:'Syne',sans-serif;font-size:52px;font-weight:800;
  letter-spacing:-0.03em;margin:8px 0;line-height:1;
}
.verdict-prob{
  font-family:'IBM Plex Mono',monospace;font-size:44px;
  font-weight:600;margin:4px 0;
}

.risk-track{
  height:6px;background:rgba(255,255,255,0.06);
  border-radius:3px;overflow:hidden;margin:6px 0;
}
.risk-fill{height:100%;border-radius:3px;transition:width 0.8s ease;}
.risk-crit{background:linear-gradient(90deg,#FF3B5C,#cc1a37);}
.risk-high{background:linear-gradient(90deg,#FF8C42,#cc6420);}
.risk-mod{background:linear-gradient(90deg,#FFD166,#ccA030);}
.risk-low{background:linear-gradient(90deg,#00E5A0,#00b87a);}

.chip-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;}
.chip{
  background:rgba(3,7,15,0.6);border:1px solid var(--border);
  border-radius:8px;padding:12px;
}
.chip-key{font-size:9px;color:var(--muted);text-transform:uppercase;
  letter-spacing:0.1em;font-family:'IBM Plex Mono',monospace;margin-bottom:4px;}
.chip-val{font-family:'IBM Plex Mono',monospace;font-size:20px;font-weight:600;}

.abcd-row{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0;}
.abcd-cell{
  background:rgba(0,212,255,0.03);border:1px solid rgba(0,212,255,0.1);
  border-radius:8px;padding:12px;text-align:center;
}
.abcd-letter{font-family:'Syne',sans-serif;font-size:18px;font-weight:800;color:var(--cyan);}
.abcd-num{font-family:'IBM Plex Mono',monospace;font-size:16px;font-weight:600;margin:2px 0;}
.abcd-tag{font-size:9px;color:var(--muted);}

.tone-badge{
  display:inline-flex;align-items:center;gap:8px;
  padding:5px 12px;border-radius:999px;
  background:rgba(255,255,255,0.04);border:1px solid var(--border);
  font-family:'IBM Plex Mono',monospace;font-size:11px;
}
.tone-dot{width:12px;height:12px;border-radius:50%;border:1px solid rgba(255,255,255,0.2);}
.clahe-on{color:var(--cyan);background:rgba(0,212,255,0.08);
  border-color:rgba(0,212,255,0.3);padding:3px 10px;border-radius:4px;
  font-size:10px;font-family:'IBM Plex Mono',monospace;}
.clahe-off{color:var(--muted);background:rgba(255,255,255,0.04);
  border:1px solid var(--border);padding:3px 10px;border-radius:4px;
  font-size:10px;font-family:'IBM Plex Mono',monospace;}

.stFileUploader{
  background:rgba(0,212,255,0.02) !important;
  border:2px dashed rgba(0,212,255,0.15) !important;
  border-radius:12px !important;transition:border-color 0.2s !important;
}
.stFileUploader:hover{border-color:rgba(0,212,255,0.4) !important;}
.stFileUploader label{color:var(--muted) !important;font-size:13px !important;}

.stButton>button{
  background:var(--cyan) !important;
  color:#03070F !important;
  border:none !important;border-radius:8px !important;
  font-family:'Syne',sans-serif !important;
  font-weight:700 !important;font-size:14px !important;
  letter-spacing:0.06em !important;text-transform:uppercase !important;
  padding:12px 24px !important;width:100% !important;
  transition:all 0.2s !important;
  box-shadow:0 0 20px rgba(0,212,255,0.25) !important;
}
.stButton>button:hover{
  filter:brightness(1.1) !important;
  box-shadow:0 0 32px rgba(0,212,255,0.4) !important;
  transform:translateY(-1px) !important;
}

.stDownloadButton>button{
  background:rgba(0,212,255,0.08) !important;
  color:var(--cyan) !important;
  border:1px solid rgba(0,212,255,0.45) !important;
  border-radius:8px !important;
  font-family:'IBM Plex Mono',monospace !important;
  font-weight:600 !important;
  font-size:12px !important;
  text-transform:uppercase !important;
  letter-spacing:0.06em !important;
  padding:10px 18px !important;
  width:100% !important;
  box-shadow:none !important;
}
.stDownloadButton>button:hover{
  background:rgba(0,212,255,0.15) !important;
  border-color:var(--cyan) !important;
}

.stTabs [data-baseweb="tab-list"]{
  background:#03070F !important;
  border-bottom:1px solid var(--border) !important;
  padding:0 24px !important;gap:0 !important;
}
.stTabs [data-baseweb="tab"]{
  font-family:'IBM Plex Mono',monospace !important;
  font-size:12px !important;font-weight:500 !important;
  text-transform:uppercase !important;letter-spacing:0.08em !important;
  color:var(--muted) !important;
  padding:14px 20px !important;
  border-bottom:2px solid transparent !important;
}
.stTabs [aria-selected="true"]{
  color:var(--cyan) !important;
  border-bottom:2px solid var(--cyan) !important;
  background:transparent !important;
}
.stTabs [data-baseweb="tab-panel"]{
  padding:0 !important;background:transparent !important;
}

.stNumberInput>div>div>input,.stSelectbox>div>div{
  background:var(--bg2) !important;
  border:1px solid var(--border) !important;
  border-radius:6px !important;color:#F1F5F9 !important;
  font-family:'IBM Plex Mono',monospace !important;
}
.stSlider [data-baseweb="slider"]{padding:8px 0 !important;}

.hist-row{
  display:flex;align-items:center;gap:12px;
  padding:12px 14px;
  background:rgba(10,16,30,0.6);
  border:1px solid var(--border);
  border-radius:8px;margin-bottom:6px;font-size:12px;
}

.eval-box{
  background:var(--bg2);border:1px solid var(--border);
  border-radius:10px;padding:16px;text-align:center;
}
.eval-num{font-family:'IBM Plex Mono',monospace;font-size:28px;
  font-weight:600;color:var(--cyan);}
.eval-lbl{font-size:10px;color:var(--muted);text-transform:uppercase;
  letter-spacing:0.08em;margin-top:3px;}

.stSpinner>div{border-top-color:var(--cyan) !important;}

.disclaimer-bar{
  background:rgba(255,209,102,0.06);
  border-top:1px solid rgba(255,209,102,0.2);
  padding:10px 32px;font-size:11px;
  color:rgba(255,209,102,0.7);
  font-family:'IBM Plex Mono',monospace;
}

.img-label{
  font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:var(--muted);text-align:center;
  padding:6px;background:var(--bg2);border-top:1px solid var(--border);
  letter-spacing:0.08em;text-transform:uppercase;
}

@keyframes border-pulse{
  0%,100%{border-color:rgba(255,59,92,0.4);}
  50%{border-color:rgba(255,59,92,0.8);}
}
.pulse-red{animation:border-pulse 2s infinite;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
RISK = {
    "CRITICAL": {
        "color": "#FF3B5C",
        "cls": "risk-crit",
        "icon": "CRITICAL",
        "action": "Immediate referral — do not wait",
    },
    "HIGH": {"color": "#FF8C42", "cls": "risk-high", "icon": "HIGH", "action": "Book appointment within 1 week"},
    "MODERATE": {"color": "#FFD166", "cls": "risk-mod", "icon": "MODERATE", "action": "Schedule clinical review"},
    "LOW": {"color": "#00E5A0", "cls": "risk-low", "icon": "LOW", "action": "Routine annual monitoring"},
}
TONES = [
    (55, "Very Light", "#F5CBA7"),
    (41, "Light", "#E59866"),
    (28, "Intermediate", "#CA8A5A"),
    (10, "Tan", "#A0522D"),
    (-30, "Brown", "#6B3A2A"),
    (-99, "Dark", "#3D1C0E"),
]


def get_tone(ita: float) -> tuple[str, str]:
    for t, n, c in TONES:
        if ita > t:
            return n, c
    return "Dark", "#3D1C0E"


def abcd_col(v) -> str:
    if not isinstance(v, (int, float)):
        return "#64748B"
    return "#FF3B5C" if v > 0.7 else "#FFD166" if v > 0.4 else "#00E5A0"


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND CONNECTORS (safe wrappers — never modify backend files)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_model():
    from backend.model import load_model

    return load_model()


@st.cache_data(show_spinner=False)
def get_metrics() -> dict:
    from backend.metrics import load_metrics

    m = load_metrics()
    if m:
        return m
    return {}


def run_prediction(img_path: str, age: int = 40, threshold: float = 0.50) -> dict:
    try:
        from risk_engine import compute_risk_score, estimate_abcd
        from backend.model import predict_image

        t0 = time.time()
        result = predict_image(img_path, threshold=threshold)
        elapsed = time.time() - t0
        prob = float(result.get("probability", 0.5))
        ita = float(result.get("ita", 45.0))
        clahe = bool(result.get("clahe_applied", False))
        verdict = "MALIGNANT" if prob >= threshold else "BENIGN"
        risk = compute_risk_score(prob, ita, threshold, age=age)
        abcd = estimate_abcd(prob)
        return {
            "ok": True,
            "verdict": verdict,
            "prob": prob,
            "ita": ita,
            "clahe": clahe,
            "risk": risk,
            "abcd": abcd,
            "elapsed": elapsed,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def run_gradcam(img_path: str):
    try:
        from backend.gradcam import generate_gradcam

        out = generate_gradcam(img_path)
        if out and os.path.exists(out):
            st.session_state["gradcam_path"] = out
            return Image.open(out)
    except Exception:
        pass
    return None


def demo_samples() -> dict[str, str]:
    """Use existing demo_samples/ assets (no dependency on data/)."""
    samples = {
        "Benign": str(ROOT / "demo_samples" / "benign_ISIC_0024322.jpg"),
        "Malignant": str(ROOT / "demo_samples" / "malignant_ISIC_0024310.jpg"),
        "Dark Skin": str(ROOT / "demo_samples" / "dark_skin_ISIC_0024313.jpg"),
    }
    return {k: v for k, v in samples.items() if os.path.exists(v)}


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []
if "sample_path" not in st.session_state:
    st.session_state.sample_path = None

# Pre-warm model silently
get_model()
metrics = get_metrics() or {}
auc_val = metrics.get("auc_roc") if metrics.get("auc_roc") is not None else metrics.get("roc_auc")
acc_val = metrics.get("accuracy")
sen_val = metrics.get("sensitivity")
spe_val = metrics.get("specificity")
f1_val = metrics.get("f1_score")
pre_val = metrics.get("precision")

auc = float(auc_val) if auc_val is not None else None
acc = float(acc_val) if acc_val is not None else None
sen = float(sen_val) if sen_val is not None else None
spe = float(spe_val) if spe_val is not None else None
f1 = float(f1_val) if f1_val is not None else None
pre = float(pre_val) if pre_val is not None else None
gpu = torch.cuda.is_available()

acc_disp = (acc * 100 if acc <= 1 else acc) if acc is not None else None
sen_disp = (sen * 100 if sen <= 1 else sen) if sen is not None else None
spe_disp = (spe * 100 if spe <= 1 else spe) if spe is not None else None
pre_disp = (pre * 100 if pre <= 1 else pre) if pre is not None else None

acc_str = f"{acc_disp:.2f}%" if acc_disp is not None else "N/A"
auc_str = f"{auc:.4f}" if auc is not None else "N/A"
sen_str = f"{sen_disp:.2f}%" if sen_disp is not None else "N/A"
spe_str = f"{spe_disp:.2f}%" if spe_disp is not None else "N/A"
f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
pre_str = f"{pre_disp:.2f}%" if pre_disp is not None else "N/A"

# ══════════════════════════════════════════════════════════════════════════════
# TOP NAV
# ══════════════════════════════════════════════════════════════════════════════
if False:  # Patient List / Archive / Settings — hidden (recoverable)
    _nav_extra_pills = (
        '    <div class="nav-pill">Patient List</div>\n'
        '    <div class="nav-pill">Archive</div>\n'
        '    <div class="nav-pill">Settings</div>\n'
    )
else:
    _nav_extra_pills = ""

st.markdown(
    f"""
<div class="topnav">
  <div class="brand">
    <div class="brand-icon">M</div>
    <div class="brand-name">Melanoma<span>AI</span></div>
  </div>
  <div class="nav-pills">
    <div class="nav-pill active">Dashboard</div>
{_nav_extra_pills}
  </div>
  <div class="nav-right">
    <div style="display:flex;align-items:center;gap:6px;font-size:11px;color:#64748B;
         font-family:'IBM Plex Mono',monospace">
      <div class="status-dot"></div>
      {"GPU Ready" if gpu else "CPU Mode"}
    </div>
    <div class="doctor-chip">
      <div class="avatar">MA</div>
      <div style="font-size:12px;line-height:1.3">
        <div style="font-weight:600;color:#F1F5F9">Clinical Demo</div>
        <div style="font-size:10px;color:#64748B;font-family:'IBM Plex Mono',monospace">
            MelanomaAI v2</div>
      </div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"""
<div class="kpi-strip">
  <div class="kpi-item">
    <div class="kpi-key">Accuracy</div>
    <div class="kpi-val">{acc_str}</div>
    <div class="kpi-delta" style="color:#64748B">Locked v2</div>
  </div>
  <div class="kpi-item">
    <div class="kpi-key">AUC-ROC</div>
    <div class="kpi-val">{auc_str}</div>
    <div class="kpi-delta" style="color:#64748B">Locked v2</div>
  </div>
  <div class="kpi-item">
    <div class="kpi-key">Sensitivity</div>
    <div class="kpi-val">{sen_str}</div>
    <div class="kpi-delta" style="color:#64748B">Malignant recall</div>
  </div>
  <div class="kpi-item">
    <div class="kpi-key">Specificity</div>
    <div class="kpi-val">{spe_str}</div>
    <div class="kpi-delta" style="color:#64748B">Benign clearance</div>
  </div>
  <div class="kpi-item">
    <div class="kpi-key">F1 Score</div>
    <div class="kpi-val">{f1_str}</div>
    <div class="kpi-delta" style="color:#64748B">Harmonic mean</div>
  </div>
  <div class="kpi-item" style="border-right:none">
    <div class="kpi-key">Precision</div>
    <div class="kpi-val">{pre_str}</div>
    <div class="kpi-delta" style="color:#64748B">Positive predictive</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["⬡  Diagnosis", "◈  Model Analytics", "▤  Session Archive"])

with tab1:
    st.markdown("<div style='padding:20px 24px 0'>", unsafe_allow_html=True)
    col_left, col_right = st.columns([5, 7], gap="medium")

    with col_left:
        st.markdown(
            """
        <div class="glass" style="padding:18px;margin-bottom:14px">
          <div class="section-label">Patient Information</div>
        """,
            unsafe_allow_html=True,
        )
        pc1, pc2 = st.columns(2)
        patient_name = pc1.text_input("Full Name", value="", placeholder="e.g. Jane Doe", label_visibility="collapsed")
        patient_id = pc2.text_input("Patient ID", value="", placeholder="e.g. #MEL-0001", label_visibility="collapsed")
        pc3, pc4 = st.columns(2)
        patient_age = pc3.number_input("Age", min_value=1, max_value=120, value=40, label_visibility="collapsed")

        try:
            from backend.metrics import get_threshold

            default_thr = float(get_threshold())
        except Exception:
            default_thr = 0.50

        threshold = pc4.slider(
            "Threshold",
            0.30,
            0.70,
            float(default_thr),
            0.01,
            label_visibility="collapsed",
            help="Detection threshold",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
        <div class="glass" style="padding:18px">
          <div class="section-label">Lesion Capture</div>
        """,
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Drop dermoscopic image here — JPG · PNG (Max 25MB)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        img = None
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, width="stretch")

        if False:  # quick-load sample buttons hidden (logic kept for recovery)
            st.markdown(
                """
            <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                 color:#64748B;margin:12px 0 6px;letter-spacing:0.08em">
              TRY SAMPLE —
            </div>
            """,
                unsafe_allow_html=True,
            )
            s1, s2, s3 = st.columns(3)
            samples = demo_samples()
            if s1.button("Benign", key="s_benign"):
                st.session_state.sample_path = samples.get("Benign")
            if s2.button("Malignant", key="s_mal"):
                st.session_state.sample_path = samples.get("Malignant")
            if s3.button("Dark Skin", key="s_dark"):
                st.session_state.sample_path = samples.get("Dark Skin")

        run_btn = st.button("⬡  RUN DIAGNOSTIC ANALYSIS", key="run_diag")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        img_path_to_use = None
        if uploaded:
            suffix = Path(uploaded.name).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                img_path_to_use = tmp.name
        elif st.session_state.get("sample_path"):
            img_path_to_use = st.session_state.sample_path

        if run_btn and img_path_to_use:
            with st.spinner("Analyzing..."):
                result = run_prediction(img_path_to_use, age=int(patient_age), threshold=float(threshold))

            if result.get("ok"):
                verdict = result["verdict"]
                prob = float(result["prob"])
                ita = float(result["ita"])
                clahe = bool(result["clahe"])
                risk = result.get("risk", {}) or {}
                abcd = result.get("abcd", {}) or {}
                elapsed = float(result.get("elapsed", 0))

                rl = risk.get("level", "MODERATE")
                rs = float(risk.get("score", 50))
                rc = RISK.get(rl, RISK["MODERATE"])
                tone_n, tone_c = get_tone(ita)
                is_mal = verdict == "MALIGNANT"
                v_color = "#FF3B5C" if is_mal else "#00E5A0"
                v_class = "verdict-mal pulse-red" if is_mal else "verdict-ben"

                st.markdown(
                    f"""
                <div class="glass {v_class}" style="border-color:{v_color}40;margin-bottom:12px">
                  <div class="verdict-wrap">
                    <div class="verdict-label">Diagnostic Verdict</div>
                    <div class="verdict-text" style="color:{v_color}">{verdict}</div>
                    <div class="verdict-prob" style="color:{v_color}">{prob*100:.1f}%</div>
                    <div style="font-size:11px;color:#64748B;margin-top:6px;
                         font-family:'IBM Plex Mono',monospace">
                      {"Seek immediate medical attention" if is_mal else "No malignancy detected"}
                      &nbsp;·&nbsp; {elapsed:.2f}s &nbsp;·&nbsp; τ {float(threshold):.2f}
                    </div>
                    <div style="max-width:360px;margin:16px auto 0">
                      <div style="display:flex;justify-content:space-between;
                           font-size:9px;color:#64748B;font-family:'IBM Plex Mono',monospace;
                           text-transform:uppercase;letter-spacing:0.1em;margin-bottom:5px">
                        <span>Low Risk</span><span>High Risk</span>
                      </div>
                      <div class="risk-track">
                        <div class="risk-fill {rc['cls']}" style="width:{min(100, max(0, rs))}%"></div>
                      </div>
                    </div>
                  </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                benign_pct = (1 - prob) * 100
                clahe_html = (
                    '<span class="clahe-on">CLAHE ON</span>' if clahe else '<span class="clahe-off">CLAHE OFF</span>'
                )
                st.markdown(
                    f"""
                <div class="chip-grid" style="margin-bottom:12px">
                  <div class="chip">
                    <div class="chip-key">Malignant %</div>
                    <div class="chip-val" style="color:#FF3B5C">{prob*100:.1f}%</div>
                  </div>
                  <div class="chip">
                    <div class="chip-key">Benign %</div>
                    <div class="chip-val" style="color:#00E5A0">{benign_pct:.1f}%</div>
                  </div>
                  <div class="chip">
                    <div class="chip-key">Risk Score</div>
                    <div class="chip-val" style="color:{rc['color']}">{rs:.0f}/100</div>
                  </div>
                  <div class="chip">
                    <div class="chip-key">ITA Angle</div>
                    <div class="chip-val">{ita:.1f}°</div>
                  </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                <div style="display:flex;align-items:center;justify-content:space-between;
                     padding:10px 14px;background:rgba(10,16,30,0.6);
                     border:1px solid var(--border);border-radius:8px;margin-bottom:12px">
                  <div style="display:flex;align-items:center;gap:10px">
                    <span style="font-size:13px;font-weight:700;color:{rc['color']};
                         font-family:'IBM Plex Mono',monospace">{rc['icon']}</span>
                    <span style="font-size:12px;color:#94A3B8">{rc['action']}</span>
                  </div>
                  <div style="display:flex;align-items:center;gap:8px">
                    <span class="tone-badge">
                      <span class="tone-dot" style="background:{tone_c}"></span>
                      {tone_n} ({ita:.0f}°)
                    </span>
                    {clahe_html}
                    <span style="font-size:10px;color:#475569;font-family:'IBM Plex Mono',monospace">
                        Age {int(patient_age)}yr</span>
                  </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                a = abcd.get("asymmetry", {})
                b_ = abcd.get("border", {})
                c_ = abcd.get("color", {})
                d_ = abcd.get("diameter", {})
                st.markdown(
                    f"""
                <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                     color:#64748B;text-transform:uppercase;letter-spacing:0.1em;
                     margin-bottom:8px">— ABCD Biomarkers</div>
                <div class="abcd-row">
                  <div class="abcd-cell">
                    <div class="abcd-letter">A</div>
                    <div class="abcd-num" style="color:{abcd_col(a.get('score', 0))}">{float(a.get('score', 0)):.3f}</div>
                    <div class="abcd-tag">{a.get('label','—')}</div>
                  </div>
                  <div class="abcd-cell">
                    <div class="abcd-letter">B</div>
                    <div class="abcd-num" style="color:{abcd_col(b_.get('score', 0))}">{float(b_.get('score', 0)):.3f}</div>
                    <div class="abcd-tag">{b_.get('label','—')}</div>
                  </div>
                  <div class="abcd-cell">
                    <div class="abcd-letter">C</div>
                    <div class="abcd-num" style="color:{abcd_col(c_.get('score', 0))}">{float(c_.get('score', 0)):.3f}</div>
                    <div class="abcd-tag">{c_.get('label','—')}</div>
                  </div>
                  <div class="abcd-cell">
                    <div class="abcd-letter">D</div>
                    <div class="abcd-num" style="color:#FFD166">{float(d_.get('mm', 0)):.1f}mm</div>
                    <div class="abcd-tag">{d_.get('label','—')}</div>
                  </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Generate Grad-CAM before PDF so heatmap path exists for embedding
                run_gradcam(img_path_to_use)
                resolved_img = str(Path(img_path_to_use).resolve())
                gradcam_for_pdf = st.session_state.get("gradcam_path")

                case_id = f"MEL-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                report_payload = {
                    "verdict": verdict,
                    "probability": prob,
                    "risk_level": rl,
                    "risk_score": int(rs),
                    "ita": ita,
                    "clahe": clahe,
                    "tone_name": tone_n,
                    "abcd": abcd,
                    "age": int(patient_age),
                    "patient_name": patient_name or "",
                    "patient_id": patient_id or "",
                    "img_path": resolved_img,
                    "gradcam_path": gradcam_for_pdf,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "case_id": case_id,
                    "threshold": float(threshold),
                }
                try:
                    from backend.report_generator import generate_report

                    pdf_path = generate_report(report_payload)
                    st.session_state["last_pdf_path"] = pdf_path
                    st.session_state["last_case_id"] = case_id
                    with open(pdf_path, "rb") as pdf_f:
                        st.download_button(
                            label="Download Diagnostic Report (PDF)",
                            data=pdf_f.read(),
                            file_name=f"MelanomaAI_{case_id}.pdf",
                            mime="application/pdf",
                            key="download_pdf",
                        )
                except Exception as pdf_exc:
                    st.warning(f"PDF report could not be generated: {pdf_exc}")

                st.session_state.history.append(
                    {
                        "ts": datetime.now().strftime("%H:%M:%S"),
                        "file": uploaded.name if uploaded else Path(img_path_to_use).name,
                        "patient": patient_name or "—",
                        "verdict": verdict,
                        "prob": f"{prob*100:.1f}%",
                        "risk": rl,
                        "score": rs,
                        "tone": tone_n,
                        "age": int(patient_age),
                    }
                )
                st.session_state["gcam_src"] = img_path_to_use
                st.session_state["gcam_orig"] = img if uploaded else Image.open(img_path_to_use)
                st.session_state["gcam_ready"] = True
            else:
                st.markdown(
                    f"""
                <div class="glass" style="padding:24px;border-color:#FF3B5C40;text-align:center">
                  <div style="font-size:32px;margin-bottom:8px">!</div>
                  <div style="color:#FF3B5C;font-family:'IBM Plex Mono',monospace;font-size:13px">
                    Analysis failed</div>
                  <div style="color:#64748B;font-size:11px;margin-top:6px">
                    {result.get('error','Unknown error')}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
            <div class="glass" style="padding:60px 20px;text-align:center;height:100%">
              <div style="font-size:48px;margin-bottom:14px;opacity:0.15">⬡</div>
              <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;
                   color:#CBD5E1;letter-spacing:-0.01em">Awaiting Analysis</div>
              <div style="font-size:12px;color:#64748B;margin-top:8px;
                   font-family:'IBM Plex Mono',monospace">
                Upload a dermoscopic image, then run analysis.
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("gcam_ready") and run_btn:
        st.markdown("<div style='padding:16px 24px 24px'>", unsafe_allow_html=True)
        st.markdown('<div class="section-label" style="margin-bottom:14px">Grad-CAM Explainability</div>', unsafe_allow_html=True)
        gc1, gc2, gc3 = st.columns([1, 1, 1], gap="small")
        with st.spinner("Generating heatmap..."):
            gcam = run_gradcam(st.session_state.get("gcam_src", ""))
        with gc1:
            orig = st.session_state.get("gcam_orig")
            if orig:
                st.image(orig, width="stretch")
            st.markdown('<div class="img-label">Original Image</div>', unsafe_allow_html=True)
        with gc2:
            if gcam:
                st.image(gcam, width="stretch")
                st.markdown('<div class="img-label">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    """
                <div class="glass" style="padding:40px;text-align:center">
                  <div style="font-size:28px;opacity:0.3">◈</div>
                  <div style="font-size:11px;color:#64748B;margin-top:8px;
                       font-family:'IBM Plex Mono',monospace">Heatmap unavailable</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        with gc3:
            st.markdown(
                """
            <div class="glass" style="padding:18px;height:100%">
              <div class="section-label">Attention Map Key</div>
              <div style="font-size:12px;line-height:2.4;font-family:'IBM Plex Mono',monospace;color:#94A3B8">
                <span style="color:#FF3B5C">■</span> RED &nbsp; Highest attention<br>
                <span style="color:#FF8C42">■</span> ORANGE &nbsp; Secondary focus<br>
                <span style="color:#FFD166">■</span> YELLOW &nbsp; Border region<br>
                <span style="color:#00E5A0">■</span> GREEN &nbsp; Low attention<br>
                <span style="color:#00D4FF">■</span> BLUE &nbsp; Background
              </div>
              <div style="margin-top:16px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.06);
                   font-size:10px;color:#64748B;line-height:1.7;font-family:'IBM Plex Mono',monospace">
                Verify attention aligns with the lesion (not hair, rulers, or ink marks).
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div style='padding:20px 24px'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Performance Overview — Clean Retraining Holdout</div>', unsafe_allow_html=True)

    ev1, ev2, ev3, ev4 = st.columns(4)
    comps = [
        ("Accuracy", 88.62, acc_disp, "%"),
        ("AUC-ROC", 0.9439, auc, ""),
        ("Sensitivity", 80.73, sen_disp, "%"),
        ("Specificity", 89.90, spe_disp, "%"),
    ]
    for col, (name, v1v, v2v, unit) in zip([ev1, ev2, ev3, ev4], comps):
        v2v_str = f"{v2v:.2f}{unit}" if v2v is not None else "N/A"
        col.markdown(
            f"""
        <div class="eval-box">
          <div class="eval-lbl">{name}</div>
          <div class="eval-num">{v2v_str}</div>
          <div style="font-size:10px;color:#64748B;margin-top:3px;font-family:'IBM Plex Mono',monospace">v1: {v1v:.2f}{unit}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Clean retraining and external validation evaluation charts
    chart_candidates = [
        ("Clean Holdout ROC Curve", [
            ROOT / "outputs" / "clean_retraining" / "roc_curve_clean.png",
            ROOT / "outputs" / "clinical_utility_external" / "roc_curve_test.png",
            ROOT / "outputs" / "roc_curve.png",
        ]),
        ("Calibration Reliability Curve", [
            ROOT / "outputs" / "clean_retraining" / "calibration_curve_clean.png",
            ROOT / "outputs" / "clinical_utility_external" / "calibration_curve_test.png",
        ]),
        ("Decision Curve Analysis (DCA)", [
            ROOT / "outputs" / "clean_retraining" / "dca_curve_clean.png",
            ROOT / "outputs" / "gap8_clinical_utility" / "dca_net_benefit_curves.png",
            ROOT / "outputs" / "clinical_utility_external" / "decision_curve_analysis_test.png",
        ]),
        ("Confusion Matrix (External Test)", [
            ROOT / "outputs" / "clinical_utility_external" / "confusion_matrix_test.png",
            ROOT / "outputs" / "confusion_matrix.png",
        ]),
    ]

    ccols = st.columns(2)
    for i, (label, candidate_list) in enumerate(chart_candidates):
        target_path = next((p for p in candidate_list if p.exists()), None)
        with ccols[i % 2]:
            if target_path and target_path.exists():
                st.image(str(target_path), caption=label, width="stretch")
            else:
                st.markdown(f"<div class='glass' style='padding:18px'>{label} missing</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div style='padding:20px 24px'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Session Archive</div>', unsafe_allow_html=True)
    hist = st.session_state.history
    if hist:
        for i, h in enumerate(reversed(hist)):
            vc = "#FF3B5C" if h["verdict"] == "MALIGNANT" else "#00E5A0"
            st.markdown(
                f"""
            <div class="hist-row">
              <span style="font-family:'IBM Plex Mono',monospace;color:#334155;width:70px">
                {h['ts']}</span>
              <span style="flex:1;color:#CBD5E1;overflow:hidden;text-overflow:ellipsis;
                   white-space:nowrap">{h['file']} · {h['patient']}</span>
              <span style="color:{vc};font-weight:700;font-family:'IBM Plex Mono',monospace;
                   width:110px">{h['verdict']}</span>
              <span style="font-family:'IBM Plex Mono',monospace;color:#00D4FF;width:80px">
                {h['prob']}</span>
              <span style="color:#94A3B8;font-family:'IBM Plex Mono',monospace;font-size:10px;
                   width:160px">{h['tone']} · {h['age']}yr · {h['risk']} ({h['score']:.0f})</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
        if st.button("Clear Archive", key="clear_hist"):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown(
            """
        <div class="glass" style="padding:80px 20px;text-align:center">
          <div style="font-size:44px;margin-bottom:14px;opacity:0.08">▤</div>
          <div style="font-family:'Syne',sans-serif;font-size:16px;color:#CBD5E1;font-weight:700">
            No Cases Analyzed</div>
          <div style="font-size:11px;color:#64748B;margin-top:6px;
               font-family:'IBM Plex Mono',monospace">
            Analyzed cases will appear here during this session
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

auc_bar_str = f"AUC {auc:.4f}" if auc is not None else "AUC N/A"
acc_bar_str = f"ACC {acc_disp:.2f}%" if acc_disp is not None else "ACC N/A"

st.markdown(
    f"""
<div class="disclaimer-bar">
  FOR CLINICAL DECISION SUPPORT ONLY — Not a replacement for professional dermatological diagnosis
  &nbsp;&nbsp;·&nbsp;&nbsp;
  MelanomaAI v2.0 &nbsp;·&nbsp; EfficientNetB3 &nbsp;·&nbsp;
  {auc_bar_str} &nbsp;·&nbsp; {acc_bar_str} &nbsp;·&nbsp;
  {'GPU' if gpu else 'CPU'} &nbsp;·&nbsp;
  {datetime.now().strftime('%B %Y')}
</div>
""",
    unsafe_allow_html=True,
)
