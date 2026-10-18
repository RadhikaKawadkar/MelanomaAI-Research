"""
MelanomaAI v2 — shared Streamlit UI shell.
Used by app.py (navigation hub) and all pages/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "demo_samples"

DEMO_SAMPLES = {
    "benign": {
        "file": "benign_ISIC_0024322.jpg",
        "title": "Benign",
        "hint": "Light skin · low malignancy probability",
        "icon": "🟢",
    },
    "malignant": {
        "file": "malignant_ISIC_0024310.jpg",
        "title": "Malignant",
        "hint": "Dark skin · CLAHE on · high probability",
        "icon": "🔴",
    },
    "dark_skin": {
        "file": "dark_skin_ISIC_0024313.jpg",
        "title": "Dark skin",
        "hint": "Fairness routing · CLAHE preprocessing",
        "icon": "🟤",
    },
}

RISK_CONFIG = {
    "CRITICAL": {
        "color": "#EF4444",
        "icon": "🔴",
        "action": "Immediate referral — do not wait",
        "cls": "background:linear-gradient(90deg,#EF4444,#DC2626)",
    },
    "HIGH": {
        "color": "#F97316",
        "icon": "🟠",
        "action": "Book appointment within 1 week",
        "cls": "background:linear-gradient(90deg,#F97316,#EA580C)",
    },
    "MODERATE": {
        "color": "#EAB308",
        "icon": "🟡",
        "action": "Schedule clinical review",
        "cls": "background:linear-gradient(90deg,#EAB308,#CA8A04)",
    },
    "LOW": {
        "color": "#22C55E",
        "icon": "🟢",
        "action": "Routine annual monitoring",
        "cls": "background:linear-gradient(90deg,#22C55E,#16A34A)",
    },
}

TONE_TABLE = [
    (55, "Very Light", "#F5CBA7"),
    (41, "Light", "#E59866"),
    (28, "Intermediate", "#CA8A5A"),
    (10, "Tan", "#A0522D"),
    (-30, "Brown", "#6B3A2A"),
    (-99, "Dark", "#3D1C0E"),
]

DEFAULT_METRICS = {
    "accuracy": 92.80,
    "auc_roc": 0.9770,
    "sensitivity": 79.51,
    "specificity": 97.38,
    "f1_score": 0.8498,
    "precision": 91.27,
}

V1_METRICS = {
    "accuracy": 88.62,
    "auc_roc": 0.9439,
    "sensitivity": 80.73,
    "specificity": 89.90,
    "f1_score": 0.6478,
    "precision": 65.0,
}

ABCD_LABELS = {
    "asymmetry": ("A", "Asymmetry"),
    "border": ("B", "Border irregularity"),
    "color": ("C", "Color variation"),
    "diameter": ("D", "Diameter risk"),
}


def setup_paths() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "threshold" not in st.session_state:
        st.session_state.threshold = get_default_threshold()
    if "patient_age" not in st.session_state:
        st.session_state.patient_age = 40
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "demo_source" not in st.session_state:
        st.session_state.demo_source = None
    if "active_image_path" not in st.session_state:
        st.session_state.active_image_path = None


def get_default_threshold() -> float:
    for p in ("outputs/optimal_threshold.json", "optimal_threshold.json"):
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return float(json.load(f).get("optimal_threshold", 0.50))
            except Exception:
                pass
    return 0.50


@st.cache_data(show_spinner=False)
def get_metrics() -> dict:
    for p in (
        "outputs/metrics.json",
        "outputs/metrics_run8.json",
        "outputs/metrics_phase1.json",
    ):
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return dict(DEFAULT_METRICS)


def inject_css() -> None:
    css_path = ROOT / "stitch_ui.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def get_tone(ita: float) -> tuple[str, str]:
    for thr, name, color in TONE_TABLE:
        if ita > thr:
            return name, color
    return "Dark", "#3D1C0E"


def abcd_color(v) -> str:
    if not isinstance(v, (int, float)):
        return "#94A3B8"
    if v > 0.7:
        return "#EF4444"
    if v > 0.4:
        return "#EAB308"
    return "#22C55E"


def fmt_pct(value, scale: float = 100.0) -> str:
    if value is None:
        return "—"
    v = float(value)
    if v <= 1.0 and scale == 100.0:
        v *= 100.0
    return f"{v:.2f}%"


def render_sidebar() -> None:
    metrics = get_metrics()
    acc = metrics.get("accuracy", DEFAULT_METRICS["accuracy"])
    auc = metrics.get("auc_roc", DEFAULT_METRICS["auc_roc"])
    sen = metrics.get("sensitivity", DEFAULT_METRICS["sensitivity"])
    spe = metrics.get("specificity", DEFAULT_METRICS["specificity"])
    f1 = metrics.get("f1_score", DEFAULT_METRICS["f1_score"])
    pre = metrics.get("precision", DEFAULT_METRICS["precision"])
    gpu_ok = torch.cuda.is_available()

    st.markdown(
        """
        <div style="font-family:'DM Serif Display',serif;font-size:20px;color:#F8FAFC;
             padding:8px 4px 12px;border-bottom:1px solid rgba(76,215,246,0.15);margin-bottom:14px">
            Melanoma<span style="color:#14B8A8">AI</span>
            <div style="font-family:'DM Sans',sans-serif;font-size:11px;color:#94A3B8;
                 margin-top:3px;font-weight:400">Clinical Decision Support v2</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Clinical settings**")
    st.session_state.patient_age = st.number_input(
        "Patient age",
        min_value=1,
        max_value=120,
        value=int(st.session_state.patient_age),
        key="sidebar_patient_age",
    )
    st.session_state.threshold = st.slider(
        "Detection threshold",
        0.30,
        0.70,
        float(st.session_state.threshold),
        0.01,
        key="sidebar_threshold",
    )

    st.markdown("---")
    st.markdown("**Production metrics**")
    rows = [
        ("Accuracy", fmt_pct(acc)),
        ("AUC-ROC", f"{float(auc):.4f}" if auc is not None else "—"),
        ("Sensitivity", fmt_pct(sen)),
        ("Specificity", fmt_pct(spe)),
        ("F1", f"{float(f1):.4f}" if f1 is not None else "—"),
        ("Precision", fmt_pct(pre)),
    ]
    for label, val in rows:
        c1, c2 = st.columns([3, 2])
        c1.markdown(
            f"<span style='color:#94A3B8;font-size:12px'>{label}</span>",
            unsafe_allow_html=True,
        )
        c2.markdown(
            f"<span style='color:#14B8A8;font-size:13px;font-weight:600'>{val}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    gpu_bg = "rgba(34,197,94,0.1)" if gpu_ok else "rgba(71,85,105,0.2)"
    gpu_border = "#22C55E" if gpu_ok else "#475569"
    gpu_color = "#22C55E" if gpu_ok else "#94A3B8"
    gpu_label = "GPU ready" if gpu_ok else "CPU mode"
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:8px;padding:8px 10px;
             background:{gpu_bg};border:1px solid {gpu_border};border-radius:8px;font-size:12px">
            {'🟢' if gpu_ok else '⚪'}
            <span style="color:{gpu_color}">{gpu_label}</span>
        </div>
        <div class="disclaimer">⚠️ Clinical assistance only. Not a replacement for professional diagnosis.</div>
        """,
        unsafe_allow_html=True,
    )


def render_topbar(page_title: str, subtitle: str = "") -> None:
    gpu_ok = torch.cuda.is_available()
    sub = subtitle or "EfficientNetB3 · HAM10000 balanced · Production v2"
    gpu_border = "#22C55E" if gpu_ok else "#475569"
    st.markdown(
        f"""
        <div class="topbar">
          <div>
            <div class="topbar-title">{page_title}</div>
            <div style="font-size:11px;color:#64748B;margin-top:2px">{sub}</div>
          </div>
          <div style="display:flex;align-items:center;gap:10px">
            <div class="topbar-badge">v2.0 Production</div>
            <div class="topbar-badge" style="border-color:{gpu_border}">
                {'GPU' if gpu_ok else 'CPU'}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_strip() -> None:
    metrics = get_metrics()
    acc = metrics.get("accuracy", DEFAULT_METRICS["accuracy"])
    auc = metrics.get("auc_roc", DEFAULT_METRICS["auc_roc"])
    sen = metrics.get("sensitivity", DEFAULT_METRICS["sensitivity"])
    spe = metrics.get("specificity", DEFAULT_METRICS["specificity"])
    f1 = metrics.get("f1_score", DEFAULT_METRICS["f1_score"])
    pre = metrics.get("precision", DEFAULT_METRICS["precision"])

    def _pct(val):
        if val is None:
            return "—"
        v = float(val)
        if v <= 1.0:
            v *= 100.0
        return f"{v:.1f}%"

    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi-card"><div class="kpi-label">Accuracy</div>
            <div class="kpi-value">{_pct(acc)}</div></div>
          <div class="kpi-card"><div class="kpi-label">AUC-ROC</div>
            <div class="kpi-value">{float(auc):.4f}</div></div>
          <div class="kpi-card"><div class="kpi-label">Sensitivity</div>
            <div class="kpi-value">{_pct(sen)}</div></div>
          <div class="kpi-card"><div class="kpi-label">Specificity</div>
            <div class="kpi-value">{_pct(spe)}</div></div>
          <div class="kpi-card"><div class="kpi-label">F1 Score</div>
            <div class="kpi-value">{float(f1):.4f}</div></div>
          <div class="kpi-card"><div class="kpi-label">Precision</div>
            <div class="kpi-value">{_pct(pre)}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    metrics = get_metrics()
    auc = metrics.get("auc_roc", DEFAULT_METRICS["auc_roc"])
    acc = metrics.get("accuracy", DEFAULT_METRICS["accuracy"])
    thr = st.session_state.get("threshold", get_default_threshold())
    gpu_ok = torch.cuda.is_available()
    st.markdown(
        f"""
        <div class="app-footer">
            MelanomaAI v2.0 · EfficientNetB3 · AUC {float(auc):.4f} ·
            Accuracy {fmt_pct(acc)} · Threshold {float(thr):.2f} ·
            {'GPU' if gpu_ok else 'CPU'}
        </div>
        """,
        unsafe_allow_html=True,
    )


def append_history(entry: dict) -> None:
    st.session_state.history.append(entry)


def get_demo_path(key: str) -> Path | None:
    meta = DEMO_SAMPLES.get(key)
    if not meta:
        return None
    path = DEMO_DIR / meta["file"]
    return path if path.exists() else None


def render_demo_quick_load() -> None:
    """Sidebar-style quick load for live viva demonstration."""
    st.markdown(
        '<p style="font-size:12px;font-weight:600;color:#94a3b8;margin:0 0 8px 0">'
        "⚡ VIVA QUICK LOAD</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for col, (key, meta) in zip(cols, DEMO_SAMPLES.items()):
        with col:
            if st.button(
                f"{meta['icon']} {meta['title']}",
                key=f"demo_load_{key}",
                use_container_width=True,
                help=meta["hint"],
            ):
                path = get_demo_path(key)
                if path:
                    st.session_state.demo_source = key
                    st.session_state.active_image_path = str(path)
                    st.session_state.analysis_done = False
                    st.session_state.run_analysis = False
                    st.rerun()
                else:
                    st.error(f"Missing demo file: {meta['file']}")


def resolve_workspace_image(uploaded) -> tuple[Image.Image | None, str | None, str]:
    """Return (PIL image, filesystem path, display name) from upload or demo selection."""
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        return img, None, uploaded.name

    path = st.session_state.get("active_image_path")
    if path and os.path.exists(path):
        img = Image.open(path).convert("RGB")
        name = Path(path).name
        if st.session_state.get("demo_source"):
            meta = DEMO_SAMPLES.get(st.session_state.demo_source, {})
            name = f"[Demo] {meta.get('title', name)}"
        return img, path, name

    return None, None, ""


def run_diagnosis(tmp_path: str, threshold: float, patient_age: int) -> dict:
    """Execute inference pipeline. Does not modify risk_engine or backend logic."""
    from backend.model import predict_image
    from risk_engine import compute_risk_score, estimate_abcd

    import time

    t0 = time.time()
    result = predict_image(tmp_path, threshold=threshold)
    elapsed = time.time() - t0
    prob = float(result.get("probability", 0.5))
    ita = float(result.get("ita", 45.0))
    clahe = result.get("clahe_applied", False)
    verdict = "MALIGNANT" if prob >= threshold else "BENIGN"
    risk = compute_risk_score(prob, ita, threshold, age=patient_age)
    abcd = estimate_abcd(prob)
    tone_name, tone_color = get_tone(ita)
    return {
        "success": True,
        "prob": prob,
        "ita": ita,
        "clahe": clahe,
        "verdict": verdict,
        "risk": risk,
        "abcd": abcd,
        "tone_name": tone_name,
        "tone_color": tone_color,
        "elapsed": elapsed,
    }


def render_diagnosis_results(data: dict, threshold: float, patient_age: int) -> None:
    render_verdict_card(data["verdict"], data["prob"], data["elapsed"], threshold)
    render_confidence_gauge(data["prob"], threshold, data["verdict"])
    render_risk_panel(data["risk"], patient_age)
    render_skin_tone_row(
        data["ita"], data["tone_name"], data["tone_color"], data["clahe"], patient_age
    )
    render_abcd_biomarkers(data["abcd"])


def render_demo_workflow_banner() -> None:
    st.markdown(
        """
        
        <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;padding:12px 14px;
             background:rgba(76,215,246,0.06);border:1px solid rgba(76,215,246,0.2);border-radius:12px;font-size:11px">
            <span style="color:#5eead4;font-weight:600">VIVA FLOW →</span>
            <span style="color:#94a3b8">① Quick load</span><span style="color:#475569">→</span>
            <span style="color:#94a3b8">② Analyze</span><span style="color:#475569">→</span>
            <span style="color:#94a3b8">③ Verdict & risk</span><span style="color:#475569">→</span>
            <span style="color:#94a3b8">④ Grad-CAM</span><span style="color:#475569">→</span>
            <span style="color:#94a3b8">⑤ Performance / Fairness pages</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_json_outputs(*paths: str) -> dict | None:
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def _pct_num(value) -> float:
    if value is None:
        return 0.0
    v = float(value)
    return v * 100.0 if v <= 1.0 else v


def render_verdict_card(verdict: str, prob: float, elapsed: float, threshold: float) -> None:
    is_mal = verdict == "MALIGNANT"
    cls = "malignant" if is_mal else "benign"
    color = "#EF4444" if is_mal else "#22C55E"
    icon = "⚠️" if is_mal else "✅"
    sub = "Seek immediate dermatology referral" if is_mal else "No malignancy detected at current threshold"
    st.markdown(
        f"""
        
        <div class="verdict-card {cls}">
            <div style="font-size:28px;margin-bottom:4px">{icon}</div>
            <div class="verdict-label" style="color:{color}">{verdict}</div>
            <div class="verdict-prob" style="color:{color}">{prob * 100:.1f}%</div>
            <div class="verdict-meta">{sub}</div>
            <div class="verdict-meta" style="margin-top:6px">⏱ {elapsed:.2f}s inference · threshold {threshold:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confidence_gauge(prob: float, threshold: float, verdict: str) -> None:
    pct = prob * 100.0
    is_mal = verdict == "MALIGNANT"
    color = "#EF4444" if is_mal else "#14B8A8"
    dist = abs(prob - threshold) * 100
    conf_label = "High confidence" if dist > 15 else "Moderate" if dist > 5 else "Near threshold"
    st.markdown(
        f"""
        <div class="gauge-wrap ml-card" style="padding:16px">
            <div class="gauge-ring" style="background:conic-gradient({color} 0% {pct}%, #1e293b {pct}% 100%)">
                <div class="gauge-inner">
                    <span style="color:{color}">{pct:.0f}%</span>
                    <span style="font-size:9px;color:#64748B">malignant</span>
                </div>
            </div>
            <div class="gauge-caption">
                <strong style="color:#e2e8f0;font-size:13px">Model confidence</strong><br>
                {conf_label} — {dist:.1f} pts from decision boundary ({threshold:.2f}).<br>
                Higher separation from threshold yields more reliable clinical action.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_risk_panel(risk: dict, patient_age: int) -> None:
    level = risk.get("level", "MODERATE")
    score = risk.get("score", 0)
    rc = RISK_CONFIG.get(level, RISK_CONFIG["MODERATE"])
    desc = risk.get("description", "")
    action = risk.get("action", rc["action"])
    fp = risk.get("fp_likelihood", "")
    factors = risk.get("factors", {})
    age_pts = float(factors.get("age_adjustment", 0))
    base = float(factors.get("base_probability", 0))
    conf = float(factors.get("confidence", 0))
    skin = float(factors.get("skin_reliability", 0))
    if patient_age <= 40:
        age_note = "Age factor: 0 pts (applies only for age &gt; 40)"
    else:
        age_note = f"Age factor: +{age_pts:.1f} pts applied"
    breakdown = (
        '<div class="risk-desc" style="margin-top:10px;font-size:11px;line-height:1.65">'
        f'<div style="color:#94a3b8;margin-bottom:4px">Risk Score (Patient Age: {patient_age})</div>'
        f"Base probability: {base:.1f} pts<br>"
        f"Confidence factor: {conf:.1f} pts<br>"
        f"Skin reliability: {skin:.1f} pts<br>"
        f"Age adjustment: +{age_pts:.1f} pts (age &gt; 40 only)<br>"
        f'<strong style="color:#e2e8f0">Total score: {score:.1f} / 100</strong>'
        "</div>"
    )
    st.markdown(
        f"""
        <div class="ml-card risk-panel">
            <div class="risk-header">
                <span class="risk-level-badge" style="background:{rc['color']}22;color:{rc['color']};border:1px solid {rc['color']}55">
                    {rc['icon']} {level} RISK
                </span>
                <span class="risk-score-num" style="color:{rc['color']}">{score:.0f}<span style="font-size:14px;color:#64748B">/100</span></span>
            </div>
            <div class="risk-desc" style="font-size:11px;color:#94a3b8;margin-top:6px">{age_note}</div>
            <div class="risk-bar-bg">
                <div class="risk-bar-fill" style="{rc['cls']};width:{min(100, score)}%"></div>
            </div>
            <div class="risk-action" style="color:{rc['color']}">Action: {action}</div>
            <div class="risk-desc">{desc}</div>
            <div class="risk-desc" style="margin-top:4px;font-size:11px">False-positive likelihood: {fp}</div>
            {breakdown}
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_abcd_biomarkers(abcd: dict) -> None:
    import html as html_escape

    st.markdown(
        '<p style="font-size:13px;font-weight:600;margin:12px 0 10px">'
        "ABCD Rule Biomarkers</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    for col, (key, (letter, name)) in zip(cols, ABCD_LABELS.items()):
        block = abcd.get(key, {})
        if key == "diameter":
            val = block.get("mm", 0)
            pct = min(100, (val / 14.0) * 100)
            color = "#EAB308" if val > 6 else "#22C55E"
            val_str = f"{val:.1f} mm"
        else:
            val = block.get("score", 0)
            pct = float(val) * 100
            color = abcd_color(val)
            val_str = f"{val:.2f}"
        lbl = html_escape.escape(str(block.get("label", "—")))
        name_esc = html_escape.escape(name)
        card = (
            '<div class="bio-card">'
            + f'<div class="bio-letter">{letter}</div>'
            + f'<div class="bio-name">{name_esc}</div>'
            + '<div class="bio-bar-bg">'
            + f'<div class="bio-bar-fill" style="width:{pct:.0f}%;background:{color}"></div>'
            + "</div>"
            + f'<div class="bio-val" style="color:{color}">{val_str}</div>'
            + f'<div class="bio-lbl">{lbl}</div>'
            + "</div>"
        )
        col.markdown(card, unsafe_allow_html=True)


def render_skin_tone_row(ita: float, tone_name: str, tone_color: str, clahe: bool, age: int) -> None:
    clahe_tag = (
        '<span style="background:rgba(13,148,136,0.2);color:#5EEAD4;border:1px solid #0D9488;'
        'border-radius:6px;padding:2px 8px;font-size:10px;margin-left:8px">CLAHE enhanced</span>'
        if clahe
        else '<span style="background:rgba(71,85,105,0.2);color:#94A3B8;border:1px solid #475569;'
        'border-radius:6px;padding:2px 8px;font-size:10px;margin-left:8px">Standard path</span>'
    )
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;flex-wrap:wrap;gap:8px;padding:12px 14px;
             background:rgba(255,255,255,0.03);border-radius:10px;margin-bottom:14px;font-size:12px">
            <span class="tone-dot" style="background:{tone_color}"></span>
            <span><strong>{tone_name}</strong> · ITA {ita:.1f}°</span>{clahe_tag}
            <span style="margin-left:auto;color:#64748B">Patient age {age} yr</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gradcam_section(img, tmp_path: str) -> str | None:
    """Generate and display Grad-CAM gallery. Returns output path if successful."""
    st.markdown('<p class="section-title">🔥 Grad-CAM Explainability</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Highlighted regions influenced the model prediction. '
        "Red zones indicate highest attention on lesion tissue.</p>",
        unsafe_allow_html=True,
    )
    gpath = None
    gcam_pil = None
    try:
        from backend.gradcam import generate_gradcam

        with st.spinner("Generating explainability heatmap…"):
            gpath = generate_gradcam(tmp_path)
            if gpath and os.path.exists(gpath):
                gcam_pil = Image.open(gpath)
    except Exception as exc:
        st.warning(f"Grad-CAM unavailable: {exc}")
        return None

    if gcam_pil is None:
        st.info("Grad-CAM could not be generated for this image.")
        return None

    st.session_state.last_gcam_path = gpath
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.markdown('<div class="gcam-card">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('<div class="gcam-cap">Original dermoscopy<span>Input image before overlay</span></div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="gcam-card">', unsafe_allow_html=True)
        st.image(gcam_pil, use_container_width=True)
        st.markdown('<div class="gcam-cap">Grad-CAM overlay<span>Attention heatmap blended on lesion</span></div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(
            """
            <div class="gcam-explainer">
                <div style="font-size:14px;font-weight:600;margin-bottom:10px">How to read this</div>
                <div style="font-size:12px;line-height:1.8;color:#94a3b8">
                    🔴 <b style="color:#fca5a5">Red</b> — primary suspicious region<br>
                    🟡 <b style="color:#fcd34d">Yellow</b> — secondary attention<br>
                    🟢 <b style="color:#86efac">Green</b> — moderate signal<br>
                    🔵 <b style="color:#93c5fd">Blue</b> — background (ignored)
                </div>
                <p style="font-size:11px;color:#64748B;margin-top:14px;line-height:1.5">
                    Highlighted regions influenced model prediction. Verify attention aligns with the lesion—not hair, rulers, or ink marks.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return gpath


def render_v1_v2_comparison() -> None:
    v2 = get_metrics()
    st.markdown('<p class="section-title">v1 → v2 Improvement</p>', unsafe_allow_html=True)
    metrics_keys = [
        ("Accuracy", "accuracy", True),
        ("AUC-ROC", "auc_roc", False),
        ("Sensitivity", "sensitivity", True),
        ("Specificity", "specificity", True),
        ("F1 Score", "f1_score", False),
    ]

    rows_html = ""
    for label, key, is_pct in metrics_keys:
        v1v = V1_METRICS.get(key, 0)
        v2v = v2.get(key, DEFAULT_METRICS.get(key, 0))
        if is_pct:
            d1, d2 = fmt_pct(v1v), fmt_pct(v2v)
            delta = _pct_num(v2v) - _pct_num(v1v)
            delta_s = f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%"
        else:
            d1, d2 = f"{float(v1v):.4f}", f"{float(v2v):.4f}"
            delta = float(v2v) - float(v1v)
            delta_s = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        rows_html += f"""
        <tr>
            <td style="padding:8px;color:#94a3b8">{label}</td>
            <td style="padding:8px;text-align:center">{d1}</td>
            <td style="padding:8px;text-align:center;color:#14b8a8;font-weight:600">{d2}</td>
            <td style="padding:8px;text-align:center;color:#22c55e">{delta_s}</td>
        </tr>"""

    st.markdown(
        f"""
        <div class="ml-card" style="overflow-x:auto">
            <table style="width:100%;border-collapse:collapse;font-size:13px">
                <thead><tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
                    <th style="text-align:left;padding:8px;color:#64748B">Metric</th>
                    <th style="padding:8px;color:#64748B">v1</th>
                    <th style="padding:8px;color:#64748B">v2 (production)</th>
                    <th style="padding:8px;color:#64748B">Δ</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_explainers() -> None:
    st.markdown(
        """
        <div class="metric-explainer">
            <div class="explain-card">
                <strong>AUC-ROC</strong>
                Area under the receiver operating characteristic curve. Measures discrimination between benign and malignant across all thresholds. Closer to 1.0 is better; &gt;0.95 is excellent for medical AI.
            </div>
            <div class="explain-card">
                <strong>Sensitivity</strong>
                True positive rate — malignant cases correctly flagged. Critical for screening: missing melanoma is costlier than extra referrals.
            </div>
            <div class="explain-card">
                <strong>Specificity</strong>
                True negative rate — benign cases correctly cleared. High specificity reduces unnecessary biopsies and patient anxiety.
            </div>
        
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_fairness_bars(robustness: dict | None, fairness: dict | None) -> None:
    st.markdown('<p class="section-title">Skin-tone performance</p>', unsafe_allow_html=True)
    if robustness and "light_skin" in robustness:
        light = robustness["light_skin"]
        dark = robustness["dark_skin"]
        gap = robustness.get("fairness_gap", 0) * 100
        for label, block, color in [
            ("Light skin (ITA &gt; 28°)", light, "#22C55E"),
            ("Dark skin (ITA ≤ 28°)", dark, "#14B8A8"),
        ]:
            acc = block.get("accuracy", 0) * 100
            st.markdown(
                f"""
                <div class="fair-bar-row">
                    <div class="fair-bar-label"><span>{label}</span><span style="color:{color}">{acc:.1f}% acc · AUC {block.get('auc', 0):.3f}</span></div>
                    <div class="fair-bar-bg"><div class="fair-bar-fill" style="width:{acc}%;background:{color}"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<p class="section-sub">Fairness gap (accuracy): <strong style="color:#14b8a8">{gap:.2f}%</strong> — lower is better.</p>',
            unsafe_allow_html=True,
        )

    st.markdown('<p class="section-title" style="margin-top:20px">CLAHE ablation study</p>', unsafe_allow_html=True)
    ablation = [
        ("Light skin", 91.31, "#22C55E", "No CLAHE needed"),
        ("Dark skin (baseline)", 81.88, "#EF4444", "Without preprocessing"),
        ("Dark skin + CLAHE", 84.14, "#14B8A8", "~24% fairness gap reduction"),
    ]
    for label, acc, color, note in ablation:
        st.markdown(
            f"""
            <div class="fair-bar-row">
                <div class="fair-bar-label"><span>{label}</span><span style="color:{color}">{acc:.2f}%</span></div>
                <div class="fair-bar-bg"><div class="fair-bar-fill" style="width:{acc}%;background:{color}"></div></div>
                <div style="font-size:10px;color:#64748B;margin-top:4px">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if fairness:
        gap = fairness.get("prob_gap", 0) * 100
        st.markdown(
            f"""
            <div class="ml-card" style="margin-top:16px">
                <div style="font-size:13px;font-weight:600;margin-bottom:8px">Probability calibration gap</div>
                <div style="font-size:12px;color:#94a3b8;line-height:1.6">
                    Mean predicted malignancy probability differs by <strong style="color:#eab308">{gap:.1f}%</strong>
                    between light (n={fairness.get('light_skin_n', '—')}) and dark (n={fairness.get('dark_skin_n', '—')}) subgroups.
                    Inference-time CLAHE routing mitigates this without retraining.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_history_card(h: dict, index: int) -> None:
    vc = "mal" if h.get("verdict") == "MALIGNANT" else "ben"
    rc = RISK_CONFIG.get(h.get("risk", "MODERATE"), RISK_CONFIG["MODERATE"])
    thumb = h.get("thumb_b64", "")
    thumb_html = (
        f'<img class="hist-thumb" src="data:image/jpeg;base64,{thumb}" alt="thumb"/>'
        if thumb
        else '<div class="hist-thumb" style="background:#1e293b;display:flex;align-items:center;justify-content:center;font-size:24px">🔬</div>'
    )
    st.markdown(
        f"""
        <div class="hist-card">
            {thumb_html}
            <div>
                <div style="font-weight:600;color:#e2e8f0;margin-bottom:4px">{h.get('file', 'Unknown')}</div>
                <div style="font-size:11px;color:#64748B">{h.get('time', '')} · {h.get('tone', '')} · Age {h.get('age', '—')}</div>
            </div>
            <div style="text-align:right">
                <span class="hist-badge {vc}">{h.get('verdict', '')}</span>
                <div style="font-family:'JetBrains Mono',monospace;color:#14b8a8;font-size:14px;margin-top:6px">{h.get('prob', '')}</div>
                <div style="font-size:10px;color:{rc['color']};margin-top:4px">{h.get('risk', '')} · {h.get('score', '')}/100</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pil_thumb_b64(pil_img, size: int = 96) -> str:
    import base64
    from io import BytesIO

    thumb = pil_img.copy()
    thumb.thumbnail((size, size))
    buf = BytesIO()
    thumb.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def export_history_csv(history: list) -> str:
    import io
    import csv

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["time", "file", "verdict", "prob", "risk", "score", "tone", "age"])
    w.writeheader()
    for row in history:
        w.writerow({k: row.get(k, "") for k in w.fieldnames})
    return buf.getvalue()


def make_pdf_report(img_path, gcam_path, result, out_pdf):
    try:
        base = Image.open(img_path).convert("RGB")
        w, h = base.width, base.height
        canvas_h = h + 260
        canvas = Image.new("RGB", (w, canvas_h), (15, 20, 27))
        canvas.paste(base, (0, 0))
        draw = ImageDraw.Draw(canvas)
        try:
            f1 = ImageFont.truetype("arial.ttf", 22)
            f2 = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            f1 = ImageFont.load_default()
            f2 = ImageFont.load_default()
        y = h + 16
        draw.text((12, y), "MelanomaAI v2 — Diagnostic Report", fill=(220, 230, 238), font=f1)
        y += 36
        prob = result.get("probability")
        verdict = result.get("verdict", "—")
        draw.text((12, y), f"Verdict: {verdict}", fill=(220, 230, 238), font=f2)
        if prob is not None:
            draw.text((220, y), f"Probability: {prob:.3f}", fill=(200, 220, 200), font=f2)
        if gcam_path and os.path.exists(gcam_path):
            g = Image.open(gcam_path).convert("RGB")
            g.thumbnail((w // 3, w // 3))
            canvas.paste(g, (w - g.width - 12, h + 12))
        canvas.save(out_pdf, "PDF", resolution=100.0)
        return out_pdf
    except Exception:
        return None
