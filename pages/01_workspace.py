"""Diagnostic Workspace — viva-hardened demo diagnosis flow."""

import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

import stitch_shared as shared

shared.render_topbar(
    "MelanomaAI · Workspace",
    "Upload or quick-load → Analyze → Risk → Grad-CAM",
)
shared.render_demo_workflow_banner()

st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

col_up, col_res = st.columns([5, 7], gap="large")

with col_up:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-title" style="border:none;padding:0;margin-bottom:8px">📁 Analysis Input</p>',
        unsafe_allow_html=True,
    )
    shared.render_demo_quick_load()

    uploaded = st.file_uploader(
        "Or upload your own dermoscopy image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="workspace_uploader",
    )
    if uploaded is not None:
        st.session_state.demo_source = None
        st.session_state.active_image_path = None

    img, existing_path, display_name = shared.resolve_workspace_image(uploaded)

    if img is not None:
        st.markdown(
            '<div style="border-radius:12px;overflow:hidden;border:1px solid rgba(76,215,246,0.2);margin-top:12px">',
            unsafe_allow_html=True,
        )
        st.image(img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption(f"{img.width}×{img.height}px · {display_name}")
        run_btn = st.button("🔬 Run Diagnostic Analysis", type="primary", use_container_width=True)
    else:
        st.markdown(
            """
            <div class="upload-zone-empty" style="margin-top:12px">
                <div class="upload-icon">🔬</div>
                <div style="font-size:14px;font-weight:500;color:#94a3b8">Quick-load a demo case or upload an image</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        run_btn = False

    st.markdown("</div>", unsafe_allow_html=True)

with col_res:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-title" style="border:none;padding:0">📋 Diagnostic Results</p>',
        unsafe_allow_html=True,
    )

    if img is not None and run_btn:
        st.session_state.analysis_done = False
        threshold = float(st.session_state.threshold)
        patient_age = int(st.session_state.patient_age)

        if existing_path:
            tmp_path = existing_path
        else:
            suffix = Path(display_name).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

        with st.spinner("Analyzing lesion…"):
            try:
                data = shared.run_diagnosis(tmp_path, threshold, patient_age)
            except Exception as exc:
                data = {"success": False, "error": str(exc)}

        if data.get("success"):
            shared.render_diagnosis_results(data, threshold, patient_age)
            st.session_state.last_tmp = tmp_path
            st.session_state.last_img = img
            st.session_state.last_result = True
            st.session_state.last_verdict = data["verdict"]
            st.session_state.last_prob = data["prob"]
            st.session_state.last_risk = data["risk"]
            st.session_state.analysis_done = True

            shared.append_history(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "file": display_name,
                    "verdict": data["verdict"],
                    "prob": f"{data['prob'] * 100:.1f}%",
                    "risk": data["risk"].get("level", "MODERATE"),
                    "score": data["risk"].get("score", 0),
                    "tone": data["tone_name"],
                    "age": patient_age,
                    "thumb_b64": shared.pil_thumb_b64(img),
                    "tmp_path": tmp_path,
                }
            )
            st.success("Analysis complete — scroll down for Grad-CAM explainability.")
        else:
            st.error(f"Analysis failed: {data.get('error', 'Unknown error')}")
    elif img is None:
        st.markdown(
            '<div class="upload-zone-empty" style="padding:40px 12px">'
            '<div style="font-size:14px;color:#64748B">Results appear here after analysis</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Press **Run Diagnostic Analysis** to generate prediction, risk score, and explainability.")

    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.get("analysis_done") and st.session_state.get("last_tmp"):
    st.markdown("---")
    gpath = shared.render_gradcam_section(
        st.session_state.get("last_img"),
        st.session_state.get("last_tmp"),
    )
    exp1, exp2, exp3 = st.columns(3)
    with exp1:
        if st.button("📄 Export PDF Report", use_container_width=True):
            out_pdf = f"report_{int(time.time())}.pdf"
            res = {
                "probability": st.session_state.get("last_prob"),
                "verdict": st.session_state.get("last_verdict"),
            }
            pdf = shared.make_pdf_report(
                st.session_state.get("last_tmp"),
                gpath or st.session_state.get("last_gcam_path"),
                res,
                out_pdf,
            )
            if pdf and os.path.exists(pdf):
                with open(pdf, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF",
                        f,
                        file_name=out_pdf,
                        mime="application/pdf",
                        use_container_width=True,
                    )
            else:
                st.error("Failed to create PDF")
    with exp2:
        st.caption("Next: open **Model Performance** in the sidebar for ROC & confusion matrix.")
    with exp3:
        st.caption("Then: open **Fairness & Robustness** for CLAHE ablation charts.")

st.markdown("</div>", unsafe_allow_html=True)
shared.render_footer()
