"""Fairness & Robustness — skin-tone equity and CLAHE ablation."""

import streamlit as st

import stitch_shared as shared

shared.render_topbar(
    "Fairness & Robustness",
    "ITA-based routing · CLAHE preprocessing · subgroup performance",
)
shared.render_kpi_strip()

st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
st.markdown(
    '<p class="section-title">⚖️ Fairness & Robustness Analysis</p>'
    '<p class="section-sub">Inference-time skin-tone detection routes dark-skin images through CLAHE '
    "without retraining — reducing fairness gap by ~24%.</p>",
    unsafe_allow_html=True,
)

robustness = shared.load_json_outputs("outputs/robustness_report.json")
fairness = shared.load_json_outputs("outputs/fairness_metrics.json")
risk_data = shared.load_json_outputs("outputs/risk_analysis.json")

shared.render_fairness_bars(robustness, fairness)

if risk_data:
    st.markdown('<p class="section-title" style="margin-top:20px">Risk stratification impact</p>', unsafe_allow_html=True)
    counts = risk_data.get("level_counts", {})
    total = sum(counts.values()) or 1
    for level in ("CRITICAL", "HIGH", "MODERATE", "LOW"):
        n = counts.get(level, 0)
        pct = n / total * 100
        rc = shared.RISK_CONFIG.get(level, shared.RISK_CONFIG["MODERATE"])
        st.markdown(
            f"""
            <div class="fair-bar-row">
                <div class="fair-bar-label"><span>{rc['icon']} {level}</span><span>{n:,} cases ({pct:.1f}%)</span></div>
                <div class="fair-bar-bg"><div class="fair-bar-fill" style="width:{pct}%;background:{rc['color']}"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    fp_rate = risk_data.get("fp_catch_rate", 0) * 100
    st.markdown(
        f"""
        <div class="ml-card" style="margin-top:14px">
            <div style="font-size:13px;font-weight:600;margin-bottom:8px">False-positive mitigation</div>
            <div style="font-size:12px;color:#94a3b8;line-height:1.6">
                Multi-factor risk scoring reclassified <strong style="color:#14b8a8">{fp_rate:.1f}%</strong>
                of flagged malignancies to MODERATE/LOW — reducing unnecessary escalation while preserving sensitivity.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="ml-card" style="margin-top:20px">
        <div style="font-size:13px;font-weight:600;margin-bottom:8px">Novel contribution</div>
        <div style="font-size:12px;color:#94a3b8;line-height:1.65">
            Unlike training-time augmentation alone, MelanomaAI estimates ITA from corner skin pixels at inference,
            applies CLAHE when ITA ≤ 28°, and adjusts clinical risk by skin-tone reliability — improving dark-skin
            performance without modifying melanoma_final.pth.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

shared.render_footer()
st.markdown("</div>", unsafe_allow_html=True)
