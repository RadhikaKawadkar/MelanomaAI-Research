"""Session History — clinical session log with export."""

import time

import streamlit as st

import stitch_shared as shared

shared.render_topbar("Session History", "Cases analyzed during this clinical session")
shared.render_kpi_strip()

st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
st.markdown(
    '<p class="section-title">🗂️ Diagnostic Session History</p>'
    '<p class="section-sub">All Workspace analyses in this session. Export for records or viva demonstration.</p>',
    unsafe_allow_html=True,
)

hist = st.session_state.get("history", [])

if hist:
    mal = sum(1 for h in hist if h.get("verdict") == "MALIGNANT")
    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.markdown(
        f'<div class="metric-box"><div class="metric-label">Total</div><div class="metric-val">{len(hist)}</div></div>',
        unsafe_allow_html=True,
    )
    hc2.markdown(
        f'<div class="metric-box"><div class="metric-label">Malignant</div>'
        f'<div class="metric-val" style="color:#EF4444">{mal}</div></div>',
        unsafe_allow_html=True,
    )
    hc3.markdown(
        f'<div class="metric-box"><div class="metric-label">Benign</div>'
        f'<div class="metric-val" style="color:#22C55E">{len(hist) - mal}</div></div>',
        unsafe_allow_html=True,
    )
    hc4.markdown(
        f'<div class="metric-box"><div class="metric-label">Malignant rate</div>'
        f'<div class="metric-val" style="font-size:22px">{mal / len(hist) * 100:.0f}%</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    for i, h in enumerate(reversed(hist)):
        shared.render_history_card(h, len(hist) - i)

    st.markdown("---")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("🗑️ Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.analysis_done = False
            st.rerun()
    with b2:
        csv_data = shared.export_history_csv(hist)
        st.download_button(
            "⬇️ Export CSV",
            csv_data,
            file_name=f"melanomaai_session_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with b3:
        if hist and st.button("📄 Export last case PDF", use_container_width=True):
            last = hist[-1]
            tmp = last.get("tmp_path")
            if tmp:
                out_pdf = f"session_report_{int(time.time())}.pdf"
                prob_s = last.get("prob", "0").replace("%", "")
                try:
                    prob_f = float(prob_s) / 100.0
                except ValueError:
                    prob_f = 0.5
                res = {"verdict": last.get("verdict"), "probability": prob_f}
                gpath = st.session_state.get("last_gcam_path")
                pdf = shared.make_pdf_report(tmp, gpath, res, out_pdf)
                if pdf:
                    with open(pdf, "rb") as f:
                        st.download_button(
                            "Download PDF",
                            f,
                            file_name=out_pdf,
                            mime="application/pdf",
                            use_container_width=True,
                        )
                else:
                    st.error("PDF generation failed")
            else:
                st.warning("Re-run analysis in Workspace to enable PDF export for this case.")
else:
    st.markdown(
        """
        <div style="text-align:center;padding:80px 24px;color:#475569">
            <div style="font-size:48px;margin-bottom:12px">🗂️</div>
            <div style="font-size:16px;font-weight:500;color:#94a3b8">No cases yet</div>
            <div style="font-size:12px;margin-top:8px;max-width:320px;margin-left:auto;margin-right:auto">
                Upload and analyze a dermoscopy image in <strong style="color:#14b8a8">Workspace</strong> to build your session log.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

shared.render_footer()
st.markdown("</div>", unsafe_allow_html=True)
