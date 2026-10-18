# -*- coding: utf-8 -*-
"""Patch stitch_shared.py: risk panel + ABCD columns; ASCII-safe sidebar headers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "stitch_shared.py"
text = path.read_text(encoding="utf-8")

risk_fn = '''
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
        '<motion-prevent><div class="risk-desc" style="margin-top:10px;font-size:11px;line-height:1.65">'
        f'<div style="color:#94a3b8;margin-bottom:4px">Risk Score (Patient Age: {patient_age})</div>'
        f"Base probability: {base:.1f} pts<br>"
        f"Confidence factor: {conf:.1f} pts<br>"
        f"Skin reliability: {skin:.1f} pts<br>"
        f"Age adjustment: +{age_pts:.1f} pts (age &gt; 40 only)<br>"
        f'<strong style="color:#e2e8f0">Total score: {score:.1f} / 100</strong>'
        "</div>"
    )
    breakdown = breakdown.replace("<motion-prevent>", "")
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
            <div class="risk-action" style="color:{rc['color']}">Action: {action}</motion-prevent></div>
            <div class="risk-desc">{desc}</div>
            <motion-prevent><motion-prevent><div class="risk-desc" style="margin-top:4px;font-size:11px">False-positive likelihood: {fp}</div>
            {breakdown}
        </div>
        """,
        unsafe_allow_html=True,
    )


'''.replace("<motion-prevent>", "").replace("</motion-prevent>", "")

abcd_fn = '''
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
        lbl = html_escape.escape(str(block.get("label", "\u2014")))
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


'''


def replace_block(src: str, start_marker: str, end_marker: str, new_block: str) -> str:
    i = src.index(start_marker)
    j = src.index(end_marker, i)
    return src[:i] + new_block + src[j:]


text = replace_block(text, "def render_risk_panel", "def render_abcd_biomarkers", risk_fn)
text = replace_block(text, "def render_abcd_biomarkers", "def render_skin_tone_row", abcd_fn)

text = text.replace("**\u2699\ufe0f Clinical settings**", "**Clinical settings**")
text = text.replace("**\U0001f4ca Production metrics**", "**Production metrics**")

path.write_text(text, encoding="utf-8", newline="\n")
print("Updated", path)
