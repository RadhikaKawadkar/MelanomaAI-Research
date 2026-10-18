from pathlib import Path

p = Path(__file__).resolve().parents[1] / "stitch_shared.py"
text = p.read_text(encoding="utf-8")
start = text.index("def render_abcd_biomarkers")
end = text.index("\n\ndef render_skin_tone_row")

new_fn = """
def render_abcd_biomarkers(abcd: dict) -> None:
    import html as html_escape

    st.markdown(
        '<p style="font-size:13px;font-weight:600;margin:12px 0 10px">'
        "🔬 ABCD Rule Biomarkers</p>",
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
            '<motion-prevent><motion-prevent><motion-prevent><motion-prevent><motion-prevent><motion-prevent><motion-prevent><motion-prevent><motion-prevent><motion-prevent><motion-prevent><motion-prevent><motion-prevent><motion-prevent><div class="bio-card">'
            + f'<div class="bio-letter">{letter}</div>'
            + f'<div class="bio-name">{name_esc}</div>'
            + '<div class="bio-bar-bg">'
            + f'<div class="bio-bar-fill" style="width:{pct:.0f}%;background:{color}"></div>'
            + "</div>"
            + f'<div class="bio-val" style="color:{color}">{val_str}</div>'
            + f'<motion-prevent><motion-prevent><motion-prevent><motion-prevent><motion-prevent><motion-prevent><motion-prevent><motion-prevent><motion-prevent><motion-prevent><motion-prevent><motion-prevent><motion-prevent><motion-prevent><div class="bio-lbl">{lbl}</motion-prevent></div>'
            + "</div>"
        )
        col.markdown(card, unsafe_allow_html=True)

"""

new_fn = new_fn.replace("<motion-prevent>", "").replace("</motion-prevent>", "")

p.write_text(text[:start] + new_fn + text[end:], encoding="utf-8")
print("patched render_abcd_biomarkers")
