# -*- coding: utf-8 -*-
"""
MelanomaAI v2 — PDF diagnostic report generator (reportlab).
Does not modify model, risk engine, or inference logic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "outputs" / "reports"

NAVY = colors.HexColor("#1A2B4A")
RED = colors.HexColor("#DC2626")
GREEN = colors.HexColor("#16A34A")
ORANGE = colors.HexColor("#EA580C")
MUTED = colors.HexColor("#64748B")
WHITE = colors.white
BLACK = colors.black

CLINICAL_ACTIONS = {
    "CRITICAL": "Immediate dermatologist referral",
    "HIGH": "Book appointment within 1 week",
    "MODERATE": "Schedule clinical review",
    "LOW": "Routine annual monitoring",
}


def _abcd_value_color(value: float) -> colors.Color:
    if value > 0.7:
        return RED
    if value >= 0.4:
        return ORANGE
    return GREEN


def _normalize_abcd(abcd: dict) -> dict:
    """Accept flat or nested ABCD dict from risk_engine.estimate_abcd."""
    if not abcd:
        return {
            "asymmetry": 0.0,
            "border": 0.0,
            "color": 0.0,
            "diameter": 0.0,
            "asymmetry_label": "—",
            "border_label": "—",
            "color_label": "—",
            "diameter_label": "—",
        }
    if "asymmetry" in abcd and isinstance(abcd["asymmetry"], dict):
        a = abcd.get("asymmetry", {})
        b = abcd.get("border", {})
        c = abcd.get("color", {})
        d = abcd.get("diameter", {})
        return {
            "asymmetry": float(a.get("score", 0)),
            "border": float(b.get("score", 0)),
            "color": float(c.get("score", 0)),
            "diameter": float(d.get("mm", 0)),
            "asymmetry_label": str(a.get("label", "—")),
            "border_label": str(b.get("label", "—")),
            "color_label": str(c.get("label", "—")),
            "diameter_label": str(d.get("label", "—")),
        }
    return {
        "asymmetry": float(abcd.get("asymmetry", 0)),
        "border": float(abcd.get("border", 0)),
        "color": float(abcd.get("color", 0)),
        "diameter": float(abcd.get("diameter", 0)),
        "asymmetry_label": str(abcd.get("asymmetry_label", "—")),
        "border_label": str(abcd.get("border_label", "—")),
        "color_label": str(abcd.get("color_label", "—")),
        "diameter_label": str(abcd.get("diameter_label", "—")),
    }


def _safe_case_id(case_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in (case_id or "MEL-UNKNOWN"))


def _resolve_existing_path(path: str | Path | None) -> str | None:
    """Return absolute path string if the file exists, else None."""
    if not path:
        return None
    p = Path(path)
    candidates = [p]
    if not p.is_absolute():
        candidates.append(ROOT / p)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if resolved.is_file():
                return str(resolved)
        except OSError:
            continue
    return None


def _resolve_gradcam_path(gradcam_path: str | Path | None, img_path: str | None) -> str | None:
    """Resolve Grad-CAM file from payload path or standard outputs/gradcam/{stem}_gradcam.*"""
    resolved = _resolve_existing_path(gradcam_path)
    if resolved:
        return resolved
    img = _resolve_existing_path(img_path)
    if not img:
        return None
    stem = Path(img).stem
    gdir = (ROOT / "outputs" / "gradcam").resolve()
    for name in (f"{stem}_gradcam.jpg", f"{stem}_gradcam.jpeg", f"{stem}_gradcam.png"):
        candidate = gdir / name
        if candidate.is_file():
            return str(candidate)
    sibling = Path(img).with_name(f"{stem}_gradcam.png")
    if sibling.is_file():
        return str(sibling.resolve())
    return None


def _load_image_flowable(path: str | None, max_size: float = 2.2 * inch) -> RLImage | Paragraph:
    path = _resolve_existing_path(path)
    if not path:
        return Paragraph(
            "<i>Image not available</i>",
            ParagraphStyle("img_missing", fontName="Helvetica", fontSize=9, textColor=MUTED),
        )
    try:
        from PIL import Image as PILImage

        pil = PILImage.open(path).convert("RGB")
        w, h = pil.size
        scale = min(max_size / w, max_size / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        if nw < 1 or nh < 1:
            nw, nh = 1, 1
        if (nw, nh) != (w, h):
            pil = pil.resize((nw, nh), PILImage.Resampling.LANCZOS)
        tmp = REPORTS_DIR / f"_tmp_{Path(path).stem}.jpg"
        pil.save(tmp, format="JPEG", quality=90)
        return RLImage(str(tmp), width=nw, height=nh)
    except Exception:
        return Paragraph(
            "<i>Image could not be loaded</i>",
            ParagraphStyle("img_err", fontName="Helvetica", fontSize=9, textColor=MUTED),
        )


def generate_report(result: dict) -> str:
    """
    Generates PDF diagnostic report.
    Returns: absolute path to saved PDF file.
    """
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        case_id = _safe_case_id(str(result.get("case_id", "MEL-UNKNOWN")))
        out_path = REPORTS_DIR / f"report_{case_id}.pdf"
        out_path = out_path.resolve()

        verdict = str(result.get("verdict", "BENIGN")).upper()
        prob = float(result.get("probability", 0))
        risk_level = str(result.get("risk_level", "MODERATE")).upper()
        risk_score = int(result.get("risk_score", 0))
        ita = float(result.get("ita", 0))
        clahe = bool(result.get("clahe", False))
        tone_name = str(result.get("tone_name", "Unknown"))
        age = int(result.get("age", 0))
        patient_name = str(result.get("patient_name") or "").strip() or "Not provided"
        patient_id = str(result.get("patient_id") or "").strip() or "Not provided"
        img_path = _resolve_existing_path(result.get("img_path"))
        gradcam_path = _resolve_gradcam_path(result.get("gradcam_path"), img_path)
        timestamp = str(result.get("timestamp", ""))
        threshold = float(result.get("threshold", 0.5))
        abcd = _normalize_abcd(result.get("abcd", {}))

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "Title",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=WHITE,
            alignment=TA_LEFT,
        )
        subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=12,
            textColor=colors.HexColor("#CBD5E1"),
        )
        h2_style = ParagraphStyle(
            "H2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=NAVY,
            spaceBefore=14,
            spaceAfter=8,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            textColor=BLACK,
            leading=14,
        )
        small_style = ParagraphStyle(
            "Small",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=MUTED,
            leading=11,
        )

        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=A4,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.6 * inch,
            bottomMargin=0.6 * inch,
        )
        story: list[Any] = []

        # SECTION 1 — HEADER
        header_table = Table(
            [
                [Paragraph("MelanomaAI", title_style)],
                [Paragraph("Clinical Diagnostic Report", subtitle_style)],
            ],
            colWidths=[6.5 * inch],
        )
        header_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                    ("LEFTPADDING", (0, 0), (-1, -1), 14),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                    ("TOPPADDING", (0, 0), (-1, -1), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )
        story.append(header_table)
        story.append(Spacer(1, 10))
        story.append(
            HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=10)
        )
        story.append(
            Paragraph(
                f"<b>Case ID:</b> {case_id} &nbsp;|&nbsp; <b>Timestamp:</b> {timestamp} "
                f"&nbsp;|&nbsp; <b>Model:</b> MelanomaAI v2.0",
                body_style,
            )
        )
        story.append(
            Paragraph(
                f"<b>Threshold:</b> {threshold:.2f} &nbsp;|&nbsp; "
                f"<b>Dataset:</b> HAM10000 + ISIC 2020 &nbsp;|&nbsp; "
                f"<b>Architecture:</b> EfficientNetB3",
                body_style,
            )
        )
        story.append(Spacer(1, 12))

        # SECTION 2 — PATIENT INFORMATION
        story.append(Paragraph("Patient Information", h2_style))
        patient_data = [
            ["Full Name", patient_name],
            ["Patient ID", patient_id],
            ["Age", f"{age} years"],
            ["Analysis Date", timestamp],
        ]
        pt = Table(patient_data, colWidths=[1.6 * inch, 4.4 * inch])
        pt.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(pt)
        story.append(Spacer(1, 12))

        # SECTION 3 — DIAGNOSTIC VERDICT
        story.append(Paragraph("Diagnostic Verdict", h2_style))
        v_color = RED if verdict == "MALIGNANT" else GREEN
        action = CLINICAL_ACTIONS.get(risk_level, CLINICAL_ACTIONS["MODERATE"])
        verdict_rows = [
            [Paragraph(f"<b>VERDICT: {verdict}</b>", ParagraphStyle(
                "v", fontName="Helvetica-Bold", fontSize=16, textColor=WHITE, alignment=TA_CENTER
            ))],
            [Paragraph(
                f"Probability: {prob * 100:.1f}% &nbsp;|&nbsp; "
                f"Risk Level: {risk_level} &nbsp;|&nbsp; Risk Score: {risk_score}/100",
                ParagraphStyle("v2", fontName="Helvetica", fontSize=11, textColor=WHITE, alignment=TA_CENTER),
            )],
            [Paragraph(f"Clinical Action: {action}", ParagraphStyle(
                "v3", fontName="Helvetica", fontSize=10, textColor=WHITE, alignment=TA_CENTER
            ))],
        ]
        vt = Table(verdict_rows, colWidths=[6.2 * inch])
        vt.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), v_color),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        story.append(vt)
        story.append(Spacer(1, 14))

        # SECTION 4 — IMAGES
        story.append(Paragraph("Lesion Images", h2_style))
        orig_flow = _load_image_flowable(img_path)
        if gradcam_path:
            gcam_flow = _load_image_flowable(gradcam_path)
        else:
            gcam_flow = Paragraph(
                "<i>Heatmap not available</i>",
                ParagraphStyle("hm", fontName="Helvetica", fontSize=10, textColor=MUTED, alignment=TA_CENTER),
            )
        cap_style = ParagraphStyle("cap", fontName="Helvetica", fontSize=8, textColor=MUTED, alignment=TA_CENTER)
        img_table = Table(
            [
                [orig_flow, gcam_flow],
                [Paragraph("Original lesion image", cap_style), Paragraph("Grad-CAM heatmap", cap_style)],
            ],
            colWidths=[3.1 * inch, 3.1 * inch],
        )
        img_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(img_table)
        story.append(Spacer(1, 14))

        # SECTION 5 — ABCD BIOMARKERS
        story.append(Paragraph("ABCD Biomarkers", h2_style))
        abcd_rows = [
            ["Feature", "Value", "Label"],
            [
                "A — Asymmetry",
                f"{abcd['asymmetry']:.3f}",
                abcd["asymmetry_label"],
            ],
            [
                "B — Border",
                f"{abcd['border']:.3f}",
                abcd["border_label"],
            ],
            [
                "C — Color",
                f"{abcd['color']:.3f}",
                abcd["color_label"],
            ],
            [
                "D — Diameter",
                f"{abcd['diameter']:.1f} mm",
                abcd["diameter_label"],
            ],
        ]
        abcd_table = Table(abcd_rows, colWidths=[1.8 * inch, 1.2 * inch, 2.7 * inch])
        abcd_style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
        for i, key in enumerate(["asymmetry", "border", "color"], start=1):
            col = _abcd_value_color(float(abcd[key]))
            abcd_style_cmds.append(("TEXTCOLOR", (1, i), (1, i), col))
        d_mm = float(abcd["diameter"])
        d_col = ORANGE if d_mm > 6 else GREEN
        abcd_style_cmds.append(("TEXTCOLOR", (1, 4), (1, 4), d_col))
        abcd_table.setStyle(TableStyle(abcd_style_cmds))
        story.append(abcd_table)
        story.append(Spacer(1, 14))

        # SECTION 6 — SKIN TONE & PREPROCESSING
        story.append(Paragraph("Skin Tone &amp; Preprocessing", h2_style))
        clahe_txt = "YES" if clahe else "NO"
        skin_lines = [
            f"<b>Skin Tone:</b> {tone_name} (ITA: {ita:.1f}°)",
            f"<b>CLAHE Applied:</b> {clahe_txt}",
        ]
        if clahe:
            skin_lines.append(
                "Enhanced preprocessing applied for improved dark skin reliability."
            )
        if ita <= 28:
            skin_lines.append(
                "<b>Note:</b> Prediction reliability may be reduced for this skin tone. "
                "Clinical review advised."
            )
        for line in skin_lines:
            story.append(Paragraph(line, body_style))
        story.append(Spacer(1, 12))

        # SECTION 7 — MODEL PERFORMANCE
        story.append(Paragraph("Model Performance", h2_style))
        perf_rows = [
            ["Accuracy", "92.80%", "AUC-ROC", "0.9770"],
            ["Sensitivity", "79.51%", "Specificity", "97.38%"],
            ["F1 Score", "0.8498", "Precision", "91.27%"],
        ]
        perf = Table(perf_rows, colWidths=[1.4 * inch, 1.6 * inch, 1.4 * inch, 1.6 * inch])
        perf.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(perf)
        story.append(Paragraph("Training: 20,000 images (HAM10000 + ISIC 2020)", body_style))
        story.append(
            Paragraph("Fairness: CLAHE reduces dark skin gap by 33.1%", body_style)
        )
        story.append(Spacer(1, 16))

        # SECTION 8 — DISCLAIMER
        story.append(
            HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0"), spaceBefore=8)
        )
        disclaimer = (
            "This report is generated by MelanomaAI v2.0, an AI-assisted clinical decision "
            "support system. It is NOT a substitute for professional medical diagnosis. "
            "Always consult a qualified dermatologist. For research and educational purposes only."
        )
        story.append(Paragraph(disclaimer, small_style))

        doc.build(story)
        return str(out_path)

    except Exception as exc:
        raise RuntimeError(f"PDF report generation failed: {exc}") from exc
