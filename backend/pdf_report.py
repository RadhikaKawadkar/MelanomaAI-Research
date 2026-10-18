from datetime import datetime

def generate_report(result, metrics, threshold):
    now = datetime.now().strftime("%d %B %Y, %H:%M")
    prob  = result["prob"]
    label = result["label"]
    risk  = result["risk"]
    abcd  = result["abcd"]
    lines = [
        "=" * 60,
        "        MelanomaAI DIAGNOSTIC REPORT",
        "=" * 60,
        f"Generated:    {now}",
        f"Model:        EfficientNetB3 (HAM10000)",
        f"Threshold:    {threshold:.2f}",
        "",
        "── DIAGNOSIS ──────────────────────────────────────────",
        f"Result:       {label}",
        f"Probability:  {prob*100:.1f}%",
        f"Risk Level:   {risk['level']}",
        f"Risk Score:   {risk['score']}/100",
        f"Action:       {risk['action']}",
        "",
        "── SKIN TONE ANALYSIS ─────────────────────────────────",
        f"Tone:         {result['tone_lbl']}",
        f"ITA Score:    {result['ita']:.1f} degrees",
        f"CLAHE:        {'Applied' if result['clahe'] else 'Not applied'}",
        f"Reliability:  {result['reliability']*100:.0f}%",
        "",
        "── ABCD ANALYSIS ──────────────────────────────────────",
        f"Asymmetry:    {abcd['asymmetry']['label']} ({abcd['asymmetry']['score']:.3f})",
        f"Border:       {abcd['border']['label']} ({abcd['border']['score']:.3f})",
        f"Color:        {abcd['color']['label']} ({abcd['color']['score']:.3f})",
        f"Diameter:     {abcd['diameter']['mm']}mm — {abcd['diameter']['label']}",
        "",
        "── RISK SCORE BREAKDOWN ───────────────────────────────",
        f"Base prob:    {risk['factors']['base_probability']:.1f} pts",
        f"Confidence:   {risk['factors']['confidence']:.1f} pts",
        f"Skin tone:    {risk['factors']['skin_reliability']:.1f} pts",
        "",
    ]
    if metrics:
        lines += [
            "── MODEL PERFORMANCE ──────────────────────────────────",
            f"Accuracy:     {metrics.get('accuracy',0)*100:.2f}%",
            f"AUC-ROC:      {metrics.get('auc_roc',0):.4f}",
            f"Sensitivity:  {metrics.get('sensitivity',0)*100:.2f}%",
            f"Specificity:  {metrics.get('specificity',0)*100:.2f}%",
            "",
        ]
    lines += [
        "── DISCLAIMER ─────────────────────────────────────────",
        "This report is AI-generated for clinical decision support",
        "only. It does not replace professional medical diagnosis.",
        "Consult a qualified dermatologist for clinical evaluation.",
        "=" * 60,
    ]
    return "\n".join(lines)
