#!/usr/bin/env python3
"""Smoke-test MelanomaAI app dependencies, assets, and inference pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  OK  {msg}")
    else:
        print(f"  FAIL {msg}")
        FAILURES.append(msg)


def main() -> int:
    print("MelanomaAI — validation\n" + "=" * 40)

    print("\n[1] Core imports")
    try:
        import streamlit  # noqa: F401
        import torch  # noqa: F401
        import cv2  # noqa: F401
        import albumentations  # noqa: F401
        from PIL import Image  # noqa: F401

        check(True, "Python dependencies import")
    except Exception as e:
        check(False, f"Imports: {e}")

    print("\n[2] Application modules")
    try:
        import stitch_shared as shared  # noqa: F401
        import app  # noqa: F401

        check(True, "app.py + stitch_shared.py")
    except Exception as e:
        check(False, f"App modules: {e}")

    print("\n[3] Page modules")
    for page in [
        "pages/01_workspace.py",
        "pages/02_model_performance.py",
        "pages/03_fairness_robustness.py",
        "pages/04_session_history.py",
    ]:
        p = ROOT / page
        check(p.exists(), f"{page} exists")
        try:
            compile(p.read_text(encoding="utf-8"), str(p), "exec")
            check(True, f"{page} syntax")
        except Exception as e:
            check(False, f"{page} syntax: {e}")

    print("\n[4] Production assets")
    check((ROOT / "models/melanoma_final.pth").exists(), "models/melanoma_final.pth")
    for chart in ["roc_curve.png", "confusion_matrix.png", "training_curves.png"]:
        check((ROOT / "outputs" / chart).exists(), f"outputs/{chart}")
    for js in ["metrics.json", "robustness_report.json", "fairness_metrics.json"]:
        check((ROOT / "outputs" / js).exists(), f"outputs/{js}")
    check((ROOT / "stitch_ui.css").exists(), "stitch_ui.css")

    print("\n[5] Demo samples")
    manifest = ROOT / "demo_samples/manifest.json"
    check(manifest.exists(), "demo_samples/manifest.json")
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for key, meta in data.get("samples", {}).items():
            fp = ROOT / "demo_samples" / meta["file"]
            check(fp.exists(), f"demo_samples/{meta['file']} ({key})")

    print("\n[6] Inference + Grad-CAM (one benign demo)")
    try:
        from backend.model import load_model, predict_image
        from backend.gradcam import generate_gradcam

        model, path = load_model()
        check(model is not None, f"Model loaded from {path}")
        demo = ROOT / "demo_samples/benign_ISIC_0024322.jpg"
        if demo.exists():
            r = predict_image(str(demo), threshold=0.5)
            check("probability" in r, f"predict_image prob={r.get('probability', 0):.3f}")
            out = generate_gradcam(str(demo), output_path=str(ROOT / "outputs/gradcam/_validate_gradcam.jpg"))
            check(out and Path(out).exists(), "generate_gradcam output")
    except Exception as e:
        check(False, f"Inference pipeline: {e}")

    print("\n[7] Metrics sanity")
    metrics_path = ROOT / "outputs/metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        check(m.get("accuracy", 0) > 0.9, f"accuracy={m.get('accuracy')}")
        check(m.get("auc_roc", 0) > 0.95, f"auc_roc={m.get('auc_roc')}")

    print("\n" + "=" * 40)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)} issues):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED — ready for demo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
