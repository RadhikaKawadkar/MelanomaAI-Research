#!/usr/bin/env python3
"""Create final release backup snapshot (excludes venv, raw data, git)."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAMP = datetime.now().strftime("%Y-%m-%d")
DEST = ROOT / "backups" / f"final_release_{STAMP}"

INCLUDE_DIRS = ["app.py", "pages", "backend", "models", "outputs", "demo_samples", "demo_assets", "stitch_shared.py", "stitch_ui.css", "risk_engine.py", "demo.py", "scripts", "README.md", "PROJECT_STATUS.md", "VIVA_NOTES.md", "DEMO_GUIDE.md", "requirements.txt", ".streamlit"]
INCLUDE_FILES = ["evaluate_model.py", "evaluate_robustness.py"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def main() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    manifest = {"created": datetime.now().isoformat(), "files": []}

    def copy_item(name: str) -> None:
        src = ROOT / name
        if not src.exists():
            return
        dst = DEST / name
        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"),
            )
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        for fp in dst.rglob("*"):
            if fp.is_file() and fp.suffix in {".pth", ".json", ".png", ".jpg", ".py", ".md", ".css", ".toml", ".txt"}:
                rel = fp.relative_to(DEST).as_posix()
                entry = {"path": rel, "bytes": fp.stat().st_size}
                if fp.suffix == ".pth":
                    entry["sha256"] = sha256_file(fp)
                manifest["files"].append(entry)

    for item in INCLUDE_DIRS + INCLUDE_FILES:
        copy_item(item)

    (DEST / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Backup created: {DEST}")
    print(f"Files tracked: {len(manifest['files'])}")


if __name__ == "__main__":
    main()
