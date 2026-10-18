# MelanomaAI — Project Status (Stabilization Freeze)

**Date:** 2026-05-17  
**Phase:** Stabilization / polish — **no training, no architecture changes, no new datasets**

---

## Final production metrics

| Metric | Value |
|--------|-------|
| Accuracy | 92.8% |
| AUC-ROC | 0.9770 |
| Sensitivity | 79.51% |
| Specificity | 97.38% |
| Precision | 91.27% |
| F1 Score | 0.8498 |

Source: `outputs/metrics.json` (4000-sample eval, 2026-05-16)

---

## Completed systems (frozen)

| Area | Status | Key paths |
|------|--------|-----------|
| Balanced dataset pipeline | Done | `balance_dataset.py`, `data/v2/balanced/` |
| GPU training (focal loss, 2-phase) | Done | `auto_train.py` — **do not modify** |
| EfficientNetB3 classifier | Done | `backend/model.py`, `demo.py` |
| Fairness / robustness | Done | `evaluate_robustness.py`, `risk_engine.py` |
| Grad-CAM backend | Done | `backend/gradcam.py` |
| Metrics export | Done | `outputs/metrics.json`, `outputs/*.png` |
| Streamlit multipage app | Done | `app.py` + `pages/01–04_*.py` + `stitch_shared.py` |
| Stitch UI assets | Exported | `stitch_ui/`, `stitch_ui.css` |
| Legacy `frontend/` | Deprecated | Reference only — see `frontend/README.md` |

---

## Verified project structure

```
EDI/
├── app.py                      # Multipage navigation hub (st.navigation)
├── pages/                      # Workspace · Performance · Fairness · History
├── stitch_shared.py            # Shared theme, sidebar, metrics, PDF helper
├── stitch_ui.css               # Stitch design tokens / glass panels
├── stitch_ui/                  # Exported HTML mockups + DESIGN.md
├── backend/
│   ├── model.py                # Load + predict + robustness routing
│   ├── gradcam.py              # Grad-CAM generation
│   ├── metrics.py              # JSON loaders
│   └── pdf_report.py           # ReportLab PDF (used by legacy frontend/)
├── frontend/                   # Legacy component modules (not wired to pages/)
├── models/
│   └── melanoma_final.pth      # Production weights (~45 MB)
├── outputs/                    # Metrics JSON, plots, Grad-CAM samples
├── data/                       # Raw + processed + v2 balanced (gitignored)
├── backups/stabilization_2026-05-17/  # Local backup (gitignored)
├── auto_train.py               # FROZEN
├── evaluate_model.py           # FROZEN
├── evaluate_robustness.py      # FROZEN
├── demo.py                     # CLI demo
├── risk_engine.py              # ITA, CLAHE, risk scoring
└── balance_dataset.py          # FROZEN
```

---

## Backups

Local snapshot: `backups/stabilization_2026-05-17/`

- Production checkpoint copy: `melanoma_final.pth`
- Metrics JSON copies + `MANIFEST.md` (SHA-256 checksum)

**Action:** Copy `backups/stabilization_2026-05-17/` to external storage.

---

## How to run (unchanged)

```bash
# Primary UI
streamlit run app.py --server.port 8502

# CLI single-image demo
python demo.py path/to/image.jpg

# Re-run evaluation plots (read-only; does not train)
python evaluate_model.py --threshold 0.50
python evaluate_robustness.py
```

---

## Cleanup performed (2026-05-17)

- Removed shell/agent debug text files (`*_out.txt`, `push_*.txt`, `status*.txt`, etc.)
- Removed `outputs/*_temp.txt` and demo scratch files
- Removed one-off `package_check.py`
- Tidied imports in `pages/02_Model_Performance_Analytics.py`
- Fixed deprecated `st.experimental_rerun()` → `st.rerun()` in history page
- Extended `.gitignore` for debug artifacts and `backups/`

**Preserved:** `training.log`, `balance_log.txt`, `outputs/final_*.log`, all model weights, all training scripts.

---

## Remaining frontend tasks only

These are **UI/polish** items. Backend and model are complete.

1. **Unify navigation** — Decide single entry: `app.py` monolith vs `pages/` multipage; avoid duplicate UIs.
2. **Wire Stitch design** — Apply `stitch_ui.css` + DESIGN.md tokens consistently across all pages (today: `app.py` uses inline CSS; pages use partial Stitch).
3. **Diagnostic Workspace (`pages/01`)** — Risk tier cards, ABCD grid, skin-tone badge, optimal threshold from `outputs/optimal_threshold.json`, append results to `st.session_state.history`.
4. **Performance Analytics (`pages/02`)** — Embed `outputs/confusion_matrix.png`, `roc_curve.png`, `training_curves.png`; formatted KPI grid (not raw JSON).
5. **Session History (`pages/03`)** — Persist history across pages; filters/export; match Stitch table layout.
6. **Fairness page (`pages/04`)** — Charts for skin-tone subgroups (from `robustness_report.json` / `fairness_metrics.json`), not JSON dump.
7. **Grad-CAM in pages** — Ensure `generate_gradcam` paths work from multipage context (already works in `app.py`).
8. **PDF reports** — Consolidate `stitch_shared.make_pdf_report` vs `backend/pdf_report.py` into one path.
9. **Legacy `frontend/components/`** — Either migrate into `pages/` or delete after migration.
10. **Stitch HTML → Streamlit** — Port layout sections from `stitch_ui/.../code.html` into reusable `st.components` or markdown blocks.
11. **README** — Update stale metrics table (still shows ~88% run); point to this file.
12. **`requirements.txt`** — Generate pinned list from `venv` for reproducible deploys.

---

## Roadmap — next session

| Priority | Task | Est. |
|----------|------|------|
| P0 | External backup of `backups/stabilization_2026-05-17/` | 5 min |
| P1 | Choose `app.py` vs `pages/` as canonical UI | 30 min |
| P1 | Wire optimal threshold + history append in Workspace | 1–2 hr |
| P2 | Performance page charts from `outputs/` | 1 hr |
| P2 | Fairness visualizations (bar/line by skin tone) | 2 hr |
| P3 | Stitch CSS parity + sidebar/topbar from mockups | 2–3 hr |
| P3 | `requirements.txt` + README metrics sync | 30 min |

**Explicitly out of scope until frontend phase ends:**

- Retraining, hyperparameter sweeps, new datasets
- Architecture changes (EfficientNetB3 head, focal loss params)
- Changes to `auto_train.py`, `balance_dataset.py`, or `backend/model.py` structure

---

## Git note

Model weights and `data/` are gitignored. Commit code and `outputs/*.png`; store `.pth` in release assets or cloud backup.
