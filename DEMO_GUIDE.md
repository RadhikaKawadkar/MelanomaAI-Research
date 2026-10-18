# MelanomaAI — Live Demo & Viva Guide

## Prerequisites

```bash
cd C:\EDI
.\venv\Scripts\activate
pip install -r requirements.txt
python scripts/validate_app.py
streamlit run app.py --server.port 8502
```

Open: **http://localhost:8502**

## Pre-demo checklist (5 min)

- [ ] Run `python scripts/validate_app.py` — all checks pass
- [ ] `models/melanoma_final.pth` present (~45 MB)
- [ ] `demo_samples/` contains 3 images
- [ ] Close heavy GPU apps if using CUDA
- [ ] Browser zoom 100%, wide window

## Recommended demo flow (examiner)

### Step 1 — Workspace · Benign (45 sec)

1. Click **🟢 Benign** quick-load
2. Click **Run Diagnostic Analysis**
3. Point out: BENIGN verdict, low confidence %, green risk bar, ABCD scores
4. Scroll to Grad-CAM — attention on lesion area

### Step 2 — Workspace · Malignant (60 sec)

1. Click **🔴 Malignant** quick-load → Analyze
2. Point out: MALIGNANT verdict, higher probability, risk tier, CLAHE badge if shown
3. Grad-CAM red regions on lesion

### Step 3 — Workspace · Dark skin / fairness (30 sec)

1. Click **🟤 Dark skin** quick-load → Analyze
2. Highlight **CLAHE enhanced** — inference-time fairness without retraining

### Step 4 — Model Performance (45 sec)

Sidebar → **Model Performance**

- KPI strip (92.8% accuracy, 0.977 AUC)
- ROC curve, confusion matrix, training curves
- v1 → v2 improvement table

### Step 5 — Fairness & Robustness (45 sec)

- Skin-tone accuracy bars from `robustness_report.json`
- CLAHE ablation (91.31% / 81.88% / 84.14%)
- Risk stratification distribution

### Step 6 — Session History (20 sec)

- Show logged cases from demo
- Optional: Export CSV

## Presentation assets

Pre-exported screenshots for PPT: `demo_assets/presentation/`

| File | Use in slides |
|------|----------------|
| `roc_curve.png` | Model discrimination |
| `confusion_matrix.png` | Error analysis |
| `training_curves.png` | Training convergence |
| `ISIC_*_gradcam.jpg` | Explainability examples |

Take fresh screenshots from the running app for dashboard slides (Streamlit UI).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Model not found | Ensure `models/melanoma_final.pth` exists |
| Grad-CAM slow on CPU | Wait 10–30 s; mention GPU speeds this up |
| Charts missing | Run `python evaluate_model.py` once (read-only eval) |
| Port in use | `streamlit run app.py --server.port 8503` |

## Backup

Final release snapshot:

```bash
python scripts/create_final_backup.py
```

Output: `backups/final_release_YYYY-MM-DD/`
