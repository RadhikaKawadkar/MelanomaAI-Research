# MelanomaAI — Skin Cancer Detection System

Production-ready AI clinical decision support for dermoscopic melanoma screening.

## Final results (frozen)

| Metric | Value |
|--------|-------|
| Accuracy | 92.8% |
| AUC-ROC | 0.9770 |
| Sensitivity | 79.51% |
| Specificity | 97.38% |
| Precision | 91.27% |
| F1 Score | 0.8498 |

## Quick start (demo / viva)

```bash
cd C:\EDI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts\validate_app.py
streamlit run app.py --server.port 8502
```

Open **http://localhost:8502**

- **Viva script:** [DEMO_GUIDE.md](DEMO_GUIDE.md)
- **Technical Q&A:** [VIVA_NOTES.md](VIVA_NOTES.md)
- **Project status:** [PROJECT_STATUS.md](PROJECT_STATUS.md)

## Streamlit app (multipage)

| Page | Purpose |
|------|---------|
| Workspace | Upload / quick-load demo → diagnose → Grad-CAM |
| Model Performance | KPIs, ROC, confusion matrix, v1→v2 |
| Fairness & Robustness | Skin-tone equity, CLAHE ablation |
| Session History | Session log, CSV/PDF export |

### Demo quick-load

Pre-curated images in `demo_samples/` — use the **Benign / Malignant / Dark skin** buttons on Workspace.

## CLI tools (optional)

```bash
python demo.py demo_samples/benign_ISIC_0024322.jpg
python evaluate_model.py --threshold 0.50
python evaluate_robustness.py
```

Training is **frozen** — do not run `auto_train.py` unless starting a new research cycle.

## Requirements

- Python 3.10+
- ~6 GB disk (model + venv)
- NVIDIA GPU optional (CPU supported)

```
pip install -r requirements.txt
```

## Project structure

```
EDI/
├── app.py                 # Multipage entry (st.navigation)
├── pages/                 # Workspace, Performance, Fairness, History
├── stitch_shared.py       # Shared UI + demo helpers
├── stitch_ui.css          # Design system
├── demo_samples/          # Viva quick-load images
├── demo_assets/           # PPT-ready charts & Grad-CAM
├── backend/               # Model + Grad-CAM (frozen)
├── risk_engine.py         # ITA, CLAHE, risk scoring (frozen)
├── models/melanoma_final.pth
├── outputs/               # metrics.json, charts
├── scripts/               # validate_app.py, create_final_backup.py
├── VIVA_NOTES.md
├── DEMO_GUIDE.md
└── backups/               # Release snapshots
```

## Backup

```bash
python scripts/create_final_backup.py
```

Creates `backups/final_release_YYYY-MM-DD/` with app, pages, model, outputs, and manifest.

## GitHub

https://github.com/satyamrahane/Melanoma-CNN-Detection
