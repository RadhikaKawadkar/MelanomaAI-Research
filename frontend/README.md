# Frontend UI

The primary production entry point for the MelanomaAI research interface is the multipage Streamlit dashboard:
```bash
streamlit run app.py
```

### Optional Flask Backend (`frontend/app.py`)
The Flask web interface connects to the research inference backend (`backend/model.py`, `backend/metrics.py`, `backend/report_generator.py`).

To run the Flask application, set the `MELANOMAAI_FLASK_SECRET_KEY` environment variable:
```bash
# Linux/macOS
export MELANOMAAI_FLASK_SECRET_KEY="<generate-random-secret>"
python frontend/app.py

# Windows PowerShell
$env:MELANOMAAI_FLASK_SECRET_KEY="<generate-random-secret>"
python frontend/app.py
```
