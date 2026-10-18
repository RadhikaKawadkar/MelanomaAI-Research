"""
MelanomaAI — Flask Backend
Connects to existing backend/model.py, backend/metrics.py, backend/pdf_report.py
Run: python app.py
"""

import os
import json
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
from datetime import datetime
app = Flask(__name__)

flask_secret = os.environ.get("MELANOMAAI_FLASK_SECRET_KEY")
if not flask_secret:
    raise RuntimeError(
        "MELANOMAAI_FLASK_SECRET_KEY environment variable is required but not set."
    )
app.secret_key = flask_secret

UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ── Try to import real backend, fall back to demo mode ──────────────────────
DEMO_MODE = False
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backend.model import predict
    from backend.metrics import load_metrics
    print("✅ Real backend loaded")
except ImportError:
    DEMO_MODE = True
    print("⚠️  Backend not found — running in DEMO mode")

# ── Demo fallback functions ──────────────────────────────────────────────────
def demo_predict(image_path):
    """Simulates backend/model.py predict() output"""
    import random, math
    random.seed(hash(image_path) % 999)
    prob = round(random.uniform(0.08, 0.94), 4)
    label = "Malignant" if prob >= 0.5 else "Benign"
    ita = round(random.uniform(-35, 68), 1)
    if   ita >  55: tone = "Very Light"
    elif ita >  41: tone = "Light"
    elif ita >  28: tone = "Tan"
    elif ita >  10: tone = "Brown"
    elif ita > -30: tone = "Dark"
    else:           tone = "Very Dark"
    clahe = ita < 28
    risk_score = int(prob * 100)
    if   prob < 0.25: risk = "LOW"
    elif prob < 0.50: risk = "MODERATE"
    elif prob < 0.75: risk = "HIGH"
    else:             risk = "CRITICAL"
    abcd = {
        "asymmetry": round(random.uniform(0.2, 0.9), 2),
        "border":    round(random.uniform(0.2, 0.9), 2),
        "color":     round(random.uniform(0.2, 0.9), 2),
        "diameter":  round(random.uniform(3.5, 16.0), 1),
    }
    return {
        "probability": prob,
        "label": label,
        "ita": ita,
        "tone": tone,
        "clahe": clahe,
        "risk": risk,
        "risk_score": risk_score,
        "abcd": abcd,
    }

def demo_metrics():
    return {
        "accuracy":    88.94,
        "auc":         0.9392,
        "sensitivity": 82.57,
        "specificity": 89.97,
        "precision":   86.60,
        "f1":          84.50,
        "threshold":   0.50,
        "confusion_matrix": [[341, 28], [37, 172]],
        "fairness": {
            "light_acc": 93.36,
            "dark_acc":  75.19,
            "gap":       0.1834,
        }
    }

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Main page — renders the SPA shell"""
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Accepts image upload, returns prediction JSON"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Only JPG and PNG allowed"}), 400

    filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Call real or demo predict
    if DEMO_MODE:
        result = demo_predict(filepath)
    else:
        result = predict(filepath)

    # Enrich result
    result["filename"]  = filename
    result["image_url"] = f"/static/uploads/{filename}"
    result["timestamp"] = datetime.now().strftime("%H:%M:%S")

    # Add to session history
    if "history" not in session:
        session["history"] = []
    history_entry = {
        "name":       file.filename,
        "verdict":    result["label"].upper(),
        "prob":       f"{result['probability']*100:.1f}%",
        "prob_raw":   result["probability"],
        "risk":       result["risk"],
        "score":      result["risk_score"],
        "tone":       result["tone"],
        "time":       result["timestamp"],
        "image_url":  result["image_url"],
    }
    history = session["history"]
    history.insert(0, history_entry)
    session["history"] = history[:50]  # keep last 50
    session.modified = True

    return jsonify(result)

@app.route("/api/metrics")
def api_metrics():
    """Returns model evaluation metrics"""
    if DEMO_MODE:
        return jsonify(demo_metrics())
    return jsonify(load_metrics())

@app.route("/api/history")
def api_history():
    """Returns session scan history"""
    return jsonify(session.get("history", []))

@app.route("/api/history/clear", methods=["POST"])
def api_history_clear():
    session["history"] = []
    session.modified = True
    return jsonify({"ok": True})

@app.route("/api/report", methods=["POST"])
def api_report():
    """Generate and return PDF report"""
    data = request.get_json()
    if DEMO_MODE:
        # Return a minimal demo PDF placeholder
        return jsonify({"error": "PDF generation requires real backend/pdf_report.py"}), 501
    try:
        from backend.pdf_report import generate_report
        pdf_bytes = generate_report(data)
        return send_file(
            BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name="melanoma_report.pdf"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n🔬 MelanomaAI Flask Server")
    print(f"   Mode: {'DEMO' if DEMO_MODE else 'LIVE'}")
    print("   URL:  http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
