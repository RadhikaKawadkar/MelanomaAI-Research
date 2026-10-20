import json
import os

def get_threshold():
    for p in ["outputs/optimal_threshold.json", "optimal_threshold.json"]:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f).get("optimal_threshold", 0.48)
    return 0.48

def load_metrics():
    clean_paths = [
        os.path.join("outputs", "clean_retraining", "test_metrics_summary.json"),
        os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "outputs", "clean_retraining", "test_metrics_summary.json")),
    ]
    for p in clean_paths:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    data = json.load(f)
                roc_auc = data.get("roc_auc")
                op_raw = data.get("operating_points_raw", [])
                op_50 = next((op for op in op_raw if abs(op.get("threshold", 0.0) - 0.5) < 1e-3), None)
                sensitivity = op_50.get("sensitivity") if op_50 else None
                specificity = op_50.get("specificity") if op_50 else None
                precision = op_50.get("ppv") if op_50 else None
                f1_score = op_50.get("f1") if op_50 else None
                n_total = data.get("n_test_images", 0)
                accuracy = None
                if op_50 and n_total > 0:
                    accuracy = round((op_50.get("tp", 0) + op_50.get("tn", 0)) / n_total, 4)

                cm_data = None
                if op_50:
                    cm_data = [
                        [int(op_50.get("tn", 0)), int(op_50.get("fp", 0))],
                        [int(op_50.get("fn", 0)), int(op_50.get("tp", 0))],
                    ]

                return {
                    "accuracy": accuracy,
                    "auc_roc": roc_auc,
                    "roc_auc": roc_auc,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "precision": precision,
                    "f1_score": f1_score,
                    "confusion_matrix": cm_data,
                    "brier_score": data.get("brier_score", {}).get("raw"),
                    "ece": data.get("ece", {}).get("raw"),
                    "evaluation_dataset": data.get("evaluation_dataset", "ISIC_2020"),
                    "checkpoint": data.get("checkpoint"),
                    "source": p,
                }
            except Exception:
                pass

    for p in ["outputs/metrics.json", "metrics.json"]:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                pass
    return None

def load_risk_analysis():
    for p in ["outputs/risk_analysis.json", "risk_analysis.json"]:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None

def graph_path(filename):
    return os.path.join("outputs", "graphs", filename)
