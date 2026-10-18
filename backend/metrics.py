import json
import os

def get_threshold():
    for p in ["outputs/optimal_threshold.json", "optimal_threshold.json"]:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f).get("optimal_threshold", 0.48)
    return 0.48

def load_metrics():
    for p in ["outputs/metrics.json", "metrics.json"]:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None

def load_risk_analysis():
    for p in ["outputs/risk_analysis.json", "risk_analysis.json"]:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None

def graph_path(filename):
    return os.path.join("outputs", "graphs", filename)
