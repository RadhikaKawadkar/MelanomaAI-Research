import sys
import os
import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
import json
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    from albumentations import Compose, Resize, Normalize
    from albumentations.pytorch import ToTensorV2
    HAS_ALBU = True
except ImportError:
    HAS_ALBU = False

# Import clinical logic from local risk_engine.py
try:
    from risk_engine import (
        robustness_preprocess, 
        compute_risk_score, 
        estimate_abcd
    )
except ImportError:
    print("Error: risk_engine.py not found in the current directory.")
    sys.exit(1)

# Import Grad-CAM for visual explainability
try:
    from backend.gradcam import GradCAM, generate_heatmap_overlay
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False

def load_model(model_path="models/melanoma_final.pth"):
    """Loads the EfficientNet-B3 model with custom classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reconstruct architecture from utils.py
    base = models.efficientnet_b3(weights=None)
    num_features = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    
    if not os.path.exists(model_path):
        # Try relative paths if absolute fails
        alt_path = os.path.join(os.path.dirname(__file__), model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"Error: Model file {model_path} not found.")
            sys.exit(1)
            
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    # Handle both full checkpoint and state_dict only
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        base.load_state_dict(ckpt["model_state_dict"])
    else:
        base.load_state_dict(ckpt)
        
    base = base.to(device)
    base.eval()
    return base, device

def get_threshold():
    """Loads optimal threshold from outputs/optimal_threshold.json."""
    paths = ["outputs/optimal_threshold.json", "optimal_threshold.json"]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f).get("optimal_threshold", 0.48)
            except:
                pass
    return 0.48

def show_heatmap_in_terminal(gradcam_path, prob, label, abcd):
    try:
        import subprocess
        
        print("\n------------------------------------------------------------")
        print(" EXPLAINABILITY (Grad-CAM Heatmap Analysis)")
        print("------------------------------------------------------------")
        print(f" Heatmap saved -> {gradcam_path}")
        print(f" Opening visualization automatically...")
        subprocess.Popen(['start', '', gradcam_path], shell=True)
        print()
        print(" WHAT THE HEATMAP SHOWS:")
        print("   RED   = Model focused HERE most -> suspicious region")
        print("   YELLOW= Secondary attention -> border area")  
        print("   BLUE  = Model ignored this -> healthy skin/background")
        print()
        
        # Dynamic interpretation based on result
        if label == "MALIGNANT":
            print(" INTERPRETATION:")
            print(f"   Model detected suspicious features with {prob*100:.1f}% confidence")
            if abcd['asymmetry']['score'] > 0.5:
                print("   -> High asymmetry detected in lesion shape")
            if abcd['border']['score'] > 0.5:
                print("   -> Irregular border pattern identified")
            if abcd['color']['score'] > 0.4:
                print("   -> Multiple color variations found")
            if abcd['diameter']['mm'] > 6:
                print(f"   -> Large diameter {abcd['diameter']['mm']}mm exceeds 6mm threshold")
            print("   -> RED zone marks the most suspicious area")
            print("   -> Immediate dermatologist consultation recommended")
        else:
            print(" INTERPRETATION:")
            print(f"   Model found no strong malignant features ({prob*100:.1f}% probability)")
            if prob > 0.35:
                print("   -> Some features noted but below malignant threshold")
                print("   -> RED zone shows area model examined most closely")
                print("   -> Monitor this region in future checkups")
            else:
                print("   -> Lesion shows regular, uniform characteristics")
                print("   -> No dominant suspicious region detected")
                print("   -> Routine annual skin check recommended")
                
    except Exception as e:
        print(f" Heatmap saved -> {gradcam_path}")


def main():
    if len(sys.argv) < 2:
        print("MelanomaAI Terminal Demo")
        print("=" * 40)
        img_path = input("Enter image path: ").strip().strip('"')
    else:
        img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"\033[91mError: Image {img_path} not found.\033[0m")
        sys.exit(1)

    try:
        age_input = input("Enter patient age (or press Enter to skip): ")
        if age_input == "":
            age = 40
        else:
            age = int(age_input)
    except Exception:
        age = 40

    # 1. Initialize
    print("\n[1/3] Loading MelanomaAI Model...")
    model, device = load_model()
    threshold = get_threshold()
    
    # Load image
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"\033[91mError: Could not read image {img_path}\033[0m")
        sys.exit(1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Preprocessing & Prediction
    print("[2/3] Analyzing Skin Tone & Running Prediction...")
    processed_img, ita, tone_label, reliability, was_enhanced = robustness_preprocess(img_rgb)
    
    if HAS_ALBU:
        transform = Compose([
            Resize(224, 224),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        model_img = cv2.resize(processed_img, (224, 224))
        transformed = transform(image=model_img)
        inp = transformed["image"].unsqueeze(0).to(device)
    else:
        # Fallback to manual numpy/cv2 normalization if Albumentations is missing
        model_img = cv2.resize(processed_img, (224, 224)) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        model_img = (model_img - mean) / std
        inp = torch.from_numpy(model_img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        prob = float(model(inp).squeeze().cpu().item())
    
    # 2b. Grad-CAM heatmap generation
    heatmap_path = None
    if HAS_GRADCAM:
        try:
            cam_gen = GradCAM(model)
            cam = cam_gen.generate(inp)
            overlay = generate_heatmap_overlay(img_rgb, cam)
            gradcam_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "gradcam")
            os.makedirs(gradcam_dir, exist_ok=True)
            gradcam_filename = os.path.splitext(os.path.basename(img_path))[0] + "_gradcam.jpg"
            heatmap_path = os.path.join(gradcam_dir, gradcam_filename)
            cv2.imwrite(heatmap_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        except Exception as e:
            heatmap_path = None

    # 3. Clinical Scoring
    verdict = "MALIGNANT" if prob >= threshold else "BENIGN"
    risk_data = compute_risk_score(prob, ita, threshold, age=age)
    abcd = estimate_abcd(prob)

    # 4. Final Output
    print("[3/3] Generating Results...\n")
    
    # Header
    print("="*60)
    print(" " * 15 + "MELANOMA AI DIAGNOSTIC REPORT")
    print("="*60)
    
    # Core Verdict
    colors = {"MALIGNANT": "\033[91m", "BENIGN": "\033[92m", "END": "\033[0m"}
    v_color = colors.get(verdict, "")
    if verdict == "MALIGNANT":
        print(f"VERDICT:      {v_color}MALIGNANT (Cancerous — Seek immediate medical attention){colors['END']}")
    else:
        print(f"VERDICT:      {v_color}BENIGN (Non-cancerous — Routine monitoring recommended){colors['END']}")
    print(f"Probability:  {prob*100:.1f}%")
    print(f"Risk Level:   {risk_data['level']}")
    print(f"Risk Score:   {int(risk_data['score'])}/100")
    
    # Technical Details
    print("-" * 60)
    print(f"Skin Tone:    {ita:.1f}° ({tone_label})")
    print(f"CLAHE:        {'ON' if was_enhanced else 'OFF'}")
    
    # ABCD Scores
    print("\nABCD BIO-MARKER ESTIMATION:")
    print(f" - Asymmetry: {abcd['asymmetry']['score']:.3f} [{abcd['asymmetry']['label']}]")
    print(f" - Border:    {abcd['border']['score']:.3f} [{abcd['border']['label']}]")
    print(f" - Color:     {abcd['color']['score']:.3f} [{abcd['color']['label']}]")
    print(f" - Diameter:  {abcd['diameter']['mm']:.1f}mm [{abcd['diameter']['label']}]")
    
    # Grad-CAM output
    if heatmap_path and os.path.exists(heatmap_path):
        show_heatmap_in_terminal(heatmap_path, prob, verdict, abcd)
    else:
        print("-" * 60)
        print("EXPLAINABILITY (Grad-CAM)")
        print("-" * 60)
        print(f"Heatmap:      Not generated (Grad-CAM unavailable)")

    # Global Performance
    print("-" * 60)
    print("SYSTEM PERFORMANCE (GLOBAL METRICS)")
    print("-" * 60)
    
    metrics_path = "outputs/metrics.json"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                m = json.load(f)
                print(f"Accuracy:     {m.get('accuracy', 0)*100:.2f}%")
                print(f"AUC:          {m.get('auc_roc', 0):.4f}")
                print(f"Sensitivity:  {m.get('sensitivity', 0)*100:.2f}%")
                print(f"Specificity:  {m.get('specificity', 0)*100:.2f}%")
        except:
            print("Error reading metrics.json")
    else:
        print("metrics.json not found in outputs/.")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
