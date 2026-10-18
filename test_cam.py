import cv2
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.model import load_model, val_tf
from backend.gradcam import GradCAM, generate_heatmap_overlay

def main():
    model, p = load_model()
    img_path = "data/processed/benign/ISIC_0024322.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found:", img_path)
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = val_tf(image=img_resized)["image"].unsqueeze(0).to(device)

    cam_gen = GradCAM(model)
    cam = cam_gen.generate(t)
    
    print("CAM max:", cam.max(), "min:", cam.min())
    print("Zeros:", np.sum(cam == 0), "Total:", cam.size)
    print("CAM shape:", cam.shape)
    
    heatmap = np.uint8(255 * cam)
    print("Unique heatmap values:", np.unique(heatmap))

if __name__ == '__main__':
    main()
