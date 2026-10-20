"""
Grad-CAM — Gradient-weighted Class Activation Mapping
=====================================================
Generates visual heatmaps showing which regions of the dermoscopic image
the EfficientNet-B3 model focused on to make its prediction.

This is the EXPLAINABILITY component of MelanomaAI.
Judges and clinicians can see *where* the model is looking —
confirming it focuses on lesion boundaries, asymmetry, and color variation.

Reference: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (ICCV 2017)
"""

import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class GradCAM:
    """Grad-CAM for EfficientNet-B3 (targets last convolutional block)."""

    def __init__(self, model, target_layer=None):
        self.model = model
        # Default: last conv block of EfficientNet
        self.target_layer = target_layer or model.features[-1]
        self.gradients = None
        self.activations = None
        self._handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def remove_hooks(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: preprocessed image tensor [1, 3, H, W]

        Returns:
            cam: numpy array [H, W] in range [0, 1]
        """
        try:
            self.model.eval()
            input_tensor = input_tensor.clone().detach().requires_grad_(True)
            self.model.zero_grad()

            output = self.model(input_tensor)
            
            # To avoid vanishing gradients from Sigmoid when prob is near 0 or 1,
            # we compute Grad-CAM with respect to the pre-sigmoid logit.
            p_val = torch.clamp(output, 1e-7, 1.0 - 1e-7)
            logit = torch.log(p_val / (1.0 - p_val))
            
            # If the model predicts benign (p < 0.5), we want the gradients for the benign class
            if p_val.item() < 0.5:
                logit = -logit
                
            logit.backward()

            gradients = self.gradients.cpu().numpy()[0]     # [C, h, w]
            activations = self.activations.cpu().numpy()[0] # [C, h, w]

            # Global average pool gradients → channel weights
            weights = np.mean(gradients, axis=(1, 2))       # [C]

            # Weighted combination of activation maps
            cam = np.zeros(activations.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * activations[i]

            # ReLU — keep only positive influence
            cam = np.maximum(cam, 0)

            # Normalize to [0, 1]
            if cam.max() > 0:
                cam = cam / cam.max()

            # Resize to input image dimensions
            cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
            return cam
        finally:
            self.remove_hooks()


def generate_heatmap_overlay(
    original_img: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        original_img: RGB image [H, W, 3] uint8
        cam: Grad-CAM output [H, W] float32 [0,1]
        alpha: transparency of heatmap overlay
        colormap: OpenCV colormap constant

    Returns:
        overlay: RGB image [H, W, 3] uint8
    """
    h, w = original_img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Convert to uint8 heatmap
    heatmap = np.uint8(255 * cam_resized)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = np.uint8(original_img * (1 - alpha) + heatmap_rgb * alpha)
    return overlay


def generate_gradcam(image_path, output_path=None, alpha=0.45, use_robustness=True, model=None):
    """Generate and save a Grad-CAM overlay image for a single dermoscopy image."""
    from pathlib import Path

    if isinstance(image_path, np.ndarray):
        image = image_path.copy()
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        path_str = str(image_path)
        if not os.path.exists(path_str):
            return None
        image = cv2.imread(path_str)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    from backend.model import load_model, IMG_SIZE, val_tf
    from risk_engine import robustness_preprocess

    if model is None:
        model, _ = load_model()
    if model is None:
        raise FileNotFoundError('Model checkpoint not found. Ensure models/melanoma_final.pth or clean_model_best.pth exists.')

    # Ensure Grad-CAM explanation uses the exact same conditional CLAHE routing as model inference
    img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    if use_robustness:
        processed, ita, tone_lbl, reliability, clahe_used = robustness_preprocess(img_resized)
    else:
        processed = img_resized

    tensor = val_tf(image=processed)["image"].unsqueeze(0).to(next(model.parameters()).device)
    cam = GradCAM(model).generate(tensor)
    overlay = generate_heatmap_overlay(image, cam, alpha=alpha)

    if output_path is None:
        if isinstance(image_path, (str, Path)):
            p = Path(image_path)
            if "EDI - Copy" in str(p) or not os.access(p.parent, os.W_OK):
                output_path = os.path.join("outputs", "gradcam", f"{p.stem}_gradcam.png")
            else:
                root, _ = os.path.splitext(str(image_path))
                output_path = f"{root}_gradcam.png"
        else:
            output_path = "outputs/gradcam/gradcam_output.png"

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return output_path

