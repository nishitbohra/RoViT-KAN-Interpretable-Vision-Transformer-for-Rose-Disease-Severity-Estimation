import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple


class GradCAMPlusPlus:
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None
        
    def _register_hooks(self, target_layer):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.forward_handle = target_layer.register_forward_hook(forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    def _remove_hooks(self):
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()
    
    def compute(self, image_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.eval()
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad = True
        
        # Get target layer (last transformer block's layer norm)
        target_layer = self.model.backbone.model.blocks[-1].norm1
        
        # Register hooks
        self._register_hooks(target_layer)
        
        # Forward pass
        outputs = self.model(image_tensor)
        logits = outputs['cls_logits']
        
        # Get target class
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        target = logits[0, class_idx]
        target.backward()
        
        # Get gradients and activations
        gradients = self.gradients  # (1, N, D)
        activations = self.activations  # (1, N, D)
        
        # Remove hooks
        self._remove_hooks()
        
        if gradients is None or activations is None:
            print("Warning: Could not capture gradients or activations")
            return np.zeros((224, 224))
        
        # Grad-CAM++ weights computation
        # Alpha: second-order gradients
        alpha_num = gradients.pow(2)
        alpha_denom = 2 * gradients.pow(2) + \
                     (activations * gradients.pow(3)).sum(dim=1, keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        
        alpha = alpha_num / alpha_denom
        
        # Weights: ReLU of gradients * alpha
        weights = (alpha * F.relu(gradients)).sum(dim=2, keepdim=True)  # (1, N, 1)
        
        # Weighted combination of activations
        cam = (weights * activations).sum(dim=2)  # (1, N)
        
        # Remove CLS token (first token)
        cam = cam[:, 1:]
        
        # Reshape to spatial grid
        num_patches = int(np.sqrt(cam.shape[1]))
        cam = cam.reshape(1, num_patches, num_patches)
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze(0).cpu().numpy()
        
        # Resize to image size
        cam = cv2.resize(cam, (224, 224))
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam
    
    def overlay_on_image(self, image: np.ndarray, cam: np.ndarray,
                        alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        # Convert CAM to heatmap
        cam_uint8 = (cam * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam_uint8, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if image.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Blend
        overlaid = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlaid
    
    def visualize(self, image_tensor: torch.Tensor, original_image: np.ndarray,
                 class_idx: int = None, save_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        # Generate CAM
        cam = self.compute(image_tensor, class_idx)
        
        # Get predicted class if not specified
        if class_idx is None:
            with torch.no_grad():
                outputs = self.model(image_tensor.to(self.device))
                class_idx = outputs['cls_logits'].argmax(dim=1).item()
        
        # Overlay on image
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)
        
        overlaid = self.overlay_on_image(original_image, cam)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title(f'Grad-CAM++ (Class {class_idx})')
        axes[1].axis('off')
        
        axes[2].imshow(overlaid)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
            print(f"Grad-CAM++ visualization saved to {save_path}")
        
        plt.close()
        
        return cam, overlaid
