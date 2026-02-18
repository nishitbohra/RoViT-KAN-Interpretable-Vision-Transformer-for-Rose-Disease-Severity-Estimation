import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from typing import Tuple


class ViTAttentionRollout:
    
    def __init__(self, model, device='cuda', discard_ratio=0.9):
        self.model = model
        self.device = device
        self.discard_ratio = discard_ratio
        self.attention_maps = []
        
    def _register_hooks(self):
        self.hooks = []
        
        def get_attention_hook(module, input, output):
            # Extract attention weights from ViT attention layer
            # DeiT attention output: (B, H, N, N) where H=heads, N=tokens
            if hasattr(module, 'attn_drop'):
                # This is an attention module
                # Get attention before dropout
                attn = output[1] if isinstance(output, tuple) else output
                self.attention_maps.append(attn.detach().cpu())
        
        # Register hooks on attention layers in backbone
        for block in self.model.backbone.model.blocks:
            hook = block.attn.register_forward_hook(get_attention_hook)
            self.hooks.append(hook)
    
    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate(self, image_tensor: torch.Tensor, head_fusion: str = 'mean') -> np.ndarray:
        self.model.eval()
        self.attention_maps = []
        
        # Register hooks
        self._register_hooks()
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(image_tensor.to(self.device))
        
        # Remove hooks
        self._remove_hooks()
        
        if len(self.attention_maps) == 0:
            print("Warning: No attention maps captured")
            return np.zeros((224, 224))
        
        # Process attention maps
        # Each map is (B, H, N, N) where N = num_patches + 1 (CLS token)
        attention_maps_fused = []
        
        for attn in self.attention_maps:
            # Fuse heads
            if head_fusion == 'mean':
                attn_fused = attn.mean(dim=1)  # (B, N, N)
            elif head_fusion == 'max':
                attn_fused = attn.max(dim=1)[0]
            elif head_fusion == 'min':
                attn_fused = attn.min(dim=1)[0]
            else:
                attn_fused = attn.mean(dim=1)
            
            attention_maps_fused.append(attn_fused[0])  # Take first batch
        
        # Stack and compute rollout
        # Attention rollout: recursively multiply attention matrices
        num_tokens = attention_maps_fused[0].shape[0]
        batch_size = 1
        
        # Initialize with identity
        rollout = torch.eye(num_tokens)
        
        for attn in attention_maps_fused:
            # Add residual (identity matrix) to attention
            attn_with_residual = attn + torch.eye(num_tokens)
            # Normalize
            attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)
            # Multiply
            rollout = torch.matmul(rollout, attn_with_residual)
        
        # Get CLS token attention to all patches
        cls_attention = rollout[0, 1:]  # Exclude CLS token itself
        
        # Reshape to spatial grid
        num_patches = int(np.sqrt(cls_attention.shape[0]))
        attention_map = cls_attention.reshape(num_patches, num_patches)
        
        # Resize to image size (224x224)
        attention_map = attention_map.numpy()
        attention_map = cv2.resize(attention_map, (224, 224))
        
        # Normalize to [0, 1]
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        return attention_map
    
    def overlay_on_image(self, image: np.ndarray, attention_map: np.ndarray,
                        alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        # Convert attention to heatmap
        attention_map_uint8 = (attention_map * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(attention_map_uint8, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if image.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Blend
        overlaid = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlaid
    
    def visualize(self, image_tensor: torch.Tensor, original_image: np.ndarray,
                 save_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        # Generate attention map
        attention_map = self.generate(image_tensor)
        
        # Overlay on image
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)
        
        overlaid = self.overlay_on_image(original_image, attention_map)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(attention_map, cmap='jet')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        
        axes[2].imshow(overlaid)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
            print(f"Attention visualization saved to {save_path}")
        
        plt.close()
        
        return attention_map, overlaid
