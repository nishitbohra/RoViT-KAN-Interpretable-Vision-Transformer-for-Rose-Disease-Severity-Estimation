import torch
import torch.nn as nn
import timm
from typing import Optional


class DeiTTinyBackbone(nn.Module):
    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        
        # Load DeiT-Tiny from timm
        self.model = timm.create_model(
            'deit_tiny_patch16_224',
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        self.embed_dim = self.model.num_features  # Get actual embedding dimension
        
        if freeze:
            self.freeze()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        return features
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        print("Backbone frozen")
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")
    
    def get_attention_maps(self, x: torch.Tensor):
        attention_maps = []
        
        def hook_fn(module, input, output):
            # Extract attention weights
            # Output format depends on timm version
            if isinstance(output, tuple):
                attn = output[1]  # Attention weights
            else:
                attn = output
            attention_maps.append(attn)
        
        # Register hooks on attention layers
        hooks = []
        for block in self.model.blocks:
            hook = block.attn.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        # Forward pass
        _ = self.model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps


def freeze_backbone(model: nn.Module, freeze: bool = True):
    if hasattr(model, 'backbone'):
        if freeze:
            model.backbone.freeze()
        else:
            model.backbone.unfreeze()
    else:
        raise AttributeError("Model does not have 'backbone' attribute")


def get_backbone_output_dim(backbone_name: str = 'deit_tiny_patch16_224') -> int:
    backbone_dims = {
        'deit_tiny_patch16_224': 384,
        'deit_small_patch16_224': 384,
        'deit_base_patch16_224': 768,
    }
    
    return backbone_dims.get(backbone_name, 384)
