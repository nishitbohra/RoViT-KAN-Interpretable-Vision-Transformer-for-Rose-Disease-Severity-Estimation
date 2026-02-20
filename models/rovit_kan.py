import torch
import torch.nn as nn
from typing import Dict
from models.backbone import DeiTTinyBackbone
from models.heads import ClassificationHead, OrdinalHead, UncertaintyHead
from models.kan import KANSeverityModule


class RoViTKAN(nn.Module):
    def __init__(
        self,
        config_or_embed_dim = None,
        hidden_dim: int = 128,
        num_classes: int = 4,
        kan_layers: list = None,
        kan_num_knots: int = 5,
        kan_degree: int = 3,
        dropout: float = 0.3,
        pretrained: bool = True
    ):
        super().__init__()
        
        # Handle config object or individual parameters
        if hasattr(config_or_embed_dim, 'model'):
            # It's a Config object
            config = config_or_embed_dim
            embed_dim = config.model.embed_dim
            hidden_dim = config.model.hidden_dim
            num_classes = config.data.num_classes
            kan_layers = config.model.kan_layers
            kan_num_knots = config.model.kan_num_knots
            kan_degree = config.model.kan_degree
            dropout = config.model.dropout
            pretrained = config.model.pretrained
        else:
            # Individual parameters
            embed_dim = config_or_embed_dim
        
        # Backbone: DeiT-Tiny feature extractor
        self.backbone = DeiTTinyBackbone(pretrained=pretrained, freeze=False)
        
        # Use actual backbone embed_dim if not specified
        if embed_dim is None:
            embed_dim = self.backbone.embed_dim
        
        if kan_layers is None:
            kan_layers = [embed_dim, 64, 16, 1]
        
        # Task-specific heads
        self.classification_head = ClassificationHead(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.ordinal_head = OrdinalHead(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.uncertainty_head = UncertaintyHead(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.kan_module = KANSeverityModule(
            layers=kan_layers,
            num_knots=kan_num_knots,
            degree=kan_degree
        )
        
        # Curriculum stage (1-4)
        self._curriculum_stage = 4
        
    @property
    def curriculum_stage(self) -> int:
        return self._curriculum_stage
    
    @curriculum_stage.setter
    def curriculum_stage(self, stage: int):
        assert 1 <= stage <= 4, "Stage must be between 1 and 4"
        self._curriculum_stage = stage
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features from backbone
        features = self.backbone(x)  # (B, 384)
        
        # Classification head (always active)
        cls_logits = self.classification_head(features)
        
        # Initialize outputs
        outputs = {
            'cls_logits': cls_logits,
            'features': features
        }
        
        # Stage 2+: Ordinal head
        if self._curriculum_stage >= 2:
            ordinal_logits = self.ordinal_head(features)
            outputs['ordinal_logits'] = ordinal_logits
        else:
            outputs['ordinal_logits'] = None
        
        # Stage 3+: Uncertainty head
        if self._curriculum_stage >= 3:
            mu, log_var = self.uncertainty_head(features)
            outputs['mu'] = mu
            outputs['log_var'] = log_var
        else:
            outputs['mu'] = None
            outputs['log_var'] = None
        
        # Stage 4: KAN module
        if self._curriculum_stage >= 4:
            kan_severity = self.kan_module(features)
            outputs['kan_severity'] = kan_severity
        else:
            outputs['kan_severity'] = None
        
        return outputs
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            # Classification predictions
            cls_probs = torch.softmax(outputs['cls_logits'], dim=1)
            cls_pred = torch.argmax(cls_probs, dim=1)
            
            predictions = {
                'class': cls_pred,
                'class_probs': cls_probs,
                'features': outputs['features']
            }
            
            # Ordinal predictions
            if outputs['ordinal_logits'] is not None:
                ordinal_probs = self.ordinal_head.predict_probabilities(
                    outputs['features']
                )
                ordinal_severity = self.ordinal_head.predict_severity(
                    outputs['features']
                )
                predictions['ordinal_probs'] = ordinal_probs
                predictions['ordinal_severity'] = ordinal_severity
            
            # Uncertainty predictions
            if outputs['mu'] is not None:
                predictions['uncertainty_mu'] = outputs['mu']
                predictions['uncertainty_std'] = torch.exp(0.5 * outputs['log_var'])
            
            # KAN predictions
            if outputs['kan_severity'] is not None:
                predictions['kan_severity'] = outputs['kan_severity']
            
            return predictions
    
    def freeze_backbone(self):
        self.backbone.freeze()
    
    def unfreeze_backbone(self):
        self.backbone.unfreeze()
    
    def get_attention_maps(self, x: torch.Tensor):
        return self.backbone.get_attention_maps(x)
    
    def count_parameters(self) -> Dict[str, int]:
        counts = {
            'backbone': sum(p.numel() for p in self.backbone.parameters() if p.requires_grad),
            'classification_head': sum(p.numel() for p in self.classification_head.parameters() if p.requires_grad),
            'ordinal_head': sum(p.numel() for p in self.ordinal_head.parameters() if p.requires_grad),
            'uncertainty_head': sum(p.numel() for p in self.uncertainty_head.parameters() if p.requires_grad),
            'kan_module': sum(p.numel() for p in self.kan_module.parameters() if p.requires_grad),
        }
        counts['total'] = sum(counts.values())
        return counts
