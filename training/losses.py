import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute cross entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal term
        focal_term = (1 - pt) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class OrdinalBCELoss(nn.Module):
    def __init__(self, num_classes: int = 4, reduction: str = 'mean'):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.reduction = reduction
    
    def forward(self, cum_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = targets.size(0)
        
        # Create binary targets for each threshold
        # If target = k, then P(y <= 0), ..., P(y <= k-1) = 1, rest = 0
        binary_targets = torch.zeros(batch_size, self.num_thresholds,
                                    device=cum_logits.device)
        
        for k in range(self.num_thresholds):
            binary_targets[:, k] = (targets > k).float()
        
        # Compute BCE on each threshold
        loss = F.binary_cross_entropy_with_logits(
            cum_logits, binary_targets, reduction='none'
        )
        
        # Average across thresholds
        loss = loss.mean(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class UncertaintyLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, mu: torch.Tensor, log_var: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 1:
            targets = targets.unsqueeze(1).float()
        
        # Compute precision (inverse variance)
        precision = torch.exp(-log_var)
        
        # Reconstruction loss weighted by precision
        reconstruction = (targets - mu) ** 2 * precision
        
        # Regularization term (encourages appropriate uncertainty)
        regularization = log_var
        
        loss = 0.5 * (reconstruction + regularization)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class KANRegressionLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 1:
            targets = targets.unsqueeze(1).float()
        
        loss = F.mse_loss(predictions, targets, reduction=self.reduction)
        return loss


class JointLoss(nn.Module):
    def __init__(
        self,
        lambda_ord: float = 1.0,
        mu_unc: float = 0.5,
        nu_kan: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[torch.Tensor] = None,
        num_classes: int = 4
    ):
        super().__init__()
        
        self.lambda_ord = lambda_ord
        self.mu_unc = mu_unc
        self.nu_kan = nu_kan
        
        # Individual loss functions
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.ordinal_loss = OrdinalBCELoss(num_classes=num_classes)
        self.uncertainty_loss = UncertaintyLoss()
        self.kan_loss = KANRegressionLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        class_targets: torch.Tensor,
        severity_targets: torch.Tensor,
        stage: int = 4
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        
        # Stage 1+: Classification loss (always computed)
        cls_loss = self.focal_loss(outputs['cls_logits'], class_targets)
        losses['cls_loss'] = cls_loss
        total_loss = cls_loss
        
        # Stage 2+: Ordinal loss
        if stage >= 2 and outputs['ordinal_logits'] is not None:
            ord_loss = self.ordinal_loss(outputs['ordinal_logits'], severity_targets)
            losses['ord_loss'] = ord_loss
            total_loss = total_loss + self.lambda_ord * ord_loss
        else:
            losses['ord_loss'] = torch.tensor(0.0, device=cls_loss.device)
        
        # Stage 3+: Uncertainty loss
        if stage >= 3 and outputs['mu'] is not None and outputs['log_var'] is not None:
            unc_loss = self.uncertainty_loss(
                outputs['mu'], outputs['log_var'], severity_targets
            )
            losses['unc_loss'] = unc_loss
            total_loss = total_loss + self.mu_unc * unc_loss
        else:
            losses['unc_loss'] = torch.tensor(0.0, device=cls_loss.device)
        
        # Stage 4: KAN loss
        if stage >= 4 and outputs['kan_severity'] is not None:
            k_loss = self.kan_loss(outputs['kan_severity'], severity_targets)
            losses['kan_loss'] = k_loss
            total_loss = total_loss + self.nu_kan * k_loss
        else:
            losses['kan_loss'] = torch.tensor(0.0, device=cls_loss.device)
        
        losses['total_loss'] = total_loss
        
        return losses
