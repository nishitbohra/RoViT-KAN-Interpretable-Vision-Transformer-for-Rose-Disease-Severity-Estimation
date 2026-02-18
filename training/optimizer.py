import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Tuple


def build_optimizer(model: torch.nn.Module, config) -> AdamW:
    # Separate parameters into backbone and heads
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    # Different learning rates: backbone gets lr/10, heads get lr
    optimizer = AdamW([
        {'params': backbone_params, 'lr': config.train.learning_rate / 10},
        {'params': head_params, 'lr': config.train.learning_rate}
    ], weight_decay=config.train.weight_decay)
    
    print(f"Optimizer: AdamW")
    print(f"  Backbone LR: {config.train.learning_rate / 10:.6f}")
    print(f"  Heads LR: {config.train.learning_rate:.6f}")
    print(f"  Weight Decay: {config.train.weight_decay}")
    
    return optimizer


def build_scheduler(optimizer: AdamW, config) -> CosineAnnealingLR:
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.train.epochs,
        eta_min=1e-6
    )
    
    print(f"Scheduler: CosineAnnealingLR (T_max={config.train.epochs})")
    
    return scheduler


def get_lr(optimizer: AdamW) -> float:
    for param_group in optimizer.param_groups:
        return param_group['lr']
