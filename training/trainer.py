import torch
import torch.nn as nn
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np
import sys
from pathlib import Path

from data.transforms import cutmix_or_mixup

# Use the non-deprecated amp API (works for both CPU and CUDA)
try:
    from torch.amp import autocast, GradScaler
    _AMP_DEVICE = 'cuda'
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    _AMP_DEVICE = 'cuda'


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        config,
        device: torch.device,
        logger=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.logger = logger
        
        # Mixed precision training — only useful on CUDA; disable silently on CPU
        if config.flags.mixed_precision and device.type == 'cuda':
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        
        # Determine curriculum stage
        stage = self.config.get_stage_for_epoch(epoch)
        self.model.curriculum_stage = stage
        
        # Unfreeze backbone after initial epochs
        if epoch == self.config.flags.freeze_backbone_epochs + 1:
            self.model.unfreeze_backbone()
        
        total_loss = 0
        cls_loss_sum = 0
        ord_loss_sum = 0
        unc_loss_sum = 0
        kan_loss_sum = 0
        
        correct = 0
        total = 0
        
        # Print epoch header once
        print(f"Epoch {epoch}/{self.config.train.epochs} (Stage {stage}): ", end='', flush=True)
        
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, class_labels, severity_labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            class_labels = class_labels.to(self.device)
            severity_labels = severity_labels.to(self.device)
            
            # Apply CutMix or MixUp
            if self.config.flags.use_cutmix or self.config.flags.use_mixup:
                images, labels_a, labels_b, lam = cutmix_or_mixup(
                    images, class_labels,
                    use_cutmix=self.config.flags.use_cutmix,
                    use_mixup=self.config.flags.use_mixup,
                    cutmix_alpha=self.config.flags.cutmix_alpha,
                    mixup_alpha=self.config.flags.mixup_alpha
                )
                mixed = True
            else:
                mixed = False
                lam = 1.0
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast('cuda'):
                    outputs = self.model(images)
                    
                    # Compute loss
                    if mixed:
                        # For mixed samples, compute loss for both labels
                        losses_a = self.loss_fn(outputs, labels_a, severity_labels, stage)
                        losses_b = self.loss_fn(outputs, labels_b, severity_labels, stage)
                        losses = {
                            k: lam * losses_a[k] + (1 - lam) * losses_b[k]
                            for k in losses_a.keys()
                        }
                    else:
                        losses = self.loss_fn(outputs, class_labels, severity_labels, stage)
                    
                    loss = losses['total_loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.flags.gradient_clip
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                losses = self.loss_fn(outputs, class_labels, severity_labels, stage)
                loss = losses['total_loss']
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.flags.gradient_clip
                )
                self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            cls_loss_sum += losses['cls_loss'].item()
            ord_loss_sum += losses['ord_loss'].item()
            unc_loss_sum += losses['unc_loss'].item()
            kan_loss_sum += losses['kan_loss'].item()
            
            # Compute accuracy
            _, predicted = outputs['cls_logits'].max(1)
            total += class_labels.size(0)
            correct += predicted.eq(class_labels).sum().item()
            
            # Print progress every 10% of batches
            progress_pct = int((batch_idx + 1) / num_batches * 100)
            if (batch_idx + 1) % max(1, num_batches // 10) == 0 or (batch_idx + 1) == num_batches:
                print(f"{progress_pct}%", end='...' if (batch_idx + 1) < num_batches else '', flush=True)
        
        # Print completion with metrics
        print(f" Loss: {total_loss/num_batches:.4f}, Acc: {100.*correct/total:.2f}%")
        
        # Compute average metrics
        num_batches = len(self.train_loader)
        metrics = {
            'loss': total_loss / num_batches,
            'cls_loss': cls_loss_sum / num_batches,
            'ord_loss': ord_loss_sum / num_batches,
            'unc_loss': unc_loss_sum / num_batches,
            'kan_loss': kan_loss_sum / num_batches,
            'accuracy': 100. * correct / total
        }
        
        return metrics
    
    def val_epoch(self) -> Dict[str, float]:
        self.model.eval()
        
        total_loss = 0
        cls_loss_sum = 0
        ord_loss_sum = 0
        unc_loss_sum = 0
        kan_loss_sum = 0
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, class_labels, severity_labels in self.val_loader:
                images = images.to(self.device)
                class_labels = class_labels.to(self.device)
                severity_labels = severity_labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss (always use full model for validation)
                losses = self.loss_fn(outputs, class_labels, severity_labels, stage=4)
                loss = losses['total_loss']
                
                # Accumulate losses
                total_loss += loss.item()
                cls_loss_sum += losses['cls_loss'].item()
                ord_loss_sum += losses['ord_loss'].item()
                unc_loss_sum += losses['unc_loss'].item()
                kan_loss_sum += losses['kan_loss'].item()
                
                # Compute accuracy
                _, predicted = outputs['cls_logits'].max(1)
                total += class_labels.size(0)
                correct += predicted.eq(class_labels).sum().item()
        
        # Compute average metrics
        num_batches = len(self.val_loader)
        metrics = {
            'loss': total_loss / num_batches,
            'cls_loss': cls_loss_sum / num_batches,
            'ord_loss': ord_loss_sum / num_batches,
            'unc_loss': unc_loss_sum / num_batches,
            'kan_loss': kan_loss_sum / num_batches,
            'accuracy': 100. * correct / total
        }
        
        return metrics
    
    def fit(self) -> Dict[str, any]:
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Total Epochs: {self.config.train.epochs}")
        print(f"Curriculum: {self.config.flags.curriculum}")
        print(f"Mixed Precision: {self.config.flags.mixed_precision}")
        print(f"{'='*60}\n")
        
        # Freeze backbone initially
        if self.config.flags.freeze_backbone_epochs > 0:
            self.model.freeze_backbone()
            print(f"Backbone frozen for first {self.config.flags.freeze_backbone_epochs} epochs\n")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        for epoch in range(1, self.config.train.epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.val_epoch()
            
            # Step scheduler
            self.scheduler.step()
            
            # Log metrics
            if self.logger:
                stage = self.config.get_stage_for_epoch(epoch)
                self.logger.log_epoch(epoch, stage, train_metrics, val_metrics)
            
            # Print summary
            print(f"\nEpoch {epoch}/{self.config.train.epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.2f}%")
            
            # Save history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            # Check for improvement
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.best_epoch = epoch
                
                # Save best checkpoint
                checkpoint_path = self.config.paths.checkpoints_dir / "best_model.pth"
                self.save_checkpoint(checkpoint_path, epoch, val_metrics)
                print(f"  [BEST] New best model saved (Val Loss: {val_metrics['loss']:.4f})")
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{self.config.train.early_stop_patience})")
            
            # Early stopping
            if self.patience_counter >= self.config.train.early_stop_patience:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best epoch: {self.best_epoch} (Val Loss: {self.best_val_loss:.4f})")
                print(f"{'='*60}\n")
                break
        
        print(f"\n{'='*60}")
        print(f"Training Complete")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        return history
    
    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Checkpoint loaded from {path}")
        print(f"Epoch: {checkpoint['epoch']}")
        print(f"Val Loss: {checkpoint['best_val_loss']:.4f}")
