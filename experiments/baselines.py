import torch
import torch.nn as nn
import timm
from pathlib import Path
from typing import Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import get_config
from data.dataset import create_dataloaders
from data.transforms import augmented_transforms, original_transforms
from training.losses import FocalLoss
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from results.logger import ExperimentLogger


class BaselineModel(nn.Module):
    
    def __init__(self, backbone_name: str, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        
        # Load backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        self.model_name = backbone_name
    
    def forward(self, x):
        logits = self.backbone(x)
        return {
            'cls_logits': logits,
            'ordinal_logits': None,
            'mu': None,
            'log_var': None,
            'kan_severity': None,
            'features': None
        }


class BaselineExperiment:
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # Baseline models to compare
        self.baseline_models = {
            'ResNet50': 'resnet50',
            'VGG16': 'vgg16',
            'EfficientNet-B0': 'efficientnet_b0',
            'MobileNetV3-Large': 'mobilenetv3_large_100',
            'Swin-T': 'swin_tiny_patch4_window7_224',
            'DeiT-Tiny': 'deit_tiny_patch16_224'
        }
    
    def run_all(self, train_loader, val_loader, test_loader,
                max_epochs: int = 30) -> Dict[str, Dict]:
        results = {}
        
        for model_name, backbone_name in self.baseline_models.items():
            print(f"\n{'='*60}")
            print(f"Training Baseline: {model_name}")
            print(f"{'='*60}\n")
            
            try:
                metrics = self._train_and_evaluate(
                    model_name, backbone_name,
                    train_loader, val_loader, test_loader,
                    max_epochs
                )
                results[model_name] = metrics
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                results[model_name] = {
                    'error': str(e),
                    'accuracy': 0.0,
                    'macro_f1': 0.0
                }
        
        return results
    
    def _train_and_evaluate(self, model_name: str, backbone_name: str,
                          train_loader, val_loader, test_loader,
                          max_epochs: int) -> Dict:
        # Build model
        model = BaselineModel(
            backbone_name=backbone_name,
            num_classes=self.config.data.num_classes,
            pretrained=True
        ).to(self.device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {num_params:,}")
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs
        )
        
        # Loss function (classification only)
        loss_fn = FocalLoss(
            gamma=self.config.loss.focal_gamma,
            alpha=None
        )
        
        # Simple loss wrapper for compatibility
        class SimpleLoss:
            def __init__(self, focal_loss):
                self.focal_loss = focal_loss
            
            def __call__(self, outputs, class_targets, severity_targets, stage=1):
                cls_loss = self.focal_loss(outputs['cls_logits'], class_targets)
                return {
                    'total_loss': cls_loss,
                    'cls_loss': cls_loss,
                    'ord_loss': torch.tensor(0.0),
                    'unc_loss': torch.tensor(0.0),
                    'kan_loss': torch.tensor(0.0)
                }
        
        loss_fn_wrapped = SimpleLoss(loss_fn)
        
        # Logger
        logger = ExperimentLogger(
            log_dir=self.config.paths.logs_dir,
            experiment_name=f"baseline_{model_name.lower().replace('-', '_')}"
        )
        
        # Modify config for baseline (no curriculum)
        baseline_config = self.config
        baseline_config.train.epochs = max_epochs
        baseline_config.flags.curriculum = False
        baseline_config.flags.freeze_backbone_epochs = 0
        
        # Trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn_wrapped,
            config=baseline_config,
            device=self.device,
            logger=logger
        )
        
        # Train
        history = trainer.fit()
        
        # Evaluate on test set
        from evaluation.metrics import accuracy, macro_f1, fps, count_params
        
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, class_labels, _ in test_loader:
                images = images.to(self.device)
                outputs = model(images)
                preds = outputs['cls_logits'].argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(class_labels.numpy())
        
        import numpy as np
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        test_metrics = {
            'accuracy': accuracy(all_labels, all_preds),
            'macro_f1': macro_f1(all_labels, all_preds),
            'fps': fps(model, (1, 3, 224, 224), self.device, n=100),
            'params': num_params
        }
        
        print(f"\n{model_name} Test Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"  Macro F1: {test_metrics['macro_f1']:.2f}%")
        print(f"  FPS: {test_metrics['fps']:.1f}")
        
        return test_metrics
    
    def save_comparison_table(self, results: Dict[str, Dict], save_path: Path):
        import pandas as pd
        
        df = pd.DataFrame(results).T
        df = df.round(2)
        
        # Sort by accuracy
        df = df.sort_values('accuracy', ascending=False)
        
        # Save
        df.to_csv(save_path)
        print(f"\nBaseline comparison table saved to {save_path}")
        print("\n" + df.to_string())


def run_baseline_experiments(data_root: str = 'data', max_epochs: int = 30):
    # Load config
    config = get_config()
    
    # Handle path: if data_root already points to Augmented/Original Image, use it directly
    data_root_path = Path(data_root)
    if data_root_path.name in ["Augmented Image", "Original Image"]:
        parent_dir = data_root_path.parent
        config.data.augmented_root = parent_dir / "Augmented Image"
        config.data.original_root = parent_dir / "Original Image"
    else:
        config.data.augmented_root = data_root_path / "Augmented Image"
        config.data.original_root = data_root_path / "Original Image"
    
    config.data.dataset_root = data_root_path
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        augmented_root=config.data.augmented_root,
        original_root=config.data.original_root,
        class_names=config.data.class_names,
        severity_map=config.data.severity_map,
        augmented_transform=augmented_transforms(),
        original_transform=original_transforms(),
        batch_size=config.train.batch_size,
        train_val_split=config.data.train_val_split,
        num_workers=config.data.num_workers,
        seed=42
    )
    
    # Run experiments
    experiment = BaselineExperiment(config, device)
    results = experiment.run_all(train_loader, val_loader, test_loader, max_epochs)
    
    # Save comparison
    save_path = config.paths.results_dir / "baseline_comparison.csv"
    experiment.save_comparison_table(results, save_path)
    
    print("\nBaseline experiments complete!")
    
    return results
