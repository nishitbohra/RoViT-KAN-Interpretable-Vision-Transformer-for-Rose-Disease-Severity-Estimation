import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import copy
from dataclasses import dataclass

from configs.config import Config
from models.rovit_kan import RoViTKAN
from models.backbone import DeiTBackbone
from models.heads import ClassificationHead, OrdinalHead, UncertaintyHead
from models.kan import KANSeverityModule
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from results.logger import ResultLogger


@dataclass
class AblationConfig:
    name: str
    remove_ordinal: bool = False
    remove_uncertainty: bool = False
    remove_kan: bool = False
    disable_curriculum: bool = False
    description: str = ""


class AblationModel(nn.Module):
    
    def __init__(
        self,
        config: Config,
        remove_ordinal: bool = False,
        remove_uncertainty: bool = False,
        remove_kan: bool = False
    ):
        super().__init__()
        self.config = config
        self.remove_ordinal = remove_ordinal
        self.remove_uncertainty = remove_uncertainty
        self.remove_kan = remove_kan
        
        # Backbone
        self.backbone = DeiTBackbone(
            model_name=config.model.backbone,
            pretrained=config.model.pretrained,
            freeze_backbone=config.model.freeze_backbone
        )
        
        # Classification head (always present)
        self.classification_head = ClassificationHead(
            in_features=self.backbone.embed_dim,
            num_classes=config.model.num_classes,
            dropout=config.model.dropout
        )
        
        # Ordinal head (optional)
        if not remove_ordinal:
            self.ordinal_head = OrdinalHead(
                in_features=self.backbone.embed_dim,
                num_thresholds=config.model.num_classes - 1,
                dropout=config.model.dropout
            )
        else:
            self.ordinal_head = None
        
        # Uncertainty head (optional)
        if not remove_uncertainty:
            self.uncertainty_head = UncertaintyHead(
                in_features=self.backbone.embed_dim,
                dropout=config.model.dropout
            )
        else:
            self.uncertainty_head = None
        
        # KAN severity module (optional)
        if not remove_kan:
            self.kan_module = KANSeverityModule(
                in_features=self.backbone.embed_dim,
                hidden_features=config.model.kan_hidden_dim,
                num_severity_levels=config.model.num_classes,
                num_splines=config.model.kan_num_splines,
                spline_order=config.model.kan_spline_order,
                dropout=config.model.dropout
            )
        else:
            self.kan_module = None
        
        self.curriculum_stage = 0
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features
        features = self.backbone(x)
        
        # Classification (always present)
        logits = self.classification_head(features)
        
        outputs = {'logits': logits}
        
        # Ordinal prediction (optional)
        if self.ordinal_head is not None:
            ordinal_logits = self.ordinal_head(features)
            outputs['ordinal_logits'] = ordinal_logits
        
        # Uncertainty estimation (optional)
        if self.uncertainty_head is not None:
            epistemic, aleatoric = self.uncertainty_head(features)
            outputs['epistemic_uncertainty'] = epistemic
            outputs['aleatoric_uncertainty'] = aleatoric
        
        # KAN severity (optional)
        if self.kan_module is not None:
            severity_score, kan_features = self.kan_module(features)
            outputs['severity_score'] = severity_score
            outputs['kan_features'] = kan_features
        
        return outputs
    
    def set_curriculum_stage(self, stage: int):
        self.curriculum_stage = stage


class AblationExperiment:
    
    def __init__(
        self,
        config: Config,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        output_dir: Path
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define ablation configurations
        self.ablation_configs = [
            AblationConfig(
                name="full_model",
                description="Full RoViT-KAN with all components"
            ),
            AblationConfig(
                name="no_ordinal",
                remove_ordinal=True,
                description="Without ordinal regression head"
            ),
            AblationConfig(
                name="no_uncertainty",
                remove_uncertainty=True,
                description="Without uncertainty estimation"
            ),
            AblationConfig(
                name="no_kan",
                remove_kan=True,
                description="Without KAN severity module"
            ),
            AblationConfig(
                name="no_curriculum",
                disable_curriculum=True,
                description="Without curriculum learning"
            ),
            AblationConfig(
                name="classification_only",
                remove_ordinal=True,
                remove_uncertainty=True,
                remove_kan=True,
                description="Classification head only (minimal model)"
            ),
        ]
        
        self.results = {}
    
    def run_all_experiments(self) -> Dict[str, Dict]:
        print("=" * 80)
        print("Starting Ablation Study")
        print("=" * 80)
        
        for ablation_cfg in self.ablation_configs:
            print(f"\n{'=' * 80}")
            print(f"Experiment: {ablation_cfg.name}")
            print(f"Description: {ablation_cfg.description}")
            print(f"{'=' * 80}\n")
            
            metrics = self.run_single_experiment(ablation_cfg)
            self.results[ablation_cfg.name] = {
                'config': ablation_cfg,
                'metrics': metrics
            }
            
            print(f"\n{ablation_cfg.name} Results:")
            self._print_metrics(metrics)
        
        # Save results
        self._save_results()
        self._print_comparison()
        
        return self.results
    
    def run_single_experiment(
        self,
        ablation_cfg: AblationConfig
    ) -> Dict[str, float]:
        # Create experiment directory
        exp_dir = self.output_dir / ablation_cfg.name
        exp_dir.mkdir(exist_ok=True)
        
        # Create model
        if ablation_cfg.name == "full_model":
            # Use original RoViT-KAN
            model = RoViTKAN(self.config).to(self.device)
        else:
            # Use ablation model
            model = AblationModel(
                self.config,
                remove_ordinal=ablation_cfg.remove_ordinal,
                remove_uncertainty=ablation_cfg.remove_uncertainty,
                remove_kan=ablation_cfg.remove_kan
            ).to(self.device)
        
        # Modify config for curriculum
        train_config = copy.deepcopy(self.config)
        if ablation_cfg.disable_curriculum:
            train_config.training.use_curriculum = False
        
        # Create logger
        logger = ResultLogger(exp_dir / "training.log")
        
        # Train model
        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=train_config,
            device=self.device,
            output_dir=exp_dir,
            logger=logger
        )
        
        trainer.train()
        
        # Evaluate on test set
        evaluator = Evaluator(
            model=model,
            test_loader=self.test_loader,
            config=self.config,
            device=self.device,
            output_dir=exp_dir
        )
        
        metrics = evaluator.evaluate()
        
        return metrics
    
    def _print_metrics(self, metrics: Dict[str, float]):
        print("\nClassification Metrics:")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"  Macro F1: {metrics.get('macro_f1', 0):.4f}")
        print(f"  Weighted F1: {metrics.get('weighted_f1', 0):.4f}")
        
        print("\nOrdinal Metrics:")
        print(f"  MAE: {metrics.get('mae', 0):.4f}")
        print(f"  Spearman: {metrics.get('spearman', 0):.4f}")
        
        print("\nUncertainty Metrics:")
        print(f"  Brier Score: {metrics.get('brier_score', 0):.4f}")
        print(f"  ECE: {metrics.get('ece', 0):.4f}")
        
        print("\nEfficiency:")
        print(f"  FPS: {metrics.get('fps', 0):.2f}")
        print(f"  Params (M): {metrics.get('params_m', 0):.2f}")
    
    def _save_results(self):
        import csv
        
        csv_path = self.output_dir / "ablation_results.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['Experiment', 'Description', 'Accuracy', 'Macro F1', 
                     'Weighted F1', 'MAE', 'Spearman', 'Brier Score', 'ECE', 
                     'FPS', 'Params (M)']
            writer.writerow(header)
            
            # Data
            for exp_name, result in self.results.items():
                metrics = result['metrics']
                config = result['config']
                
                row = [
                    exp_name,
                    config.description,
                    f"{metrics.get('accuracy', 0):.4f}",
                    f"{metrics.get('macro_f1', 0):.4f}",
                    f"{metrics.get('weighted_f1', 0):.4f}",
                    f"{metrics.get('mae', 0):.4f}",
                    f"{metrics.get('spearman', 0):.4f}",
                    f"{metrics.get('brier_score', 0):.4f}",
                    f"{metrics.get('ece', 0):.4f}",
                    f"{metrics.get('fps', 0):.2f}",
                    f"{metrics.get('params_m', 0):.2f}"
                ]
                writer.writerow(row)
        
        print(f"\nResults saved to {csv_path}")
    
    def _print_comparison(self):
        print("\n" + "=" * 80)
        print("Ablation Study Summary")
        print("=" * 80)
        
        # Get full model metrics as baseline
        if 'full_model' in self.results:
            baseline = self.results['full_model']['metrics']
            
            print(f"\n{'Experiment':<25} {'Acc':<8} {'F1':<8} {'MAE':<8} {'Δ Acc':<10}")
            print("-" * 70)
            
            for exp_name, result in self.results.items():
                metrics = result['metrics']
                acc = metrics.get('accuracy', 0)
                f1 = metrics.get('macro_f1', 0)
                mae = metrics.get('mae', 0)
                
                # Calculate delta from baseline
                delta_acc = acc - baseline.get('accuracy', 0)
                delta_str = f"{delta_acc:+.4f}" if exp_name != 'full_model' else "-"
                
                print(f"{exp_name:<25} {acc:.4f}   {f1:.4f}   {mae:.4f}   {delta_str}")
        
        print("=" * 80)
    
    def get_component_importance(self) -> Dict[str, float]:
        if 'full_model' not in self.results:
            return {}
        
        baseline_acc = self.results['full_model']['metrics'].get('accuracy', 0)
        
        importance = {}
        
        # Ordinal head
        if 'no_ordinal' in self.results:
            no_ordinal_acc = self.results['no_ordinal']['metrics'].get('accuracy', 0)
            importance['ordinal_head'] = baseline_acc - no_ordinal_acc
        
        # Uncertainty head
        if 'no_uncertainty' in self.results:
            no_unc_acc = self.results['no_uncertainty']['metrics'].get('accuracy', 0)
            importance['uncertainty_head'] = baseline_acc - no_unc_acc
        
        # KAN module
        if 'no_kan' in self.results:
            no_kan_acc = self.results['no_kan']['metrics'].get('accuracy', 0)
            importance['kan_module'] = baseline_acc - no_kan_acc
        
        # Curriculum learning
        if 'no_curriculum' in self.results:
            no_curr_acc = self.results['no_curriculum']['metrics'].get('accuracy', 0)
            importance['curriculum_learning'] = baseline_acc - no_curr_acc
        
        return importance


def run_ablation_study(
    config: Config,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path
) -> Dict[str, Dict]:
    experiment = AblationExperiment(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir
    )
    
    results = experiment.run_all_experiments()
    
    # Print component importance
    importance = experiment.get_component_importance()
    
    print("\n" + "=" * 80)
    print("Component Importance (Performance Drop When Removed)")
    print("=" * 80)
    for component, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{component:<30} {score:+.4f}")
    print("=" * 80)
    
    return results
