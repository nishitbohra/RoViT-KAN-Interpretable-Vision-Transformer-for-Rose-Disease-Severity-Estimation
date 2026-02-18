import torch
import numpy as np
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from evaluation.metrics import (
    accuracy, macro_f1, mae, spearman_rho, brier_score, 
    ece, fps, count_params, per_class_metrics
)


class Evaluator:
    
    def __init__(self, model, test_loader, config, device):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.model.eval()
    
    def evaluate(self) -> Dict[str, float]:
        print(f"\n{'='*60}")
        print("Running Evaluation on Test Set")
        print(f"{'='*60}\n")
        
        all_preds = []
        all_labels = []
        all_severity_preds = []
        all_severity_labels = []
        all_probs = []
        all_uncertainties = []
        
        with torch.no_grad():
            for images, class_labels, severity_labels in tqdm(self.test_loader, desc="Evaluating"):
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Get predictions
                cls_logits = outputs['cls_logits']
                probs = torch.softmax(cls_logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # KAN severity predictions
                if outputs['kan_severity'] is not None:
                    kan_severity = outputs['kan_severity'].squeeze()
                else:
                    kan_severity = severity_labels.float()
                
                # Store results
                all_preds.append(preds.cpu().numpy())
                all_labels.append(class_labels.numpy())
                all_severity_preds.append(kan_severity.cpu().numpy())
                all_severity_labels.append(severity_labels.numpy())
                all_probs.append(probs.cpu().numpy())
                
                # Uncertainty
                if outputs['mu'] is not None and outputs['log_var'] is not None:
                    uncertainty = torch.exp(0.5 * outputs['log_var'])
                    all_uncertainties.append(uncertainty.cpu().numpy())
        
        # Concatenate all results
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        severity_pred = np.concatenate(all_severity_preds)
        severity_true = np.concatenate(all_severity_labels)
        y_probs = np.concatenate(all_probs)
        
        # Compute metrics
        metrics = {}
        
        # Classification metrics
        metrics['accuracy'] = accuracy(y_true, y_pred)
        metrics['macro_f1'] = macro_f1(y_true, y_pred)
        
        # Ordinal metrics
        metrics['mae'] = mae(severity_true, severity_pred)
        metrics['spearman_rho'] = spearman_rho(severity_true, severity_pred)
        
        # Calibration metrics
        metrics['brier_score'] = brier_score(y_true, y_probs)
        y_conf = np.max(y_probs, axis=1)
        metrics['ece'] = ece(y_true, y_probs)
        
        # Efficiency metrics
        metrics['fps'] = fps(self.model, (1, 3, 224, 224), self.device, n=100)
        metrics['params'] = count_params(self.model)
        
        # Per-class metrics
        per_class = per_class_metrics(y_true, y_pred, self.config.data.class_names)
        metrics['per_class'] = per_class
        
        # Print results
        self._print_results(metrics)
        
        # Save results
        self._save_results(metrics, y_true, y_pred, y_probs, severity_true, severity_pred)
        
        # Generate visualizations
        self._generate_visualizations(y_true, y_pred, y_probs, severity_true, severity_pred)
        
        return metrics
    
    def _print_results(self, metrics: Dict):
        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        print(f"Accuracy:       {metrics['accuracy']:.2f}%")
        print(f"Macro F1:       {metrics['macro_f1']:.2f}%")
        print(f"MAE:            {metrics['mae']:.4f}")
        print(f"Spearman's ρ:   {metrics['spearman_rho']:.4f}")
        print(f"Brier Score:    {metrics['brier_score']:.4f}")
        print(f"ECE:            {metrics['ece']:.4f}")
        print(f"FPS:            {metrics['fps']:.1f}")
        print(f"Parameters:     {metrics['params']:,}")
        print(f"{'='*60}\n")
        
        print("Per-Class Metrics:")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"{class_name:<20} "
                  f"{class_metrics['precision']:>10.2f}%  "
                  f"{class_metrics['recall']:>10.2f}%  "
                  f"{class_metrics['f1']:>10.2f}%  "
                  f"{class_metrics['support']:>8}")
        print()
    
    def _save_results(self, metrics: Dict, y_true, y_pred, y_probs, 
                     severity_true, severity_pred):
        results_file = self.config.paths.results_dir / "evaluation_results.txt"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("RoViT-KAN Evaluation Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Accuracy:       {metrics['accuracy']:.2f}%\n")
            f.write(f"Macro F1:       {metrics['macro_f1']:.2f}%\n")
            f.write(f"MAE:            {metrics['mae']:.4f}\n")
            f.write(f"Spearman's rho: {metrics['spearman_rho']:.4f}\n")
            f.write(f"Brier Score:    {metrics['brier_score']:.4f}\n")
            f.write(f"ECE:            {metrics['ece']:.4f}\n")
            f.write(f"FPS:            {metrics['fps']:.1f}\n")
            f.write(f"Parameters:     {metrics['params']:,}\n\n")
            
            f.write("Per-Class Metrics:\n")
            f.write("-" * 60 + "\n")
            for class_name, class_metrics in metrics['per_class'].items():
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {class_metrics['precision']:.2f}%\n")
                f.write(f"  Recall:    {class_metrics['recall']:.2f}%\n")
                f.write(f"  F1-Score:  {class_metrics['f1']:.2f}%\n")
                f.write(f"  Support:   {class_metrics['support']}\n\n")
        
        print(f"Results saved to {results_file}")
    
    def _generate_visualizations(self, y_true, y_pred, y_probs, 
                                severity_true, severity_pred):
        fig_dir = self.config.paths.figures_dir
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix(y_true, y_pred, fig_dir)
        
        # 2. Confidence Histogram
        self._plot_confidence_histogram(y_probs, fig_dir)
        
        # 3. Severity Scatter Plot
        self._plot_severity_scatter(severity_true, severity_pred, fig_dir)
        
        print(f"Visualizations saved to {fig_dir}")
    
    def _plot_confusion_matrix(self, y_true, y_pred, save_dir):
        cm = sk_confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.config.data.class_names,
                   yticklabels=self.config.data.class_names)
        plt.title('Confusion Matrix - Test Set', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / 'confusion_matrix.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_histogram(self, y_probs, save_dir):
        confidences = np.max(y_probs, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Prediction Confidence Distribution', fontsize=16, pad=20)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_dir / 'confidence_histogram.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / 'confidence_histogram.pdf', bbox_inches='tight')
        plt.close()
    
    def _plot_severity_scatter(self, severity_true, severity_pred, save_dir):
        plt.figure(figsize=(10, 10))
        plt.scatter(severity_true, severity_pred, alpha=0.5, s=20)
        plt.plot([0, 3], [0, 3], 'r--', linewidth=2, label='Perfect Prediction')
        plt.xlabel('True Severity', fontsize=12)
        plt.ylabel('Predicted Severity', fontsize=12)
        plt.title('KAN Severity Prediction', fontsize=16, pad=20)
        plt.xlim(-0.2, 3.2)
        plt.ylim(-0.2, 3.2)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_dir / 'severity_scatter.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / 'severity_scatter.pdf', bbox_inches='tight')
        plt.close()


def load_model_for_evaluation(checkpoint_path: Path, config, device):
    from models.rovit_kan import RoViTKAN
    
    # Build model
    model = RoViTKAN(
        embed_dim=config.model.embed_dim,
        hidden_dim=config.model.hidden_dim,
        num_classes=config.data.num_classes,
        kan_layers=config.model.kan_layers,
        kan_num_knots=config.model.kan_num_knots,
        kan_degree=config.model.kan_degree,
        dropout=config.model.dropout,
        pretrained=False  # Loading from checkpoint
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    
    return model
