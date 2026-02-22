import csv
import json
import pandas as pd
from pathlib import Path
from typing import Dict
from tabulate import tabulate
import matplotlib.pyplot as plt


class ExperimentLogger:
    
    def __init__(self, log_dir: Path, experiment_name: str = "rovit_kan"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.epoch_log_file = self.log_dir / f"{experiment_name}_epochs.csv"
        
        # Initialize epoch log
        if not self.epoch_log_file.exists():
            with open(self.epoch_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'stage', 
                    'train_loss', 'train_cls_loss', 'train_ord_loss', 
                    'train_unc_loss', 'train_kan_loss', 'train_accuracy',
                    'val_loss', 'val_cls_loss', 'val_ord_loss',
                    'val_unc_loss', 'val_kan_loss', 'val_accuracy'
                ])
    
    def log_epoch(self, epoch: int, stage: int, 
                  train_metrics: Dict, val_metrics: Dict):
        with open(self.epoch_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, stage,
                train_metrics['loss'], train_metrics['cls_loss'],
                train_metrics['ord_loss'], train_metrics['unc_loss'],
                train_metrics['kan_loss'], train_metrics['accuracy'],
                val_metrics['loss'], val_metrics['cls_loss'],
                val_metrics['ord_loss'], val_metrics['unc_loss'],
                val_metrics['kan_loss'], val_metrics['accuracy']
            ])
    
    def save_metrics(self, metrics: Dict, filename: str = 'test_metrics.json'):
        """Save metrics dict as JSON to the experiment log directory."""
        metrics_path = self.log_dir / filename

        def _make_serializable(obj):
            """Recursively convert numpy/tensor values to Python scalars."""
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            if hasattr(obj, 'item'):  # numpy scalar / torch tensor
                return obj.item()
            if isinstance(obj, float):
                return obj
            try:
                return float(obj)
            except (TypeError, ValueError):
                return str(obj)

        serializable = _make_serializable(metrics)
        with open(metrics_path, 'w') as f:
            json.dump(serializable, f, indent=4)
        print(f"Metrics saved to {metrics_path}")

    def log_experiment(self, name: str, metrics: Dict):
        exp_file = self.log_dir / f"{name}_summary.txt"
        
        with open(exp_file, 'w') as f:
            f.write(f"Experiment: {name}\n")
            f.write("=" * 60 + "\n\n")
            
            for key, value in metrics.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for sub_key, sub_value in value.items():
                        f.write(f"  {sub_key}: {sub_value}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        print(f"Experiment summary saved to {exp_file}")

    def print_table(self, metrics_dict: Dict, title: str = "Results"):
        print(f"\n{title}")
        print("=" * 60)
        
        # Flatten nested dicts
        rows = []
        for key, value in metrics_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    rows.append([f"{key}/{sub_key}", sub_value])
            else:
                rows.append([key, value])
        
        print(tabulate(rows, headers=['Metric', 'Value'], tablefmt='grid'))
        print()
    
    def plot_training_curves(self, save_path: Path = None):
        if not self.epoch_log_file.exists():
            print("No epoch log found")
            return
        
        # Load data
        df = pd.read_csv(self.epoch_log_file)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{self.experiment_name} - Training Curves', fontsize=16)
        
        # Plot 1: Total Loss
        axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Plot 2: Accuracy
        axes[0, 1].plot(df['epoch'], df['train_accuracy'], label='Train', linewidth=2)
        axes[0, 1].plot(df['epoch'], df['val_accuracy'], label='Val', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: Classification Loss
        axes[0, 2].plot(df['epoch'], df['train_cls_loss'], label='Train', linewidth=2)
        axes[0, 2].plot(df['epoch'], df['val_cls_loss'], label='Val', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('Classification Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(alpha=0.3)
        
        # Plot 4: Ordinal Loss
        axes[1, 0].plot(df['epoch'], df['train_ord_loss'], label='Train', linewidth=2)
        axes[1, 0].plot(df['epoch'], df['val_ord_loss'], label='Val', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Ordinal Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 5: Uncertainty Loss
        axes[1, 1].plot(df['epoch'], df['train_unc_loss'], label='Train', linewidth=2)
        axes[1, 1].plot(df['epoch'], df['val_unc_loss'], label='Val', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Uncertainty Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        # Plot 6: KAN Loss
        axes[1, 2].plot(df['epoch'], df['train_kan_loss'], label='Train', linewidth=2)
        axes[1, 2].plot(df['epoch'], df['val_kan_loss'], label='Val', linewidth=2)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].set_title('KAN Loss')
        axes[1, 2].legend()
        axes[1, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.close()
    
    def save_comparison_table(self, results_dict: Dict[str, Dict], 
                             save_path: Path):
        # Convert to DataFrame
        df = pd.DataFrame(results_dict).T
        
        # Save
        df.to_csv(save_path)
        print(f"Comparison table saved to {save_path}")
        
        # Also print
        print("\nComparison Table:")
        print(df.to_string())
