import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from configs.config import Config
from data.dataset import RoseLeafDataset
from data.transforms import get_train_transforms, get_test_transforms
from experiments.ablation import run_ablation_study


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run ablation study for RoViT-KAN'
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default='../Augmented Image',
        help='Path to dataset root directory'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/ablation',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs per experiment'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Run with reduced epochs for quick testing'
    )
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = Config()
    config.data.data_root = args.data_root
    config.training.batch_size = args.batch_size
    config.training.num_workers = args.num_workers
    
    # Adjust epochs if fast mode
    if args.fast:
        config.training.num_epochs = 10
        config.training.early_stopping_patience = 3
        print("\nRunning in FAST mode (10 epochs per experiment)")
    else:
        config.training.num_epochs = args.epochs
    
    # Create datasets
    print("\nLoading datasets...")
    train_transform = get_train_transforms(config)
    test_transform = get_test_transforms(config)
    
    train_dataset = RoseLeafDataset(
        root_dir=config.data.data_root,
        split='train',
        transform=train_transform,
        mode=config.data.mode
    )
    
    val_dataset = RoseLeafDataset(
        root_dir=config.data.data_root,
        split='val',
        transform=test_transform,
        mode=config.data.mode
    )
    
    test_dataset = RoseLeafDataset(
        root_dir=config.data.data_root,
        split='test',
        transform=test_transform,
        mode=config.data.mode
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    print("\n" + "=" * 80)
    print("Starting Ablation Study")
    print("=" * 80)
    print(f"Experiments to run: 6")
    print(f"  1. Full RoViT-KAN model")
    print(f"  2. Without ordinal regression head")
    print(f"  3. Without uncertainty estimation")
    print(f"  4. Without KAN severity module")
    print(f"  5. Without curriculum learning")
    print(f"  6. Classification head only (minimal)")
    print("=" * 80)
    
    # Run ablation study
    results = run_ablation_study(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir
    )
    
    # Print final summary
    print("\n" + "=" * 80)
    print("Ablation Study Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary CSV: {output_dir / 'ablation_results.csv'}")
    
    # Print key findings
    if 'full_model' in results:
        full_acc = results['full_model']['metrics']['accuracy']
        print(f"\nFull Model Accuracy: {full_acc:.4f}")
        
        print("\nPerformance Impact (Accuracy Drop):")
        print("-" * 60)
        
        comparisons = [
            ('no_ordinal', 'Removing Ordinal Head'),
            ('no_uncertainty', 'Removing Uncertainty Head'),
            ('no_kan', 'Removing KAN Module'),
            ('no_curriculum', 'Removing Curriculum Learning'),
            ('classification_only', 'Minimal Model (Cls Only)')
        ]
        
        for exp_name, description in comparisons:
            if exp_name in results:
                exp_acc = results[exp_name]['metrics']['accuracy']
                delta = full_acc - exp_acc
                impact = "significant" if abs(delta) > 0.01 else "minimal"
                print(f"{description:<40} {delta:+.4f} ({impact})")
        
        print("-" * 60)
    
    # Efficiency analysis
    print("\nEfficiency Analysis:")
    print("-" * 80)
    print(f"{'Model':<30} {'Params (M)':<15} {'FPS':<15} {'Accuracy':<15}")
    print("-" * 80)
    
    for exp_name, result in results.items():
        metrics = result['metrics']
        print(
            f"{exp_name:<30} "
            f"{metrics.get('params_m', 0):<15.2f} "
            f"{metrics.get('fps', 0):<15.2f} "
            f"{metrics.get('accuracy', 0):<15.4f}"
        )
    
    print("-" * 80)
    
    # Best tradeoff
    print("\nRecommendations:")
    
    # Find best accuracy
    best_acc_exp = max(results.items(), key=lambda x: x[1]['metrics'].get('accuracy', 0))
    print(f"  Best Accuracy: {best_acc_exp[0]} ({best_acc_exp[1]['metrics']['accuracy']:.4f})")
    
    # Find fastest
    fastest_exp = max(results.items(), key=lambda x: x[1]['metrics'].get('fps', 0))
    print(f"  Fastest: {fastest_exp[0]} ({fastest_exp[1]['metrics']['fps']:.2f} FPS)")
    
    # Find most efficient (smallest)
    smallest_exp = min(results.items(), key=lambda x: x[1]['metrics'].get('params_m', float('inf')))
    print(f"  Most Efficient: {smallest_exp[0]} ({smallest_exp[1]['metrics']['params_m']:.2f}M params)")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
