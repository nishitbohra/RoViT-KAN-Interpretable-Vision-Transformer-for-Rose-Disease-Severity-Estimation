import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from configs.config import Config
from data.dataset import RoseLeafDataset
from data.transforms import get_train_transforms, get_test_transforms
from experiments.baselines import run_baseline_experiments


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run baseline model experiments for rose leaf disease classification'
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
        default='./outputs/baselines',
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
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        choices=['resnet50', 'vgg16', 'efficientnet_b0', 'mobilenet_v3', 'swin_tiny', 'deit_tiny', 'all'],
        help='Specific models to run (default: all)'
    )
    
    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=True,
        help='Use pretrained weights'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
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
    config.training.num_epochs = args.epochs
    config.training.num_workers = args.num_workers
    
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
    
    # Determine which models to run
    if args.models is None or 'all' in args.models:
        model_names = ['resnet50', 'vgg16', 'efficientnet_b0', 'mobilenet_v3', 'swin_tiny', 'deit_tiny']
    else:
        model_names = args.models
    
    print(f"\nRunning baseline experiments for: {', '.join(model_names)}")
    
    # Run baseline experiments
    results = run_baseline_experiments(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        model_names=model_names,
        pretrained=args.pretrained
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("Baseline Experiments Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nPerformance Summary:")
    print("-" * 80)
    print(f"{'Model':<20} {'Accuracy':<12} {'Macro F1':<12} {'FPS':<12} {'Params (M)':<12}")
    print("-" * 80)
    
    # Sort by accuracy
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]['metrics'].get('accuracy', 0),
        reverse=True
    )
    
    for model_name, result in sorted_results:
        metrics = result['metrics']
        print(
            f"{model_name:<20} "
            f"{metrics.get('accuracy', 0):.4f}      "
            f"{metrics.get('macro_f1', 0):.4f}      "
            f"{metrics.get('fps', 0):.2f}      "
            f"{metrics.get('params_m', 0):.2f}"
        )
    
    print("-" * 80)
    print("\nBest Model:")
    best_model, best_result = sorted_results[0]
    print(f"  {best_model} - Accuracy: {best_result['metrics']['accuracy']:.4f}")
    
    print("\nFastest Model:")
    fastest = max(results.items(), key=lambda x: x[1]['metrics'].get('fps', 0))
    print(f"  {fastest[0]} - FPS: {fastest[1]['metrics']['fps']:.2f}")
    
    print("\nMost Efficient (Params):")
    efficient = min(results.items(), key=lambda x: x[1]['metrics'].get('params_m', float('inf')))
    print(f"  {efficient[0]} - Params: {efficient[1]['metrics']['params_m']:.2f}M")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
