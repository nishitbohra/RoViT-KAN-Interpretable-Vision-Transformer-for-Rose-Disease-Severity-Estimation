import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from torch.utils.data import DataLoader

from configs.config import Config
from data.dataset import RoseLeafDataset
from data.transforms import augmented_transforms, inference_transforms
from experiments.baselines import run_baseline_experiments


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run baseline model experiments for rose leaf disease classification'
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default='..',
        help='Path to parent directory containing "Augmented Image" and "Original Image" folders'
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Baseline Model Experiments")
    print("=" * 80)
    print(f"Data root: {args.data_root}")
    print(f"Epochs: {args.epochs}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Run baseline experiments (function creates its own config and dataloaders)
    results = run_baseline_experiments(
        data_root=args.data_root,
        max_epochs=args.epochs
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
        key=lambda x: x[1].get('accuracy', 0),
        reverse=True
    )
    
    for model_name, metrics in sorted_results:
        print(
            f"{model_name:<20} "
            f"{metrics.get('accuracy', 0):.4f}      "
            f"{metrics.get('macro_f1', 0):.4f}      "
            f"{metrics.get('fps', 0):.2f}      "
            f"{metrics.get('params', 0)/1e6:.2f}"
        )
    
    print("-" * 80)
    print("\nBest Model:")
    best_model, best_result = sorted_results[0]
    print(f"  {best_model} - Accuracy: {best_result.get('accuracy', 0):.4f}")
    
    print("\nFastest Model:")
    fastest = max(results.items(), key=lambda x: x[1].get('fps', 0))
    print(f"  {fastest[0]} - FPS: {fastest[1].get('fps', 0):.2f}")
    
    print("\nMost Efficient (Params):")
    efficient = min(results.items(), key=lambda x: x[1].get('params', float('inf')))
    print(f"  {efficient[0]} - Params: {efficient[1].get('params', 0)/1e6:.2f}M")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
