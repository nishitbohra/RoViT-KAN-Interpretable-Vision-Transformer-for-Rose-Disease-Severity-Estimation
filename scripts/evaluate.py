import torch
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import get_config
from data.dataset import RoseLeafDataset
from data.transforms import original_transforms
from models.rovit_kan import RoViTKAN
from evaluation.evaluator import Evaluator, load_model_for_evaluation


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate RoViT-KAN')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='data',
                       help='Path to dataset root')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    args = parser.parse_args()
    
    # Load config
    config = get_config()
    
    # Update paths
    config.data.dataset_root = Path(args.data_root)
    config.data.original_root = Path(args.data_root) / "Original Image"
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = RoseLeafDataset(
        root_dir=config.data.original_root,
        class_names=config.data.class_names,
        severity_map=config.data.severity_map,
        transform=original_transforms(),
        mode='original'
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model_for_evaluation(Path(args.checkpoint), config, device)
    
    # Evaluate
    evaluator = Evaluator(model, test_loader, config, device)
    metrics = evaluator.evaluate()
    
    print("\nEvaluation Complete!")


if __name__ == "__main__":
    main()
