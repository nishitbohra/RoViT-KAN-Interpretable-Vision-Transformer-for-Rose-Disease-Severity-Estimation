import torch
import argparse
import random
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import get_config
from data.dataset import create_dataloaders
from data.transforms import augmented_transforms, original_transforms
from models.rovit_kan import RoViTKAN
from training.losses import JointLoss
from training.optimizer import build_optimizer, build_scheduler
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from results.logger import ExperimentLogger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train RoViT-KAN')
    parser.add_argument('--data_root', type=str, default='data',
                       help='Path to dataset root')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    print(f"Random seed set to {args.seed}")
    
    # Load config
    config = get_config()
    
    # Update paths if specified
    if args.data_root:
        config.data.dataset_root = Path(args.data_root)
        config.data.augmented_root = Path(args.data_root) / "Augmented Image"
        config.data.original_root = Path(args.data_root) / "Original Image"
    
    if args.output_dir:
        config.paths.checkpoints_dir = Path(args.output_dir) / "checkpoints"
        config.paths.results_dir = Path(args.output_dir) / "results"
        config.paths.figures_dir = Path(args.output_dir) / "results" / "figures"
        config.paths.logs_dir = Path(args.output_dir) / "results" / "logs"
        
        # Create directories
        config.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        config.paths.results_dir.mkdir(parents=True, exist_ok=True)
        config.paths.figures_dir.mkdir(parents=True, exist_ok=True)
        config.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
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
        seed=args.seed
    )
    
    # Build model
    print("\nBuilding model...")
    model = RoViTKAN(
        embed_dim=config.model.embed_dim,
        hidden_dim=config.model.hidden_dim,
        num_classes=config.data.num_classes,
        kan_layers=config.model.kan_layers,
        kan_num_knots=config.model.kan_num_knots,
        kan_degree=config.model.kan_degree,
        dropout=config.model.dropout,
        pretrained=config.model.pretrained
    )
    
    param_counts = model.count_parameters()
    print(f"\nModel Parameters:")
    for component, count in param_counts.items():
        print(f"  {component}: {count:,}")
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    
    # Build loss function
    # Get class weights from training dataset
    train_dataset = train_loader.dataset.dataset  # Unwrap from Subset
    class_weights = train_dataset.get_class_weights().to(device)
    
    loss_fn = JointLoss(
        lambda_ord=config.loss.lambda_ord,
        mu_unc=config.loss.mu_unc,
        nu_kan=config.loss.nu_kan,
        focal_gamma=config.loss.focal_gamma,
        focal_alpha=class_weights,
        num_classes=config.data.num_classes
    )
    
    # Create logger
    logger = ExperimentLogger(
        log_dir=config.paths.logs_dir,
        experiment_name=f"rovit_kan_seed{args.seed}"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        config=config,
        device=device,
        logger=logger
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")
    
    history = trainer.fit()
    
    # Plot training curves
    print("\nGenerating training curves...")
    logger.plot_training_curves(
        save_path=config.paths.figures_dir / f"training_curves_seed{args.seed}.png"
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60 + "\n")
    
    # Load best model
    checkpoint_path = config.paths.checkpoints_dir / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    evaluator = Evaluator(model, test_loader, config, device)
    test_metrics = evaluator.evaluate()
    
    # Save final results
    logger.log_experiment(f"rovit_kan_seed{args.seed}_test", test_metrics)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Results saved to: {config.paths.results_dir}")
    print(f"Figures saved to: {config.paths.figures_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
