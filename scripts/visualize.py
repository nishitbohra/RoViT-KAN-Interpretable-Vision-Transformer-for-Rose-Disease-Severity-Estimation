import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from configs.config import Config
from models.rovit_kan import RoViTKAN
from data.dataset import RoseLeafDataset
from data.transforms import inference_transforms
from explainability.attention_maps import ViTAttentionRollout
from explainability.gradcam import GradCAMPlusPlus
from explainability.kan_viz import KANVisualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate explainability visualizations for RoViT-KAN'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
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
        default='./outputs/visualizations',
        help='Directory to save visualizations'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples to visualize'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    
    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=['attention', 'gradcam', 'kan'],
        choices=['attention', 'gradcam', 'kan', 'all'],
        help='Visualization methods to use'
    )
    
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        default=None,
        help='Specific classes to visualize (default: all)'
    )
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = Config()
    config.data.data_root = args.data_root
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = RoViTKAN(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully!")
    
    # Create dataset
    print("\nLoading dataset...")
    test_transform = inference_transforms()
    
    test_dataset = RoseLeafDataset(
        root_dir=config.data.data_root,
        split='test',
        transform=test_transform,
        mode=config.data.mode
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Filter by classes if specified
    if args.classes:
        # Filter dataset to only include specified classes
        class_to_idx = test_dataset.class_to_idx
        target_indices = [class_to_idx[cls] for cls in args.classes if cls in class_to_idx]
        
        # Create filtered indices
        filtered_indices = [i for i, (_, label) in enumerate(test_dataset) if label in target_indices]
        test_dataset = torch.utils.data.Subset(test_dataset, filtered_indices)
        print(f"Filtered to {len(test_dataset)} samples from classes: {', '.join(args.classes)}")
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    
    # Determine which methods to use
    if 'all' in args.methods:
        methods = ['attention', 'gradcam', 'kan']
    else:
        methods = args.methods
    
    print(f"\nGenerating visualizations using: {', '.join(methods)}")
    print(f"Number of samples: {args.num_samples}")
    print("=" * 80)
    
    # Initialize visualizers
    visualizers = {}
    
    if 'attention' in methods:
        visualizers['attention'] = ViTAttentionRollout(
            model=model,
            device=device
        )
        print("✓ Attention visualizer initialized")
    
    if 'gradcam' in methods:
        visualizers['gradcam'] = GradCAMPlusPlus(
            model=model,
            device=device,
            output_dir=output_dir / 'gradcam'
        )
        print("✓ Grad-CAM++ visualizer initialized")
    
    if 'kan' in methods:
        visualizers['kan'] = KANVisualizer(
            model=model,
            device=device,
            output_dir=output_dir / 'kan_visualizations'
        )
        print("✓ KAN visualizer initialized")
    
    print("=" * 80)
    
    # Generate visualizations for random samples
    class_names = test_dataset.dataset.classes if hasattr(test_dataset, 'dataset') else ['Black Spot', 'Dry Leaf', 'Healthy Leaf', 'Leaf Holes']
    
    print("\nGenerating visualizations...")
    
    sample_count = 0
    for batch_idx, (images, labels) in enumerate(test_loader):
        if sample_count >= args.num_samples:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(images)
            logits = outputs['logits']
            pred_class = torch.argmax(logits, dim=1).item()
            true_class = labels.item()
        
        pred_label = class_names[pred_class]
        true_label = class_names[true_class]
        
        print(f"\nSample {sample_count + 1}/{args.num_samples}")
        print(f"  True class: {true_label}")
        print(f"  Predicted: {pred_label}")
        
        # Generate attention maps
        if 'attention' in methods:
            try:
                visualizers['attention'].visualize_attention_rollout(
                    image=images[0],
                    true_class=true_label,
                    pred_class=pred_label,
                    sample_idx=sample_count
                )
                print("  ✓ Attention map saved")
            except Exception as e:
                print(f"  ✗ Attention map failed: {e}")
        
        # Generate Grad-CAM++
        if 'gradcam' in methods:
            try:
                visualizers['gradcam'].visualize_gradcam(
                    image=images[0],
                    true_class=true_label,
                    pred_class=pred_label,
                    sample_idx=sample_count
                )
                print("  ✓ Grad-CAM++ saved")
            except Exception as e:
                print(f"  ✗ Grad-CAM++ failed: {e}")
        
        # Generate KAN visualizations
        if 'kan' in methods and sample_count == 0:  # Only once
            try:
                # Collect features from multiple samples for KAN viz
                all_features = []
                all_labels = []
                
                temp_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                for temp_imgs, temp_lbls in temp_loader:
                    temp_imgs = temp_imgs.to(device)
                    with torch.no_grad():
                        temp_outputs = model(temp_imgs)
                        if 'kan_features' in temp_outputs:
                            all_features.append(temp_outputs['kan_features'].cpu())
                            all_labels.append(temp_lbls)
                    
                    if len(all_features) >= 5:  # ~160 samples
                        break
                
                if all_features:
                    all_features = torch.cat(all_features, dim=0)
                    all_labels = torch.cat(all_labels, dim=0)
                    
                    visualizers['kan'].visualize_spline_activations()
                    print("  ✓ KAN spline activations saved")
                    
                    visualizers['kan'].visualize_severity_trajectories(all_features, all_labels)
                    print("  ✓ KAN severity trajectories saved")
                    
                    visualizers['kan'].visualize_feature_distributions(all_features, all_labels, class_names)
                    print("  ✓ KAN feature distributions saved")
            except Exception as e:
                print(f"  ✗ KAN visualizations failed: {e}")
        
        sample_count += 1
    
    # Generate summary visualizations
    print("\n" + "=" * 80)
    print("Generating summary visualizations...")
    
    if 'attention' in methods:
        try:
            visualizers['attention'].create_attention_summary()
            print("✓ Attention summary created")
        except Exception as e:
            print(f"✗ Attention summary failed: {e}")
    
    if 'gradcam' in methods:
        try:
            visualizers['gradcam'].create_gradcam_summary()
            print("✓ Grad-CAM++ summary created")
        except Exception as e:
            print(f"✗ Grad-CAM++ summary failed: {e}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {output_dir}")
    
    if 'attention' in methods:
        print(f"  Attention maps: {output_dir / 'attention_maps'}")
    if 'gradcam' in methods:
        print(f"  Grad-CAM++: {output_dir / 'gradcam'}")
    if 'kan' in methods:
        print(f"  KAN visualizations: {output_dir / 'kan_visualizations'}")
    
    print("\nVisualization Types Generated:")
    if 'attention' in methods:
        print("  ✓ Attention rollout maps (per-sample)")
        print("  ✓ Attention summary grid")
    if 'gradcam' in methods:
        print("  ✓ Grad-CAM++ heatmaps (per-sample)")
        print("  ✓ Grad-CAM++ summary grid")
    if 'kan' in methods:
        print("  ✓ KAN spline activation curves")
        print("  ✓ KAN severity progression trajectories")
        print("  ✓ KAN feature distribution violin plots")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
