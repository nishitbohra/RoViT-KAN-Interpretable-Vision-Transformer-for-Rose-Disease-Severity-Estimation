import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List


class KANVisualizer:
    
    def __init__(self):
        sns.set_style("whitegrid")
    
    def plot_spline_activations(self, kan_module, save_path: str = None,
                               num_samples: int = 5):
        num_layers = len(kan_module.kan_layers)
        
        fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5))
        if num_layers == 1:
            axes = [axes]
        
        for layer_idx, kan_layer in enumerate(kan_module.kan_layers):
            ax = axes[layer_idx]
            
            # Sample a few input-output dimension pairs
            in_dim = kan_layer.in_features
            out_dim = kan_layer.out_features
            
            num_to_plot = min(num_samples, min(in_dim, out_dim))
            
            # Plot activation curves
            for i in range(num_to_plot):
                input_idx = i if i < in_dim else 0
                output_idx = i if i < out_dim else 0
                
                x_vals, y_vals = kan_layer.plot_activation(input_idx, output_idx)
                ax.plot(x_vals, y_vals, alpha=0.7, linewidth=2,
                       label=f'In{input_idx}→Out{output_idx}')
            
            ax.set_xlabel('Input Activation', fontsize=11)
            ax.set_ylabel('Output Activation', fontsize=11)
            ax.set_title(f'Layer {layer_idx + 1}: {in_dim}→{out_dim}', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        
        plt.suptitle('KAN Learned Spline Activations', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.savefig(Path(save_path).with_suffix('.pdf'), bbox_inches='tight')
            print(f"Spline activations saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_severity_trajectory(self, kan_module, features_batch: torch.Tensor,
                                labels_batch: torch.Tensor, class_names: List[str],
                                save_path: str = None):
        kan_module.eval()
        
        with torch.no_grad():
            # Get activations at each layer
            activations_list = kan_module.get_activation_trajectory(features_batch)
        
        # Convert to numpy
        activations_np = [act.cpu().numpy() for act in activations_list]
        labels_np = labels_batch.cpu().numpy()
        
        # Create figure
        num_layers = len(activations_np)
        fig, axes = plt.subplots(1, num_layers - 1, figsize=(5 * (num_layers - 1), 5))
        if num_layers - 1 == 1:
            axes = [axes]
        
        # Color map for severity levels
        colors = plt.cm.RdYlGn_r(labels_np / 3.0)  # 0-3 mapped to colormap
        
        # Plot transitions between layers
        for i in range(num_layers - 1):
            ax = axes[i]
            
            current_act = activations_np[i]  # (B, D_i)
            next_act = activations_np[i + 1]  # (B, D_{i+1})
            
            # Use mean activation as representative
            current_mean = current_act.mean(axis=1)  # (B,)
            next_mean = next_act.mean(axis=1)  # (B,)
            
            # Scatter plot
            scatter = ax.scatter(current_mean, next_mean, c=colors, s=50, alpha=0.6)
            
            # Add diagonal reference
            all_vals = np.concatenate([current_mean, next_mean])
            min_val, max_val = all_vals.min(), all_vals.max()
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
            
            ax.set_xlabel(f'Layer {i + 1} (mean activation)', fontsize=11)
            ax.set_ylabel(f'Layer {i + 2} (mean activation)', fontsize=11)
            ax.set_title(f'Transition {i + 1}→{i + 2}', fontsize=12)
            ax.grid(alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r,
                                   norm=plt.Normalize(vmin=0, vmax=3))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation='vertical', pad=0.02)
        cbar.set_label('Severity Level', fontsize=11)
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(class_names)
        
        plt.suptitle('KAN Severity Activation Trajectory', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.savefig(Path(save_path).with_suffix('.pdf'), bbox_inches='tight')
            print(f"Severity trajectory saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_severity_distribution(self, predictions: np.ndarray,
                                  labels: np.ndarray, class_names: List[str],
                                  save_path: str = None):
        import pandas as pd
        
        # Create DataFrame
        df = pd.DataFrame({
            'Severity': predictions,
            'True Class': [class_names[int(label)] for label in labels],
            'True Severity': labels
        })
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Violin plot
        parts = ax.violinplot(
            [predictions[labels == i] for i in range(len(class_names))],
            positions=range(len(class_names)),
            widths=0.7,
            showmeans=True,
            showextrema=True
        )
        
        # Color violins by true severity
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(class_names)))
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        # Add ground truth reference lines
        for i in range(len(class_names)):
            ax.axhline(y=i, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylabel('Predicted Severity Score', fontsize=12)
        ax.set_xlabel('True Class', fontsize=12)
        ax.set_title('KAN Severity Prediction Distribution per Class', fontsize=14)
        ax.set_ylim(-0.3, 3.3)
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='red', linestyle='--', label='True Severity')]
        ax.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.savefig(Path(save_path).with_suffix('.pdf'), bbox_inches='tight')
            print(f"Severity distribution saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_spline_weights_heatmap(self, kan_module, save_path: str = None):
        spline_weights_list = kan_module.get_spline_weights()
        
        num_layers = len(spline_weights_list)
        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 4))
        if num_layers == 1:
            axes = [axes]
        
        for layer_idx, weights in enumerate(spline_weights_list):
            # weights shape: (in_features, out_features, num_basis)
            # Average over basis functions for visualization
            weights_avg = weights.mean(dim=2).cpu().numpy()  # (in, out)
            
            ax = axes[layer_idx]
            im = ax.imshow(weights_avg, cmap='coolwarm', aspect='auto')
            ax.set_xlabel('Output Features', fontsize=11)
            ax.set_ylabel('Input Features', fontsize=11)
            ax.set_title(f'Layer {layer_idx + 1} Weights', fontsize=12)
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
        
        plt.suptitle('KAN Spline Weights Heatmap', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.savefig(Path(save_path).with_suffix('.pdf'), bbox_inches='tight')
            print(f"Spline weights heatmap saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
