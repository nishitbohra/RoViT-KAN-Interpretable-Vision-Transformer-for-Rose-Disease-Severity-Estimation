import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


class BSplineBasis:
    @staticmethod
    def compute_basis(x: torch.Tensor, knots: torch.Tensor, degree: int = 3) -> torch.Tensor:

        num_knots = knots.size(0)
        num_basis = num_knots - degree - 1
        
        # Ensure x is in valid range
        x = torch.clamp(x, knots[0], knots[-1])
        
        # Initialize basis functions
        batch_size, dim = x.shape
        basis = torch.zeros(batch_size, dim, num_basis, device=x.device)
        
        # Degree 0 (piecewise constant)
        for i in range(num_basis):
            mask = (x >= knots[i]) & (x < knots[i + 1])
            basis[:, :, i] = mask.float()
        
        # Higher degrees using Cox-de Boor recursion
        for d in range(1, degree + 1):
            new_basis = torch.zeros_like(basis)
            for i in range(num_basis):
                # Left term
                if knots[i + d] != knots[i]:
                    left = (x - knots[i]) / (knots[i + d] - knots[i])
                    new_basis[:, :, i] += left * basis[:, :, i]
                
                # Right term
                if i + d + 1 < num_knots and knots[i + d + 1] != knots[i + 1]:
                    right = (knots[i + d + 1] - x) / (knots[i + d + 1] - knots[i + 1])
                    if i + 1 < num_basis:
                        new_basis[:, :, i] += right * basis[:, :, i + 1]
            
            basis = new_basis
        
        return basis


class KANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 num_knots: int = 5, degree: int = 3):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_knots = num_knots
        self.degree = degree
        self.num_basis = num_knots + degree - 1
        
        # Initialize knots uniformly in [-1, 1]
        knots = torch.linspace(-1, 1, num_knots + 2 * degree)
        self.register_buffer('knots', knots)
        
        # Learnable spline coefficients for each (input_dim, output_dim) pair
        self.spline_weights = nn.Parameter(
            torch.randn(in_features, out_features, self.num_basis) * 0.1
        )
        
        # Linear transformation (standard neural network component)
        self.linear = nn.Linear(in_features, out_features, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Normalize input to [-1, 1] range for spline computation
        x_norm = torch.tanh(x)
        
        # Compute B-spline basis functions
        basis = BSplineBasis.compute_basis(
            x_norm, self.knots, self.degree
        )  # (B, in_features, num_basis)
        
        # Apply spline transformation
        # For each input dimension, compute weighted sum of basis functions
        spline_output = torch.zeros(batch_size, self.out_features, device=x.device)
        
        for i in range(self.in_features):
            for j in range(self.out_features):
                # Weighted combination of basis functions
                spline_contrib = (basis[:, i, :] * self.spline_weights[i, j]).sum(dim=1)
                spline_output[:, j] += spline_contrib
        
        # Combine spline output with linear transformation
        linear_output = self.linear(x)
        output = linear_output + spline_output
        
        return output
    
    def get_spline_weights(self) -> torch.Tensor:
        return self.spline_weights.detach()
    
    def plot_activation(self, input_idx: int = 0, output_idx: int = 0,
                       num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        x_vals = torch.linspace(-1, 1, num_points, device=self.knots.device)
        x_vals = x_vals.unsqueeze(0).unsqueeze(0)  # (1, 1, num_points)
        
        # Compute basis
        basis = BSplineBasis.compute_basis(
            x_vals.squeeze(0), self.knots, self.degree
        )  # (1, num_points, num_basis)
        
        # Compute activation
        weights = self.spline_weights[input_idx, output_idx]  # (num_basis,)
        y_vals = (basis[0, :, :] * weights).sum(dim=1)
        
        return x_vals.squeeze().cpu().numpy(), y_vals.cpu().numpy()


class KANSeverityModule(nn.Module):   
    def __init__(self, layers: List[int] = [384, 64, 16, 1],
                 num_knots: int = 5, degree: int = 3):
        super().__init__()
        
        self.layers_dims = layers
        self.num_knots = num_knots
        self.degree = degree
        
        # Build KAN layers
        self.kan_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.kan_layers.append(
                KANLayer(layers[i], layers[i + 1], num_knots, degree)
            )
        
        # Activation functions between layers (except last)
        self.activations = nn.ModuleList([
            nn.ReLU() for _ in range(len(layers) - 2)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, kan_layer in enumerate(self.kan_layers[:-1]):
            x = kan_layer(x)
            x = self.activations[i](x)
        
        # Final layer
        x = self.kan_layers[-1](x)
        
        # Constrain output to [0, 3] range using sigmoid
        x = 3.0 * torch.sigmoid(x)
        
        return x
    
    def get_spline_weights(self) -> List[torch.Tensor]:
        return [layer.get_spline_weights() for layer in self.kan_layers]
    
    def get_activation_trajectory(self, x: torch.Tensor) -> List[torch.Tensor]:
        activations = [x]
        
        for i, kan_layer in enumerate(self.kan_layers[:-1]):
            x = kan_layer(x)
            x = self.activations[i](x)
            activations.append(x)
        
        # Final layer
        x = self.kan_layers[-1](x)
        x = 3.0 * torch.sigmoid(x)
        activations.append(x)
        
        return activations
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
