import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim: int = 384, hidden_dim: int = 128,
                 num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class OrdinalHead(nn.Module):
    def __init__(self, embed_dim: int = 384, hidden_dim: int = 128,
                 num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1  # K-1 thresholds for K classes
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, self.num_thresholds)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)  # (B, K-1)
        return logits
    
    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        cum_logits = self.forward(x)  # (B, K-1)
        cum_probs = torch.sigmoid(cum_logits)  # P(y <= k)
        
        # Convert cumulative to individual probabilities
        # P(y = 0) = P(y <= 0)
        # P(y = k) = P(y <= k) - P(y <= k-1) for k > 0
        # P(y = K-1) = 1 - P(y <= K-2)
        
        batch_size = cum_probs.size(0)
        probs = torch.zeros(batch_size, self.num_classes, device=cum_probs.device)
        
        # First class
        probs[:, 0] = cum_probs[:, 0]
        
        # Middle classes
        for k in range(1, self.num_classes - 1):
            probs[:, k] = cum_probs[:, k] - cum_probs[:, k - 1]
        
        # Last class
        probs[:, -1] = 1.0 - cum_probs[:, -1]
        
        return probs
    
    def predict_severity(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_probabilities(x)  # (B, K)
        
        # Expected value: sum of k * P(y = k)
        severity_levels = torch.arange(self.num_classes, dtype=torch.float32,
                                      device=probs.device)
        severity = (probs * severity_levels).sum(dim=1, keepdim=True)
        
        return severity


class UncertaintyHead(nn.Module):
    def __init__(self, embed_dim: int = 384, hidden_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_mu = nn.Linear(hidden_dim, 1)  # Mean
        self.fc_logvar = nn.Linear(hidden_dim, 1)  # Log variance
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        
        # Clamp log_var to prevent numerical instability
        log_var = torch.clamp(log_var, min=-10, max=10)
        
        return mu, log_var
    
    def sample(self, x: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
        mu, log_var = self.forward(x)
        std = torch.exp(0.5 * log_var)
        
        # Sample from N(mu, std^2)
        eps = torch.randn(x.size(0), num_samples, device=x.device)
        samples = mu + std * eps
        
        return samples
