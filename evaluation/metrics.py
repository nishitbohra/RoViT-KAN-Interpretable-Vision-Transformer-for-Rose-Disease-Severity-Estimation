import numpy as np
import torch
import time
from sklearn.metrics import f1_score, confusion_matrix
from scipy.stats import spearmanr
from typing import Tuple


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred) * 100


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average='macro') * 100


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rho, _ = spearmanr(y_true, y_pred)
    return rho


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    num_classes = y_proba.shape[1]
    
    # Convert labels to one-hot
    y_true_onehot = np.zeros_like(y_proba)
    y_true_onehot[np.arange(len(y_true)), y_true] = 1
    
    # Brier score = mean squared difference
    brier = np.mean(np.sum((y_proba - y_true_onehot) ** 2, axis=1))
    
    return brier


def ece(y_true: np.ndarray, y_conf: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Get predicted labels from confidence
    y_pred = np.argmax(y_conf, axis=1) if y_conf.ndim > 1 else (y_conf > 0.5).astype(int)
    confidences = np.max(y_conf, axis=1) if y_conf.ndim > 1 else y_conf
    accuracies = (y_pred == y_true).astype(float)
    
    ece_score = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece_score += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece_score


def fps(model: torch.nn.Module, input_size: Tuple[int, int, int, int],
        device: torch.device, n: int = 100) -> float:
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(n):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    fps_score = (n * input_size[0]) / total_time
    
    return fps_score


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: list) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=range(len(class_names)))


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     class_names: list) -> dict:
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    
    metrics = {}
    for i, class_name in enumerate(class_names):
        metrics[class_name] = {
            'precision': precision[i] * 100,
            'recall': recall[i] * 100,
            'f1': f1[i] * 100,
            'support': int(support[i])
        }
    
    return metrics
