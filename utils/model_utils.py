
"""
Model utilities for the multimodal mental health detection system.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

class EarlyStopping:
    """Early stopping utility to prevent overfitting."""

    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights

        Returns:
            Whether to stop training
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True

        return False

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1

    # AUC (if probabilities available)
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except ValueError:
            metrics['auc'] = 0.0

    return metrics

def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot training history.

    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
    if 'val_acc' in history:
        axes[0, 1].plot(history['val_acc'], label='Val Accuracy')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()

    # Learning rate (if available)
    if 'lr' in history:
        axes[1, 0].plot(history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')

    # F1 Score (if available)
    if 'train_f1' in history:
        axes[1, 1].plot(history['train_f1'], label='Train F1')
        if 'val_f1' in history:
            axes[1, 1].plot(history['val_f1'], label='Val F1')
        axes[1, 1].set_title('F1 Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_model(model: nn.Module, optimizer: torch.optim.Optimizer, 
               epoch: int, loss: float, path: str):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        path: Path to save checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

    logging.info(f"Model saved to {path}")

def load_model(model: nn.Module, optimizer: torch.optim.Optimizer, 
               path: str) -> Tuple[nn.Module, torch.optim.Optimizer, int, float]:
    """
    Load model checkpoint.

    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        path: Path to checkpoint

    Returns:
        Tuple of (model, optimizer, epoch, loss)
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    logging.info(f"Model loaded from {path}")
    return model, optimizer, epoch, loss

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
