import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
import time
import logging
from config import PLOT_SAVE_PATH, LOG_PATH, SEED


def set_seed(seed=SEED):
    """Set seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name, log_file, level=logging.INFO):
    """Setup logger for tracking experiments"""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (binary)
        y_proba: Predicted probabilities (if available)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = conf_matrix
    
    # Calculate sensitivity and specificity
    if conf_matrix.shape == (2, 2):
        tn, fp, fn, tp = conf_matrix.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, fold, epoch):
    """
    Plot training and validation loss and accuracy curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accuracies: List of training accuracies
        val_accuracies: List of validation accuracies
        fold: Current fold number
        epoch: Current epoch number
    """
    epochs = range(1, epoch + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title(f'Loss Curves - Fold {fold+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title(f'Accuracy Curves - Fold {fold+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_PATH, f'training_history_fold_{fold+1}.png'))
    plt.close()


def plot_confusion_matrix(cm, fold):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        fold: Current fold number
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Fold {fold+1}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(PLOT_SAVE_PATH, f'confusion_matrix_fold_{fold+1}.png'))
    plt.close()


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=5, min_delta=0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait after min/max has been hit
            min_delta: Minimum change in monitored value to qualify as improvement
            mode: 'min' or 'max' for whether we are minimizing or maximizing
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.min_validation_loss = float('inf') if mode == 'min' else float('-inf')
    
    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif (self.mode == 'min' and current_score > self.best_score + self.min_delta) or \
             (self.mode == 'max' and current_score < self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0
        
        return self.early_stop


class AverageMeter:
    """
    Computes and stores the average and current value
    Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """Class for timing operations"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
    
    def get_elapsed_time(self):
        if self.start_time is None:
            return 0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
