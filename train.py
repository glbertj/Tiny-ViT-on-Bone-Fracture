import numpy as np
import torch
import torch.optim as optim
import time
import os
import pandas as pd
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
from torch import nn
from matplotlib import pyplot as plt
import seaborn as sns

from utils import (
    calculate_metrics, plot_training_history, plot_confusion_matrix,
    EarlyStopping, AverageMeter, Timer, setup_logger
)
from config import (
    DEVICE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE, SWA_START,
    SWA_LR, LEARNING_RATE, WEIGHT_DECAY, MODEL_SAVE_PATH,
    LOG_PATH, WEIGHTED_LOSS, PLOT_SAVE_PATH
)
from models import get_model, get_loss_fn
from dataset import prepare_loaders, get_test_loader

def train_fold(model, train_loader, val_loader, optimizer, criterion, fold, logger):
    """
    Train a model on one fold
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        fold: Current fold number
        logger: Logger for tracking metrics
    
    Returns:
        Best model state dict, best validation metrics
    """
    # Initialize SWA model and scheduler
    swa_model = AveragedModel(model)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode='min')
    
    # Initialize metrics tracking
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    best_model_state = None
    best_val_metrics = None
    
    logger.info(f"Starting training fold {fold+1}")
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        # Train one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion)
        val_acc = val_metrics['accuracy']
        
        # Update SWA model after specified epoch
        if epoch >= SWA_START:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            # Use regular scheduler before SWA kicks in
            scheduler.step(val_loss)
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                   f"Val F1: {val_metrics['f1_score']:.4f}, Val AUC: {val_metrics.get('roc_auc', 0):.4f}")
        
        # Save metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_val_metrics = val_metrics
            logger.info(f"Epoch {epoch+1}: New best model saved!")
        
        # Check early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    # Update batch normalization statistics for SWA model
    logger.info("Updating SWA BatchNorm statistics...")
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=DEVICE)
    
    # Evaluate SWA model
    swa_val_loss, swa_val_metrics = validate(swa_model, val_loader, criterion)
    logger.info(f"SWA Model - Val Loss: {swa_val_loss:.4f}, "
               f"Val Acc: {swa_val_metrics['accuracy']:.4f}, "
               f"Val F1: {swa_val_metrics['f1_score']:.4f}, "
               f"Val AUC: {swa_val_metrics.get('roc_auc', 0):.4f}")
    
    # Determine which model is better (SWA or best checkpoint)
    if swa_val_loss < best_val_loss:
        logger.info("SWA model is better than best checkpoint model!")
        best_model_state = swa_model.state_dict()
        best_val_metrics = swa_val_metrics
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, fold, len(train_losses))
    
    # Plot confusion matrix
    plot_confusion_matrix(best_val_metrics['confusion_matrix'], fold)
    
    return best_model_state, best_val_metrics


def train_one_epoch(model, train_loader, optimizer, criterion, scheduler=None):
    """
    Train for one epoch
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        scheduler: Learning rate scheduler (optional)
    
    Returns:
        Average loss, average accuracy
    """
    model.train()
    losses = AverageMeter()
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    
    for inputs, labels in pbar:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1).float()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step()
        
        # Update metrics
        losses.update(loss.item(), inputs.size(0))
        
        # Calculate accuracy
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{losses.avg:.4f}", "acc": f"{correct/total:.4f}"})
    
    return losses.avg, correct / total


def train_and_evaluate(folds=5):
    """
    Train and evaluate models using cross-validation
    
    Args:
        folds: Number of folds for cross-validation
    
    Returns:
        Dictionary with results for each fold and overall metrics
    """
    # Initialize logger
    logger = setup_logger('train_log', os.path.join(LOG_PATH, 'training.log'))
    logger.info(f"Starting training with {folds} folds")
    
    # Initialize results dictionary
    results = {
        'fold_metrics': [],
        'fold_models': [],
        'overall_metrics': {}
    }
    
    # Cross-validation loop
    for fold in range(folds):
        logger.info(f"\n{'='*50}\nFold {fold+1}/{folds}\n{'='*50}")
        
        # Prepare dataloaders for this fold
        train_loader, val_loader, train_df, val_df = prepare_loaders(fold)
        
        # Calculate class weights for handling imbalance
        if WEIGHTED_LOSS:
            # Compute class weight
            pos_count = train_df['label'].sum()
            neg_count = len(train_df) - pos_count
            pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            logger.info(f"Class weight for positive class: {pos_weight:.4f}")
        else:
            pos_weight = None
        
        # Initialize model and optimizer
        model = get_model(pretrained=True, dropout_rate=0.2, freeze_backbone=True)
        criterion = get_loss_fn(pos_weight=pos_weight)
        
        # Phase 1: Train only the classifier head
        logger.info("Phase 1: Training classifier head only")
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        best_model_state, best_val_metrics = train_fold(
            model, train_loader, val_loader, optimizer, criterion, fold, logger
        )
        
        # Phase 2: Fine-tune all layers
        logger.info("Phase 2: Fine-tuning all layers")
        model.load_state_dict(best_model_state)
        model.unfreeze_layers()  # Unfreeze the backbone
        
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE/10,
            weight_decay=WEIGHT_DECAY
        )
        
        best_model_state, best_val_metrics = train_fold(
            model, train_loader, val_loader, optimizer, criterion, fold, logger
        )
        
        # Save best model from this fold
        model_save_path = os.path.join(MODEL_SAVE_PATH, f"model_fold_{fold+1}.pth")
        torch.save(best_model_state, model_save_path)
        logger.info(f"Best model for fold {fold+1} saved to {model_save_path}")
        
        # Store results
        results['fold_metrics'].append(best_val_metrics)
        results['fold_models'].append(model_save_path)
        
        # Log fold metrics
        logger.info(f"Fold {fold+1} Results:")
        for metric_name, metric_value in best_val_metrics.items():
            if metric_name != 'confusion_matrix':
                logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # Calculate overall metrics
    accuracy_list = [metrics['accuracy'] for metrics in results['fold_metrics']]
    f1_list = [metrics['f1_score'] for metrics in results['fold_metrics']]
    auc_list = [metrics.get('roc_auc', 0) for metrics in results['fold_metrics']]
    
    results['overall_metrics'] = {
        'mean_accuracy': np.mean(accuracy_list),
        'std_accuracy': np.std(accuracy_list),
        'mean_f1': np.mean(f1_list),
        'std_f1': np.std(f1_list),
        'mean_auc': np.mean(auc_list),
        'std_auc': np.std(auc_list)
    }
    
    # Log overall metrics
    logger.info("\nOverall Cross-Validation Results:")
    logger.info(f"Mean Accuracy: {results['overall_metrics']['mean_accuracy']:.4f} ± {results['overall_metrics']['std_accuracy']:.4f}")
    logger.info(f"Mean F1 Score: {results['overall_metrics']['mean_f1']:.4f} ± {results['overall_metrics']['std_f1']:.4f}")
    logger.info(f"Mean AUC: {results['overall_metrics']['mean_auc']:.4f} ± {results['overall_metrics']['std_auc']:.4f}")
    
    return results


def validate(model, val_loader, criterion):
    """
    Evaluate model on validation data
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        criterion: Loss function
    
    Returns:
        Average loss, accuracy, and dictionaries for predictions and true labels
    """
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1).float()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Update metrics
            losses.update(loss.item(), inputs.size(0))
            
            # Store predictions and labels
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    all_probs = np.array(all_probs).flatten()
    
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    return losses.avg, metrics


def evaluate_on_test(model_paths):
    """
    Evaluate ensemble of models on test dataset
    
    Args:
        model_paths: List of paths to trained model weights
    
    Returns:
        Test metrics
    """
    logger = setup_logger('test_log', os.path.join(LOG_PATH, 'test_evaluation.log'))
    logger.info(f"Evaluating ensemble of {len(model_paths)} models on test data")
    
    # Get test data loader
    test_loader, test_df = get_test_loader()
    
    # Initialize model
    model = get_model(pretrained=False)
    criterion = nn.BCEWithLogitsLoss()
    
    # Initialize lists to store predictions and labels
    all_probs = []
    all_preds = []
    all_labels = []
    
    # For each model in the ensemble
    for i, model_path in enumerate(model_paths):
        logger.info(f"Evaluating model {i+1}/{len(model_paths)}")
        
        # Load model weights
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Evaluate
        fold_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Model {i+1}"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1).float()
                
                # Forward pass
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                
                fold_probs.extend(probs.cpu().numpy())
                
                # Only store labels once
                if i == 0:
                    all_labels.extend(labels.cpu().numpy())
        
        all_probs.append(fold_probs)
    
    # Average predictions from all models
    all_probs = np.mean(np.array(all_probs), axis=0)
    all_preds = (all_probs > 0.5).astype(np.float32)
    
    # Convert to flat arrays
    all_probs = all_probs.flatten()
    all_preds = all_preds.flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    # Log results
    logger.info("\nTest Results:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"AUC: {metrics.get('roc_auc', 0):.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"Sensitivity: {metrics['sensitivity']:.4f}")
    logger.info(f"Specificity: {metrics['specificity']:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Test Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(PLOT_SAVE_PATH, f'test_confusion_matrix.png'))
    plt.close()
    
    return metrics