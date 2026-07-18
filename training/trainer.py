"""
SupplyShield — Local Training Baseline

Trains and evaluates local (non-federated) PyTorch models for each organization.
Used to establish a baseline performance before applying Federated Learning.

Usage:
    python -m training.trainer
"""

import sys
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BATCH_SIZE, LEARNING_RATE, LOCAL_EPOCHS, MODEL_DIR, TARGET_COLUMN, ORG_IDS
from utils import logger, get_device, set_seed
from models.delay_predictor import create_model
from training.dataset import get_dataloaders


def evaluate(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> dict:
    """Evaluate model on a dataloader and return metrics."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    
    total_loss = 0.0
    all_targets = []
    all_logits = []
    
    with torch.no_grad():
        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)
            
            logits = model(features)
            loss = criterion(logits, targets)
            
            total_loss += loss.item() * features.size(0)
            
            all_targets.extend(targets.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader.dataset)
    
    # Calculate metrics
    import numpy as np
    all_targets = np.array(all_targets)
    all_probs = 1 / (1 + np.exp(-np.array(all_logits)))  # sigmoid
    all_preds = (all_probs >= 0.5).astype(int)
    
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_targets, all_preds),
        "precision": precision_score(all_targets, all_preds, zero_division=0),
        "recall": recall_score(all_targets, all_preds, zero_division=0),
        "f1": f1_score(all_targets, all_preds, zero_division=0),
    }
    
    # ROC AUC only if both classes are present
    if len(np.unique(all_targets)) > 1:
        metrics["auc"] = roc_auc_score(all_targets, all_probs)
    else:
        metrics["auc"] = 0.0
        
    return metrics


def train_model(org_id: str, epochs: int = LOCAL_EPOCHS, lr: float = LEARNING_RATE) -> tuple[nn.Module, dict]:
    """Train a local baseline model for a specific organization."""
    logger.info(f"--- Training local baseline for {org_id.upper()} ---")
    
    device = get_device()
    set_seed()
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(org_id)
    
    # Infer input_dim from the first batch
    features, _ = next(iter(train_loader))
    input_dim = features.shape[1]

    # Initialize model
    model = create_model(input_dim=input_dim).to(device)
    
    # Class weights for imbalanced data
    # (In real scenario, calculate from dataset.class_balance, here we'll assume a fixed or compute batch-wise)
    # Using BCEWithLogitsLoss
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_f1 = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, device)
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_state = model.state_dict().copy()
            
    # Load best model for final evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, device)
    logger.info(f"Final Test Metrics for {org_id}: {test_metrics}")
    
    # Save model
    model_path = MODEL_DIR / f"{org_id}_baseline.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved baseline model to {model_path}")
    
    return model, test_metrics


def run_baselines():
    """Run baseline training for all organizations."""
    all_metrics = {}
    
    for org_id in ORG_IDS:
        _, metrics = train_model(org_id)
        all_metrics[org_id] = metrics
        
    # Save metrics
    metrics_path = MODEL_DIR / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
        
    logger.info("=========================================")
    logger.info("Baseline Training Complete")
    logger.info("=========================================")
    for org_id, m in all_metrics.items():
        logger.info(f"{org_id.ljust(15)} | F1: {m['f1']:.4f} | AUC: {m['auc']:.4f} | Acc: {m['accuracy']:.4f}")
        

if __name__ == "__main__":
    run_baselines()
