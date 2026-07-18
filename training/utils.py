"""
SupplyShield - Shared Utilities for Training
Contains data loading, training loop, and evaluation functions.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, classification_report, roc_auc_score,
    confusion_matrix, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prepare_splits(csv_path, target_col='delayed', test_size=0.15, val_size=0.15):
    """Load data and create train/val/test splits with scaling."""
    df = pd.read_csv(csv_path)

    X = df.drop(columns=[target_col]).values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Second split: separate validation from training
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )

    # Scale features (fit on train ONLY, transform all)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert to float32
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Train delay rate: {y_train.mean():.2%}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def train_model(
    X_train, y_train, X_val, y_val,
    input_dim,
    hidden_dims=None,
    dropout_rate=0.3,
    lr=0.001,
    epochs=200,
    batch_size=64,
    patience=15,
    model_save_path='checkpoints/best_model.pt',
    verbose=True,
):
    """Train with early stopping, LR scheduling, and validation monitoring."""
    from training.model import DelayPredictor

    if hidden_dims is None:
        hidden_dims = [128, 64, 32]

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Handle class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    if verbose:
        print(f"  Class weight (pos_weight): {pos_weight:.2f}")

    # DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model, Loss, Optimizer, Scheduler
    model = DelayPredictor(input_dim, hidden_dims, dropout_rate)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=7, factor=0.5
    )

    # Tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_acc': [], 'val_acc': []
    }
    best_val_loss = float('inf')
    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(epochs):
        # === TRAINING ===
        model.train()
        train_losses, train_preds, train_labels = [], [], []

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            preds = (outputs > 0.5).int().squeeze().tolist()
            labels = batch_y.int().squeeze().tolist()
            if isinstance(preds, int):
                preds = [preds]
                labels = [labels]
            train_preds.extend(preds)
            train_labels.extend(labels)

        train_loss = np.mean(train_losses)
        train_f1 = f1_score(train_labels, train_preds, zero_division=0)
        train_acc = sum(1 for a, b in zip(train_preds, train_labels) if a == b) / len(train_labels)

        # === VALIDATION ===
        model.eval()
        val_losses, val_preds, val_labels = [], [], []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())
                preds = (outputs > 0.5).int().squeeze().tolist()
                labels = batch_y.int().squeeze().tolist()
                if isinstance(preds, int):
                    preds = [preds]
                    labels = [labels]
                val_preds.extend(preds)
                val_labels.extend(labels)

        val_loss = np.mean(val_losses)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_acc = sum(1 for a, b in zip(val_preds, val_labels) if a == b) / len(val_labels)

        # Record
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        scheduler.step(val_loss)

        # Print progress
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | "
                  f"LR: {current_lr:.6f}")

        # === EARLY STOPPING ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_f1': val_f1,
                'input_dim': input_dim,
                'hidden_dims': hidden_dims,
                'dropout_rate': dropout_rate,
            }, model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"\n  [STOP] Early stopping at epoch {epoch+1}. "
                          f"Best val_loss: {best_val_loss:.4f}, Best val_f1: {best_val_f1:.4f}")
                break

    # Load best model
    checkpoint = torch.load(model_save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if verbose:
        print(f"  [OK] Loaded best model from epoch {checkpoint['epoch']+1}")

    return model, history


def evaluate_model(model, X_test, y_test, title="Model Evaluation", save_dir=None):
    """Complete evaluation with all metrics and visualizations."""
    if save_dir is None:
        save_dir = os.path.join(ROOT, 'evaluation')
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_probs = model(X_tensor).squeeze().numpy()
        y_preds = (y_probs > 0.5).astype(int)

    # Metrics
    print(f"\n  {'='*50}")
    print(f"  {title}")
    print(f"  {'='*50}")
    print(classification_report(y_test, y_preds, target_names=['On-Time', 'Delayed']))

    f1 = f1_score(y_test, y_preds)
    try:
        auc = roc_auc_score(y_test, y_probs)
    except ValueError:
        auc = 0.0

    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")

    # Confusion Matrix + ROC
    cm = confusion_matrix(y_test, y_preds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['On-Time', 'Delayed'],
                yticklabels=['On-Time', 'Delayed'])
    axes[0].set_title(f'{title} - Confusion Matrix')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')

    try:
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        axes[1].plot(fpr, tpr, 'b-', label=f'AUC = {auc:.4f}')
        axes[1].plot([0, 1], [0, 1], 'r--', label='Random')
        axes[1].set_title(f'{title} - ROC Curve')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].legend()
    except Exception:
        axes[1].text(0.5, 0.5, 'ROC unavailable', ha='center', va='center')

    plt.tight_layout()
    fname = title.replace(" ", "_").lower()
    plt.savefig(os.path.join(save_dir, f'{fname}.png'), dpi=150)
    plt.close()
    print(f"  Saved plot: evaluation/{fname}.png")

    return {'f1': f1, 'auc': auc, 'confusion_matrix': cm}


def plot_training_history(history, title="Training History", save_dir=None):
    """Plot train vs val curves to diagnose under/overfitting."""
    if save_dir is None:
        save_dir = os.path.join(ROOT, 'evaluation')
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['val_loss'], label='Val Loss', color='red')
    axes[0].set_title('Loss Curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history['train_f1'], label='Train F1', color='blue')
    axes[1].plot(history['val_f1'], label='Val F1', color='red')
    axes[1].set_title('F1 Score Curves')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1')
    axes[1].legend()

    axes[2].plot(history['train_acc'], label='Train Acc', color='blue')
    axes[2].plot(history['val_acc'], label='Val Acc', color='red')
    axes[2].set_title('Accuracy Curves')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend()

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = title.replace(" ", "_").lower()
    plt.savefig(os.path.join(save_dir, f'{fname}.png'), dpi=150)
    plt.close()
    print(f"  Saved plot: evaluation/{fname}.png")
