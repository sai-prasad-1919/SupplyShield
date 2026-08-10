import os
import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from training.utils import prepare_splits
from models.delay_predictor import DelayPredictor

def recalibrate():
    print("Recalibrating Global Model Threshold...")
    
    # 1. Load Data & Prepare Splits (Fits scaler ONLY on Train!)
    csv_path = 'data/processed/combined.csv'
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_splits(csv_path)
    
    # Save the correct scaler
    os.makedirs('checkpoints', exist_ok=True)
    joblib.dump(scaler, 'checkpoints/scaler.pkl')
    print("Saved scaler (fitted only on train) to checkpoints/scaler.pkl")
    
    # 2. Load the global federated model (FL architecture: hidden_dim=64)
    model_path = 'checkpoints/federated_global_best.pt'
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    
    input_dim  = checkpoint['input_dim']
    hidden_dim = checkpoint.get('hidden_dim', 64)
    dropout    = checkpoint.get('dropout', 0.2)
    
    model = DelayPredictor(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded FL model: input_dim={input_dim}, hidden_dim={hidden_dim}")
    
    # 3. Predict on Validation Set
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(model(X_val_tensor)).numpy().squeeze()
        
    # 4. Youden's J statistic to find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_val, probs)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds[optimal_idx])
    
    print(f"Optimal Threshold (Youden's J): {optimal_threshold:.4f}")
    
    # 5. Save threshold
    joblib.dump(optimal_threshold, 'checkpoints/threshold.pkl')
    print("Saved optimal threshold to checkpoints/threshold.pkl")

if __name__ == "__main__":
    recalibrate()

