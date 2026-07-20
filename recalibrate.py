import os
import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from training.utils import prepare_splits
from training.model import DelayPredictor

def recalibrate():
    print("Recalibrating Global Model Threshold...")
    
    # 1. Load Data & Prepare Splits (Fits scaler ONLY on Train!)
    csv_path = 'data/processed/combined.csv'
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_splits(csv_path)
    
    # Save the correct scaler to replace the leaked one
    os.makedirs('checkpoints', exist_ok=True)
    joblib.dump(scaler, 'checkpoints/scaler.pkl')
    print("Saved fixed scaler (fitted only on train) to checkpoints/scaler.pkl")
    
    # 2. Load the global federated model
    model_path = 'checkpoints/federated_global_best.pt'
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    
    model = DelayPredictor(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims'],
        dropout_rate=checkpoint['dropout_rate']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. Predict on Validation Set
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    with torch.no_grad():
        # The model outputs probabilities directly because of the Sigmoid layer
        probs = model(X_val_tensor).numpy().squeeze()
        
    # 4. Calculate Youden's J statistic to find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_val, probs)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal Threshold (Youden's J): {optimal_threshold:.4f}")
    
    # 5. Save threshold
    joblib.dump(float(optimal_threshold), 'checkpoints/threshold.pkl')
    print("Saved optimal threshold to checkpoints/threshold.pkl")

if __name__ == "__main__":
    recalibrate()
