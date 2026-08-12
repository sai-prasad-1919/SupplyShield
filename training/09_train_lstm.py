"""
SupplyShield — LSTM Training Script

Trains LSTM demand forecasting models per organization based on Walmart sales data.
"""

import sys
from pathlib import Path
import json

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROJECT_ROOT, CHECKPOINTS_DIR
from utils import logger, set_seed, get_device
from models.lstm_forecaster import DemandLSTM

# We want to use past N weeks to predict the next week
SEQ_LENGTH = 8

def create_sequences(data: np.ndarray, seq_length: int):
    """Create input sequences and corresponding targets."""
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

def train_org_lstm(org_id: str, store_ids: list[int], df: pd.DataFrame, device: torch.device):
    logger.info(f"--- Training LSTM for {org_id.upper()} (Stores {min(store_ids)}-{max(store_ids)}) ---")
    
    # Filter data for the assigned stores
    org_df = df[df["Store"].isin(store_ids)].copy()
    
    # Aggregate weekly sales across assigned stores
    org_df = org_df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    org_df = org_df.sort_values("Date")
    
    # Extract target values
    sales = org_df["Weekly_Sales"].values.reshape(-1, 1)
    
    if len(sales) <= SEQ_LENGTH:
        logger.warning(f"Not enough data for {org_id}. Skipping.")
        return
        
    # Scale data
    scaler = MinMaxScaler()
    sales_scaled = scaler.fit_transform(sales)
    
    # Save scaler
    joblib.dump(scaler, CHECKPOINTS_DIR / f"{org_id}_lstm_scaler.pkl")
    
    # Create sequences
    X, y = create_sequences(sales_scaled, SEQ_LENGTH)
    
    # Split into train/test (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)
    
    # Initialize model
    model = DemandLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Train
    epochs = 50
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                out = model(batch_x)
                loss = criterion(out, batch_y)
                val_loss += loss.item()
                
        val_loss /= len(test_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()
            
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    # Save checkpoint
    torch.save(model.state_dict(), CHECKPOINTS_DIR / f"{org_id}_lstm.pt")
    logger.info(f"Saved {org_id}_lstm.pt. Best Val MSE: {best_loss:.6f}")
    
def main():
    set_seed()
    device = get_device()
    
    # Load dataset
    train_path = PROJECT_ROOT / "data" / "raw" / "train.csv"
    if not train_path.exists():
        logger.error(f"Missing Walmart dataset at {train_path}")
        sys.exit(1)
        
    df = pd.read_csv(train_path)
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Map stores to orgs (simulated assignment)
    org_store_map = {
        "novamart": list(range(1, 16)),
        "titanelec": list(range(16, 31)),
        "swiftlog": list(range(31, 46))
    }
    
    for org_id, store_ids in org_store_map.items():
        train_org_lstm(org_id, store_ids, df, device)
        
if __name__ == "__main__":
    main()
