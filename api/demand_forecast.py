"""
SupplyShield — LSTM Demand Forecast API

Inference logic for generating multi-week demand forecasts per organization.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHECKPOINTS_DIR, PROJECT_ROOT
from utils import logger, get_device
from models.lstm_forecaster import DemandLSTM

# Need 8 weeks of context to predict next week
SEQ_LENGTH = 8

def load_forecaster(org_id: str):
    device = get_device()
    model_path = CHECKPOINTS_DIR / f"{org_id}_lstm.pt"
    scaler_path = CHECKPOINTS_DIR / f"{org_id}_lstm_scaler.pkl"
    
    if not model_path.exists() or not scaler_path.exists():
        logger.error(f"Missing LSTM checkpoint or scaler for {org_id}")
        return None, None
        
    model = DemandLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    scaler = joblib.load(scaler_path)
    return model, scaler

def get_recent_history(org_id: str) -> np.ndarray:
    """Load the most recent SEQ_LENGTH weeks of sales data for the org."""
    train_path = PROJECT_ROOT / "data" / "raw" / "train.csv"
    if not train_path.exists():
        return np.zeros((SEQ_LENGTH, 1))
        
    df = pd.read_csv(train_path)
    df["Date"] = pd.to_datetime(df["Date"])
    
    org_store_map = {
        "novamart": list(range(1, 16)),
        "titanelec": list(range(16, 31)),
        "swiftlog": list(range(31, 46))
    }
    
    store_ids = org_store_map.get(org_id, [])
    org_df = df[df["Store"].isin(store_ids)].copy()
    org_df = org_df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    org_df = org_df.sort_values("Date")
    
    sales = org_df["Weekly_Sales"].values
    if len(sales) < SEQ_LENGTH:
        # Pad with mean if not enough history
        mean_val = np.mean(sales) if len(sales) > 0 else 0.0
        padding = np.full((SEQ_LENGTH - len(sales),), mean_val)
        sales = np.concatenate([padding, sales])
        
    return sales[-SEQ_LENGTH:].reshape(-1, 1)

def forecast_demand(org_id: str, n_weeks: int = 4) -> dict:
    """Generate future demand forecast for n_weeks."""
    model, scaler = load_forecaster(org_id)
    if model is None:
        return {"org": org_id, "error": "Model not loaded"}
        
    device = get_device()
    
    # Get last SEQ_LENGTH weeks of actual data
    recent_data = get_recent_history(org_id)
    
    # Scale
    recent_scaled = scaler.transform(recent_data)
    
    current_seq = torch.tensor(recent_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    
    forecasts_scaled = []
    
    with torch.no_grad():
        for _ in range(n_weeks):
            out = model(current_seq)
            forecasts_scaled.append(out.item())
            
            # Slide window: remove first element, append new prediction
            new_val = out.unsqueeze(1)
            current_seq = torch.cat([current_seq[:, 1:, :], new_val], dim=1)
            
    # Inverse transform
    forecasts = scaler.inverse_transform(np.array(forecasts_scaled).reshape(-1, 1)).flatten()
    
    return {
        "org": org_id,
        "forecast": [float(f) for f in forecasts],
        "weeks_ahead": n_weeks
    }
