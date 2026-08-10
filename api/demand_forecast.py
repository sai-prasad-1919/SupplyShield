"""
SupplyShield — LSTM Demand Forecast API

Inference logic for generating multi-week demand forecasts per organization.
Returns real historical weeks alongside the predicted forecast, both with explicit dates.
"""

import sys
from pathlib import Path
from datetime import timedelta

import torch
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHECKPOINTS_DIR, PROJECT_ROOT
from utils import logger, get_device
from models.lstm_forecaster import DemandLSTM

SEQ_LENGTH = 8          # weeks of context fed into the LSTM
HISTORY_WEEKS = 12      # weeks of real history to return alongside forecast

ORG_STORE_MAP = {
    "novamart":  list(range(1, 16)),
    "titanelec": list(range(16, 31)),
    "swiftlog":  list(range(31, 46)),
}

def load_forecaster(org_id: str):
    device = get_device()
    model_path  = CHECKPOINTS_DIR / f"{org_id}_lstm.pt"
    scaler_path = CHECKPOINTS_DIR / f"{org_id}_lstm_scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        logger.error(f"Missing LSTM checkpoint or scaler for {org_id}")
        return None, None

    model = DemandLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    scaler = joblib.load(scaler_path)
    return model, scaler


def _load_org_series(org_id: str) -> pd.DataFrame:
    """Return a DateFrame with columns [Date, Weekly_Sales] for the org's stores, sorted by date."""
    train_path = PROJECT_ROOT / "data" / "raw" / "train.csv"
    if not train_path.exists():
        return pd.DataFrame(columns=["Date", "Weekly_Sales"])

    store_ids = ORG_STORE_MAP.get(org_id, [])
    df = pd.read_csv(train_path)
    df["Date"] = pd.to_datetime(df["Date"])
    org_df = df[df["Store"].isin(store_ids)].groupby("Date")["Weekly_Sales"].sum().reset_index()
    return org_df.sort_values("Date").reset_index(drop=True)


def forecast_demand(org_id: str, n_weeks: int = 4) -> dict:
    """
    Generate a demand forecast for n_weeks ahead, together with HISTORY_WEEKS
    of real historical data. All entries carry explicit ISO date strings.

    Returns:
        {
            "org": str,
            "weeks_ahead": int,
            "history": [{"week": str, "sales": float}, ...],   # HISTORY_WEEKS entries
            "forecast": [{"week": str, "sales": float}, ...],  # n_weeks entries
            "forecast_start_date": str,                        # ISO date of first forecast week
            "model_info": {...}
        }
    """
    model, scaler = load_forecaster(org_id)
    if model is None:
        return {"org": org_id, "error": "LSTM model not loaded — run training/09_train_lstm.py first"}

    device = get_device()
    org_series = _load_org_series(org_id)

    if len(org_series) < SEQ_LENGTH:
        return {"org": org_id, "error": f"Insufficient data for {org_id}"}

    sales_arr = org_series["Weekly_Sales"].values
    dates_arr = org_series["Date"].values  # numpy datetime64

    # ── History window (last HISTORY_WEEKS weeks of real data) ──────────────
    history_slice = org_series.tail(HISTORY_WEEKS)
    history = [
        {"week": pd.Timestamp(row["Date"]).strftime("%Y-%m-%d"), "sales": float(row["Weekly_Sales"])}
        for _, row in history_slice.iterrows()
    ]

    # ── Seed sequence for the LSTM (last SEQ_LENGTH real weeks) ─────────────
    seed_sales = sales_arr[-SEQ_LENGTH:].reshape(-1, 1)
    seed_scaled = scaler.transform(seed_sales)
    current_seq = torch.tensor(seed_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    # ── Forecast ─────────────────────────────────────────────────────────────
    forecasts_scaled = []
    with torch.no_grad():
        for _ in range(n_weeks):
            out = model(current_seq)
            forecasts_scaled.append(out.item())
            new_val = out.unsqueeze(1)
            current_seq = torch.cat([current_seq[:, 1:, :], new_val], dim=1)

    forecasts = scaler.inverse_transform(np.array(forecasts_scaled).reshape(-1, 1)).flatten()

    # Forecast dates: weekly offsets from the last historical date
    last_date = pd.Timestamp(dates_arr[-1])
    forecast_entries = [
        {
            "week": (last_date + timedelta(weeks=i + 1)).strftime("%Y-%m-%d"),
            "sales": float(forecasts[i]),
        }
        for i in range(n_weeks)
    ]

    forecast_start_date = forecast_entries[0]["week"]

    return {
        "org": org_id,
        "weeks_ahead": n_weeks,
        "history": history,
        "forecast": forecast_entries,
        "forecast_start_date": forecast_start_date,
        "model_info": {
            "type": "LSTM",
            "layers": 2,
            "hidden_units": 64,
            "window_weeks": SEQ_LENGTH,
            "training": (
                "Independently trained per-org on simulated Walmart weekly sales partitions. "
                f"Stores {min(ORG_STORE_MAP.get(org_id, [0]))}-{max(ORG_STORE_MAP.get(org_id, [0]))} → {org_id}. "
                "NOT a federated model."
            ),
        },
    }
