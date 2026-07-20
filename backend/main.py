import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from typing import Optional

from backend.guidance.rules import evaluate_rules
from backend.auth.routes import router as auth_router, get_current_org

# Ensure we're in the correct root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")
DATA_PARTITIONS_DIR = os.path.join(ROOT_DIR, "data", "partitions")

app = FastAPI(title="SupplyShield API v2", version="2.0.0")

# Include auth routes
app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Configuration -----------------
ORG_CONFIG = {
    "novamart": {"display_name": "Novamart Retail", "type": "Retail", "prefix": "NVM"},
    "titanelec": {"display_name": "TitanElec Manufacturing", "type": "Manufacturing", "prefix": "TIT"},
    "swiftlog": {"display_name": "SwiftLog Warehouse", "type": "Logistics", "prefix": "SWL"}
}

# ----------------- Model Architecture -----------------
class DelayPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, dropout_rate=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ----------------- Globals -----------------
model = None
label_encoders = {}
scaler = None
global_threshold = 0.5
org_test_data = {}

@app.on_event("startup")
def load_assets():
    global model, label_encoders, scaler, global_threshold, org_test_data
    print("Loading assets for SupplyShield v2...")

    model_path = os.path.join(CHECKPOINTS_DIR, "federated_global_best.pt")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model = DelayPredictor(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims'],
        dropout_rate=checkpoint['dropout_rate']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    import joblib
    encoders_path = os.path.join(CHECKPOINTS_DIR, "label_encoders.pkl")
    if os.path.exists(encoders_path):
        label_encoders = joblib.load(encoders_path)

    scaler_path = os.path.join(CHECKPOINTS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    thresh_path = os.path.join(CHECKPOINTS_DIR, "threshold.pkl")
    if os.path.exists(thresh_path):
        global_threshold = joblib.load(thresh_path)

    for org in ORG_CONFIG.keys():
        csv_path = os.path.join(DATA_PARTITIONS_DIR, org, "data.csv")
        if os.path.exists(csv_path):
            org_test_data[org] = pd.read_csv(csv_path)

    print("Assets loaded successfully.")

# ----------------- Inference -----------------
def predict_row(row: pd.Series):
    decoded_features = {}
    encoded_features = []

    feature_cols = ['package_type', 'vehicle_type', 'delivery_mode', 'region',
                    'weather_condition', 'distance_km', 'package_weight_kg',
                    'delivery_time_hours', 'expected_time_hours', 'delivery_rating',
                    'delivery_cost', 'time_diff_hours']

    for col in feature_cols:
        val = row[col]
        if col in label_encoders:
            try:
                le = label_encoders[col]
                text_val = le.inverse_transform([int(val)])[0]
                decoded_features[col] = text_val
                encoded_features.append(float(val))
            except:
                decoded_features[col] = "unknown"
                encoded_features.append(0.0)
        else:
            decoded_features[col] = val
            encoded_features.append(float(val))

    tensor_features = torch.tensor([encoded_features], dtype=torch.float32)
    if scaler is not None:
        features_scaled = scaler.transform(np.array([encoded_features]))
        tensor_features = torch.tensor(features_scaled, dtype=torch.float32)

    with torch.no_grad():
        prob = model(tensor_features).item()

    is_delayed = prob > global_threshold
    risk_level = "High" if prob > (global_threshold + 0.15) else "Medium" if prob > (global_threshold - 0.1) else "Low"
    shipment_state = {**decoded_features, "delay_probability": prob, "risk_level": risk_level}

    return {
        "delay_probability": prob,
        "prediction": "Delayed" if is_delayed else "On-Time",
        "risk_level": risk_level,
        "features": decoded_features,
        "shipment_state": shipment_state
    }


def get_real_shipments(org_key: str):
    if org_key not in org_test_data:
        return []

    df = org_test_data[org_key]
    df_sample = df.sample(n=30, random_state=42).reset_index()
    prefix = ORG_CONFIG[org_key]["prefix"]

    shipments = []
    for idx, row in df_sample.iterrows():
        real_idx = int(row['index'])
        pred_res = predict_row(row)

        dest_map = {"north": "Delhi", "south": "Bangalore", "east": "Kolkata", "west": "Mumbai", "central": "Nagpur"}
        dest = dest_map.get(pred_res["features"]["region"], "Hub")
        guidance = evaluate_rules(pred_res["shipment_state"])

        shipments.append({
            "id": f"{prefix}-{real_idx:04d}",
            "route": f"Supplier to {dest}",
            "package_type": pred_res["features"]["package_type"].title(),
            "prediction": {
                "delay_probability": pred_res["delay_probability"],
                "prediction": pred_res["prediction"],
                "risk_level": pred_res["risk_level"]
            },
            "features": pred_res["features"],
            "guidance": {
                "text": guidance["message"],
                "rule_fired": guidance["reason"]
            }
        })

    risk_order = {"High": 0, "Medium": 1, "Low": 2}
    shipments.sort(key=lambda x: (risk_order[x["prediction"]["risk_level"]], -x["prediction"]["delay_probability"]))
    return shipments


# ----------------- Public Endpoints -----------------
@app.get("/api/orgs")
def get_orgs():
    return [{"id": k, **v} for k, v in ORG_CONFIG.items()]


# ----------------- JWT-Protected Endpoints -----------------
@app.get("/api/kpis")
def get_kpis(current_org: dict = Depends(get_current_org)):
    org_key = current_org.get("org_key", "")
    if not org_key or org_key not in org_test_data:
        raise HTTPException(status_code=404, detail="Org data not found")

    shipments = get_real_shipments(org_key)
    high_risk_count = sum(1 for s in shipments if s["prediction"]["risk_level"] == "High")
    delayed_shipments = [s for s in shipments if s["prediction"]["risk_level"] in ["High", "Medium"]]
    total_delay = sum(s["features"]["time_diff_hours"] for s in delayed_shipments if s["features"]["time_diff_hours"] > 0)
    avg_delay_hours = (total_delay / len(delayed_shipments)) if delayed_shipments else 0

    if avg_delay_hours > 24:
        delay_str = f"{avg_delay_hours / 24:.1f} days"
    else:
        delay_str = f"{avg_delay_hours:.1f} hours"

    return {
        "high_risk_count": high_risk_count,
        "avg_delay": delay_str,
        "units_in_transit": f"{len(shipments) * 450:,}"
    }


@app.get("/api/shipments")
def get_shipments_api(current_org: dict = Depends(get_current_org)):
    org_key = current_org.get("org_key", "")
    return get_real_shipments(org_key)


@app.get("/api/shipment/{shipment_id}")
def get_shipment_api(shipment_id: str, current_org: dict = Depends(get_current_org)):
    org_key = current_org.get("org_key", "")
    shipments = get_real_shipments(org_key)
    for s in shipments:
        if s["id"] == shipment_id:
            return s
    raise HTTPException(status_code=404, detail="Shipment not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
