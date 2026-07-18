import os
import pickle
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel
from typing import List, Dict, Any

# Ensure we're in the correct root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")

app = FastAPI(title="SupplyShield API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# ----------------- Globals & Loading -----------------
model = None
label_encoders = {}
shap_data = {}
scaler = None

@app.on_event("startup")
def load_assets():
    global model, label_encoders, shap_data
    print("Loading model and encoders...")
    
    # Load Model
    model_path = os.path.join(CHECKPOINTS_DIR, "federated_global_best.pt")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = DelayPredictor(input_dim=checkpoint['input_dim'], 
                           hidden_dims=checkpoint['hidden_dims'], 
                           dropout_rate=checkpoint['dropout_rate'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load Encoders
    encoders_path = os.path.join(CHECKPOINTS_DIR, "label_encoders.pkl")
    if os.path.exists(encoders_path):
        import joblib
        label_encoders = joblib.load(encoders_path)
        
    # Load Scaler
    scaler_path = os.path.join(CHECKPOINTS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        import joblib
        global scaler
        scaler = joblib.load(scaler_path)
            
    # Load SHAP data
    shap_path = os.path.join(CHECKPOINTS_DIR, "shap_data.pkl")
    if os.path.exists(shap_path):
        import joblib
        shap_data = joblib.load(shap_path)
    
    print("Assets loaded successfully.")

# ----------------- API Endpoints -----------------

class PredictionRequest(BaseModel):
    package_type: str
    vehicle_type: str
    delivery_mode: str
    region: str
    weather_condition: str
    distance_km: float
    package_weight_kg: float
    delivery_time_hours: float
    expected_time_hours: float
    delivery_rating: float
    delivery_cost: float
    
@app.get("/api/stats")
def get_stats():
    return {
        "status": "active",
        "model_type": "Federated Global Model (FedAvg)",
        "metrics": {
            "f1_score": 0.9909,
            "auc_roc": 1.0000,
            "accuracy": 0.99
        },
        "supported_features": 12,
        "processed_organizations": 3
    }

@app.get("/api/shap")
def get_shap():
    if not shap_data:
        return {"error": "SHAP data not available"}
    
    # Format SHAP data for charting
    # Convert numpy arrays to lists
    feature_names = shap_data.get('feature_names', [])
    shap_values = shap_data.get('shap_values', [])
    
    # Calculate mean absolute SHAP per feature across all samples
    import numpy as np
    
    try:
        # shap_values could be a list of arrays (if classification) or a single array
        if isinstance(shap_values, list):
            # Take the array corresponding to the positive class if binary
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values
            
        mean_abs_shap = np.abs(sv).mean(axis=0).tolist()
        
        # Create a sorted list of dictionaries for the frontend Recharts
        importance = [{"name": name, "value": float(val)} for name, val in zip(feature_names, mean_abs_shap)]
        importance.sort(key=lambda x: x["value"], reverse=True)
        
        return {"feature_importance": importance}
    except Exception as e:
        return {"error": f"Failed to process SHAP data: {str(e)}"}

@app.post("/api/predict")
def predict_delay(req: PredictionRequest):
    global model, label_encoders
    
    try:
        # Engineered feature
        time_diff_hours = req.delivery_time_hours - req.expected_time_hours
        
        # Categorical Encoding
        def encode(feature_name, value):
            if feature_name in label_encoders:
                le = label_encoders[feature_name]
                if value in le.classes_:
                    return float(le.transform([value])[0])
                # Unseen label fallback
                return 0.0
            return 0.0

        pkg_encoded = encode('package_type', req.package_type)
        veh_encoded = encode('vehicle_type', req.vehicle_type)
        mode_encoded = encode('delivery_mode', req.delivery_mode)
        reg_encoded = encode('region', req.region)
        weather_encoded = encode('weather_condition', req.weather_condition)
        
        # Feature Vector exactly matching the 12 features during training
        features = [
            pkg_encoded,
            veh_encoded,
            mode_encoded,
            reg_encoded,
            weather_encoded,
            req.distance_km,
            req.package_weight_kg,
            req.delivery_time_hours,
            req.expected_time_hours,
            req.delivery_rating,
            req.delivery_cost,
            time_diff_hours
        ]
        
        tensor_features = torch.tensor([features], dtype=torch.float32)
        
        if scaler is not None:
            import numpy as np
            # Scaler expects 2D array
            features_scaled = scaler.transform(np.array([features]))
            tensor_features = torch.tensor(features_scaled, dtype=torch.float32)
            
        with torch.no_grad():
            prob = model(tensor_features).item()
            
        is_delayed = prob > 0.5
        
        return {
            "delay_probability": prob,
            "prediction": "Delayed" if is_delayed else "On-Time",
            "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low",
            "features_used": features
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
