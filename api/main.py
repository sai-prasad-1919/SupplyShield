"""
SupplyShield — FastAPI Backend

Exposes the trained FL global model for predictions.
Integrates SHAP for explainability and LLM/Rule-based guidance.
"""

import sys
import json
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODEL_DIR, DATA_PROCESSED_DIR, TARGET_COLUMN
from utils import logger, get_device
from models.delay_predictor import create_model
from explainability.shap_explainer import DelayExplainer, generate_explanation_dict
from guidance.llm_agent import get_guidance

# Global state for model and explainers
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and explainer on startup."""
    logger.info("Initializing SupplyShield API...")
    device = get_device()
    
    # 1. Load the latest global model and metadata
    import joblib
    try:
        metadata = joblib.load(DATA_PROCESSED_DIR / "metadata.pkl")
        app_state["feature_names"] = metadata["feature_names"]
        app_state["input_dim"] = len(metadata["feature_names"])
    except FileNotFoundError:
        app_state["feature_names"] = ["distance_km", "weather", "traffic", "vehicle_type"]
        app_state["input_dim"] = 4
        
    model_path = MODEL_DIR / "global_model_latest.pt"
    if not model_path.exists():
        logger.warning(f"Global model not found at {model_path}. Using untrained model.")
        model = create_model(input_dim=app_state["input_dim"]).to(device)
    else:
        model = create_model(input_dim=app_state["input_dim"]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        
    model.eval()
    app_state["model"] = model
    app_state["device"] = device
    
    # 2. Initialize SHAP Explainer (Needs background data)
    cleaned_path = DATA_PROCESSED_DIR / "cleaned.csv"
    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)
        
        # We use feature names from metadata, no need to compute from CSV
        feature_names = app_state["feature_names"]
        
        # For the background, we need scaled/encoded data.
        # Ideally, we load this from a partition. Let's use novamart's test set.
        try:
            from training.dataset import get_dataloaders
            loaders = get_dataloaders("novamart")
            bg_data = []
            for features, _ in loaders["test"]:
                bg_data.append(features)
            bg_tensor = torch.cat(bg_data, dim=0)[:100].to(device)
            
            app_state["explainer"] = DelayExplainer(model, bg_tensor, feature_names)
            app_state["feature_names"] = feature_names
            # Calculate base value (expected value over background)
            with torch.no_grad():
                bg_preds = model(bg_tensor)
                bg_probs = torch.sigmoid(bg_preds).cpu().numpy()
                app_state["base_value"] = float(np.mean(bg_probs))
        except Exception as e:
            logger.error(f"Failed to initialize SHAP: {e}")
            app_state["explainer"] = None
    else:
        logger.warning("Cleaned data not found. SHAP explainer disabled.")
        app_state["explainer"] = None
        
    yield
    # Cleanup on shutdown
    app_state.clear()


app = FastAPI(title="SupplyShield API", lifespan=lifespan)

# Allow CORS for UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    # Matches the model input features (distance, weather, traffic, vehicle_type)
    # They should be passed as scaled/encoded floats for simplicity in this demo API
    features: list[float]
    
class PredictionResponse(BaseModel):
    prediction: dict
    explainability: dict | None = None
    guidance: list[str]


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": "model" in app_state}


@app.post("/predict", response_model=PredictionResponse)
def predict_delay(request: PredictionRequest):
    """Predict delay risk, generate SHAP explanation, and provide guidance."""
    model = app_state.get("model")
    device = app_state.get("device")
    explainer = app_state.get("explainer")
    feature_names = app_state.get("feature_names", ["F1", "F2", "F3", "F4"])
    base_value = app_state.get("base_value", 0.5)
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
        
    input_dim = app_state.get("input_dim", 4)
    if len(request.features) != input_dim:
        raise HTTPException(status_code=400, detail=f"Expected {input_dim} features, got {len(request.features)}")
        
    # Prepare input
    features_np = np.array([request.features], dtype=np.float32)
    features_tensor = torch.tensor(features_np).to(device)
    
    # 1. Prediction
    with torch.no_grad():
        logit = model(features_tensor)
        prob = torch.sigmoid(logit).item()
        
    is_delayed = bool(prob >= 0.5)
    prediction_dict = {
        "delay_probability": prob,
        "is_delayed": is_delayed
    }
    
    # 2. Explainability
    explanation_dict = {"prediction": prediction_dict, "explainability": None}
    
    if explainer is not None:
        shap_values = explainer.explain_instances(features_tensor)
        # shap_values should be shape (1, num_features)
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
            
        explanation_dict = generate_explanation_dict(
            feature_names=feature_names,
            feature_values=features_np[0],
            shap_values=shap_values,
            base_value=base_value,
            prediction_prob=prob
        )
        
    # 3. Guidance
    guidance = get_guidance(explanation_dict)
    
    return PredictionResponse(
        prediction=explanation_dict["prediction"],
        explainability=explanation_dict["explainability"],
        guidance=guidance
    )
