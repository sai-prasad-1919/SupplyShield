"""
SupplyShield — SHAP Stability Analysis

Evaluates how feature attribution (SHAP values) evolves across Federated Learning rounds.
Generates summary plots for the baseline models and the final global model.

Usage:
    python -m evaluation.shap_stability
"""

import sys
from pathlib import Path
import json

import torch
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODEL_DIR, TARGET_COLUMN, DATA_PROCESSED_DIR
from utils import logger, get_device, set_seed
from models.delay_predictor import create_model
from training.dataset import get_dataloaders
from explainability.shap_explainer import DelayExplainer


def load_model(path: Path, input_dim: int) -> torch.nn.Module:
    """Load a model from a checkpoint."""
    device = get_device()
    model = create_model(input_dim=input_dim).to(device)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    else:
        logger.warning(f"Model not found at {path}. Returning untrained model.")
    return model


def run_analysis():
    """Run SHAP stability analysis."""
    logger.info("=" * 60)
    logger.info("  SupplyShield — SHAP Stability Analysis")
    logger.info("=" * 60)
    
    set_seed()
    device = get_device()
    
    # We will use the test set from one of the orgs as the representative distribution
    # Let's use NovaMart's test set for evaluation
    org_id = "novamart"
    loaders = get_dataloaders(org_id)
    test_loader = loaders["test"]
    
    # Load metadata for feature names and input_dim
    import joblib
    try:
        metadata = joblib.load(DATA_PROCESSED_DIR / "metadata.pkl")
        feature_names = metadata["feature_names"]
        input_dim = len(feature_names)
    except FileNotFoundError:
        feature_names = ["distance_km", "weather", "traffic", "vehicle_type"]
        input_dim = 4

    # Extract background data
    all_features = []
    for features, _ in test_loader:
        all_features.append(features)
    
    X_test = torch.cat(all_features, dim=0)
    bg_size = min(500, len(X_test))
    background_data = X_test[:bg_size].to(device)
    
    eval_start = min(500, len(X_test))
    eval_end = min(1000, len(X_test))
    if eval_start == eval_end:
        logger.warning("Not enough data for separate eval set, reusing background data.")
        eval_data = background_data
    else:
        eval_data = X_test[eval_start:eval_end].to(device)
    
    
    # 1. Evaluate baseline model
    logger.info(f"\n--- Analyzing Local Baseline ({org_id.upper()}) ---")
    baseline_path = MODEL_DIR / f"{org_id}_baseline.pt"
    baseline_model = load_model(baseline_path, input_dim)
    
    baseline_explainer = DelayExplainer(baseline_model, background_data, feature_names)
    baseline_shap = baseline_explainer.explain_instances(eval_data)
    
    baseline_importance = baseline_explainer.get_feature_importance(baseline_shap)
    logger.info(f"Baseline Feature Importance: {baseline_importance}")
    
    baseline_explainer.plot_summary(
        eval_data, 
        baseline_shap, 
        output_path=str(MODEL_DIR / "shap_summary_baseline.png")
    )
    
    # 2. Evaluate global FL model
    logger.info("\n--- Analyzing Global FL Model ---")
    global_path = MODEL_DIR / "global_model_latest.pt"
    global_model = load_model(global_path, input_dim)
    
    global_explainer = DelayExplainer(global_model, background_data, feature_names)
    global_shap = global_explainer.explain_instances(eval_data)
    
    global_importance = global_explainer.get_feature_importance(global_shap)
    logger.info(f"Global FL Feature Importance: {global_importance}")
    
    global_explainer.plot_summary(
        eval_data, 
        global_shap, 
        output_path=str(MODEL_DIR / "shap_summary_global.png")
    )
    
    # 3. Stability Analysis
    logger.info("\n--- SHAP Stability Comparison ---")
    
    # Compute correlation between baseline and global SHAP values
    # Flatten the arrays to compute overall correlation
    flat_base = baseline_shap.flatten()
    flat_glob = global_shap.flatten()
    
    correlation = np.corrcoef(flat_base, flat_glob)[0, 1]
    logger.info(f"Pearson Correlation (Baseline vs Global SHAP): {correlation:.4f}")
    
    # Compare rankings
    base_rank = list(baseline_importance.keys())
    glob_rank = list(global_importance.keys())
    
    logger.info(f"Baseline Rank: {base_rank}")
    logger.info(f"Global Rank:   {glob_rank}")
    
    # Save results
    results = {
        "correlation": float(correlation),
        "baseline_importance": baseline_importance,
        "global_importance": global_importance,
        "rank_changed": base_rank != glob_rank
    }
    
    with open(MODEL_DIR / "shap_stability.json", "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"Saved stability results to {MODEL_DIR / 'shap_stability.json'}")


if __name__ == "__main__":
    run_analysis()
