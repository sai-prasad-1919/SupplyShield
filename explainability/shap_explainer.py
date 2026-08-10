"""
SupplyShield — SHAP Explainability Integration

Provides functions to generate SHAP explanations for PyTorch model predictions.
Uses SHAP's DeepExplainer (or GradientExplainer) designed for deep learning models.
"""

import sys
from pathlib import Path

import torch
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import logger, ensure_dir
from config import PROJECT_ROOT


class SHAPModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        out = self.model(x)
        if len(out.shape) == 1:
            return out.unsqueeze(1)
        return out

class DelayExplainer:
    """Wrapper for SHAP explainability on the PyTorch DelayPredictor."""

    def __init__(self, model: torch.nn.Module, background_data: torch.Tensor, feature_names: list[str]):
        """
        Initialize the explainer.
        
        Args:
            model: PyTorch DelayPredictor model.
            background_data: A representative sample of the training data (e.g., 100-500 samples)
                             to serve as the background distribution for SHAP.
            feature_names: List of feature names in the same order as the input tensor.
        """
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.background_data = background_data.to(self.device)
        self.feature_names = feature_names
        self.wrapped_model = SHAPModelWrapper(self.model)
        
        # DeepExplainer works well for PyTorch feedforward networks
        # We need to wrap the model to return a single output for binary classification
        # since DeepExplainer expects a specific output format
        self.explainer = shap.DeepExplainer(self.wrapped_model, self.background_data)
        logger.info(f"Initialized SHAP DeepExplainer with {len(background_data)} background samples")

    def explain_instances(self, instances: torch.Tensor) -> np.ndarray:
        """
        Generate SHAP values for the given instances.
        
        Args:
            instances: Tensor of shape (batch_size, num_features)
            
        Returns:
            NumPy array of SHAP values of shape (batch_size, num_features)
        """
        # Calculate SHAP values
        # DeepExplainer returns a list of arrays (one per output class) or a single array
        # For a model returning [batch_size], it might return a tensor/array of the same shape
        shap_values = self.explainer.shap_values(instances)
        
        # Format the output to standard (batch_size, num_features)
        if isinstance(shap_values, list):
            # If it returns a list (e.g., multiclass), take the first one for binary
            shap_values = shap_values[0]
            
        if len(shap_values.shape) > 2:
            shap_values = np.squeeze(shap_values, axis=-1)
            
        return shap_values

    def plot_summary(self, instances: torch.Tensor, shap_values: np.ndarray, output_path: str = None) -> None:
        """Generate and save a SHAP summary plot."""
        plt.figure(figsize=(10, 6))
        
        # SHAP expects numpy arrays
        features_np = instances.cpu().numpy()
        
        # Create summary plot
        shap.summary_plot(
            shap_values, 
            features=features_np, 
            feature_names=self.feature_names,
            show=False
        )
        
        if output_path:
            path = Path(output_path)
            ensure_dir(path.parent)
            plt.savefig(path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved SHAP summary plot to {path}")
            plt.close()
        else:
            plt.show()
            
    def get_feature_importance(self, shap_values: np.ndarray) -> dict[str, float]:
        """
        Calculate global feature importance (mean absolute SHAP value).
        """
        # Average absolute SHAP values across all instances
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Create a dictionary mapping feature names to importance
        importance = {
            name: float(val) 
            for name, val in zip(self.feature_names, mean_abs_shap)
        }
        
        # Sort by importance (descending)
        return dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))


def generate_explanation_dict(
    feature_names: list[str], 
    feature_values: np.ndarray, 
    shap_values: np.ndarray, 
    base_value: float,
    prediction_prob: float
) -> dict:
    """
    Format SHAP output into a structured dictionary suitable for the LLM Guidance Engine.
    
    Args:
        feature_names: List of feature names.
        feature_values: The raw/scaled values for a single instance.
        shap_values: The SHAP values for a single instance.
        base_value: The expected model output over the background dataset.
        prediction_prob: The final predicted probability of delay.
    """
    contributions = []
    
    for name, val, shap_val in zip(feature_names, feature_values, shap_values):
        contributions.append({
            "feature": name,
            "value": float(val),
            "impact": float(shap_val),
            "direction": "increases_delay" if shap_val > 0 else "decreases_delay"
        })
        
    # Sort contributions by absolute impact
    contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
    
    return {
        "prediction": {
            "delay_probability": float(prediction_prob),
            "is_delayed": bool(prediction_prob >= 0.5)
        },
        "explainability": {
            "base_value": float(base_value),
            "top_drivers": contributions[:3],  # Top 3 most impactful features
            "all_contributions": contributions
        }
    }

class XGBoostExplainer:
    """Wrapper for SHAP explainability on XGBoost model."""
    def __init__(self, model, feature_names: list[str]):
        # model is the xgb.XGBClassifier
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names
        
    def explain_instances(self, instances: np.ndarray) -> np.ndarray:
        # returns log-odds contributions
        shap_values = self.explainer.shap_values(instances)
        return shap_values

