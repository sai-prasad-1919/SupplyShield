"""
SupplyShield — XGBoost Delay Predictor

Second-opinion classification model. Independent from the FL Neural Network.
"""

import sys
from pathlib import Path
import json

import xgboost as xgb
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import logger

class XGBoostPredictor:
    def __init__(self, scale_pos_weight: float = 1.0, random_state: int = 42):
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            random_state=random_state,
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the XGBoost model."""
        logger.info(f"Training XGBoost on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the probability of the positive class (delay)."""
        # predict_proba returns [prob_0, prob_1], we want prob_1
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path | str):
        """Save the XGBoost model as JSON."""
        # Ensure path is a string for save_model
        str_path = str(path)
        self.model.save_model(str_path)
        logger.info(f"Saved XGBoost model to {str_path}")

    def load(self, path: Path | str):
        """Load the XGBoost model from JSON."""
        str_path = str(path)
        self.model.load_model(str_path)
        logger.info(f"Loaded XGBoost model from {str_path}")
