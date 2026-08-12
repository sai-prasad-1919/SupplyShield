"""
SupplyShield — XGBoost Training Script

Trains the XGBoost classifier on the combined training dataset.
Evaluates metrics on the test dataset and determines the optimal threshold.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_PROCESSED_DIR, CHECKPOINTS_DIR, PROJECT_ROOT
from utils import logger, set_seed
from models.xgboost_model import XGBoostPredictor


def load_combined_data():
    """Load and combine train and test data across all orgs."""
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    orgs = ["novamart", "titanelec", "swiftlog"]
    
    for org in orgs:
        train_df = pd.read_csv(PROJECT_ROOT / "data" / "partitions" / org / "train.csv")
        test_df = pd.read_csv(PROJECT_ROOT / "data" / "partitions" / org / "test.csv")
        
        target = "is_late" if "is_late" in train_df.columns else "delayed"
        
        X_train_list.append(train_df.drop(columns=[target]))
        y_train_list.append(train_df[target])
        
        X_test_list.append(test_df.drop(columns=[target]))
        y_test_list.append(test_df[target])
        
    X_train = pd.concat(X_train_list, axis=0)
    y_train = pd.concat(y_train_list, axis=0)
    
    X_test = pd.concat(X_test_list, axis=0)
    y_test = pd.concat(y_test_list, axis=0)
    
    return X_train, y_train, X_test, y_test


def main():
    set_seed()
    
    logger.info("Loading dataset for XGBoost training...")
    X_train, y_train, X_test, y_test = load_combined_data()
    
    # Scale data
    scaler = joblib.load(CHECKPOINTS_DIR / "scaler.pkl")
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    y_train_arr = y_train.values
    y_test_arr = y_test.values
    
    # Calculate scale_pos_weight
    n_positive = np.sum(y_train_arr == 1)
    n_negative = np.sum(y_train_arr == 0)
    scale_pos_weight = n_negative / n_positive
    logger.info(f"Class imbalance detected: Pos={n_positive}, Neg={n_negative}. scale_pos_weight={scale_pos_weight:.2f}")
    
    # Train
    predictor = XGBoostPredictor(scale_pos_weight=scale_pos_weight)
    predictor.train(X_train_s, y_train_arr)
    
    # Predict on test set
    y_probs = predictor.predict_proba(X_test_s)
    
    # Calculate optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(y_test_arr, y_probs)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    logger.info(f"Optimal decision threshold (Youden's J): {optimal_threshold:.4f}")
    
    # Evaluate with threshold
    y_pred = (y_probs >= optimal_threshold).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(y_test_arr, y_pred),
        "f1": f1_score(y_test_arr, y_pred),
        "auc": roc_auc_score(y_test_arr, y_probs),
        "precision": precision_score(y_test_arr, y_pred, zero_division=0),
        "recall": recall_score(y_test_arr, y_pred, zero_division=0)
    }
    
    logger.info("--- XGBoost Final Test Metrics ---")
    for k, v in metrics.items():
        logger.info(f"  {k.capitalize()}: {v:.4f}")
        
    # Save model
    predictor.save(CHECKPOINTS_DIR / "xgboost_model.json")
    
    # Save threshold
    joblib.dump(float(optimal_threshold), CHECKPOINTS_DIR / "xgboost_threshold.pkl")
    
    # Update metrics_summary.json if it exists
    metrics_path = PROJECT_ROOT / "reports" / "metrics_summary.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            all_metrics = json.load(f)
            
        all_metrics["xgboost"] = metrics
        
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=4)
        logger.info(f"Updated {metrics_path} with XGBoost metrics")

if __name__ == "__main__":
    main()
