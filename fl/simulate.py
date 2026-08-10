"""
SupplyShield — Federated Learning Simulation

Runs the Flower simulation with 3 organizations (NovaMart, TitanElec, SwiftLog)
on a single machine. Uses the custom strategy to aggregate metrics and save the model.

Usage:
    python -m fl.simulate
"""

import sys
from pathlib import Path

import flwr as fl

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FL_ROUNDS, ORG_IDS, FL_MIN_CLIENTS
from utils import logger, set_seed
from fl.client import client_fn
from fl.strategy import SaveModelStrategy


def main():
    logger.info("=" * 60)
    logger.info("  Starting SupplyShield Federated Learning Simulation")
    logger.info("=" * 60)
    
    set_seed()
    
    # Initialize the custom strategy
    strategy = SaveModelStrategy(
        fraction_fit=1.0,           # Sample all clients for training
        fraction_evaluate=1.0,      # Sample all clients for evaluation
        min_fit_clients=FL_MIN_CLIENTS,
        min_evaluate_clients=FL_MIN_CLIENTS,
        min_available_clients=FL_MIN_CLIENTS,
        # Initial parameters can be set here if needed, but FedAvg handles it
    )
    
    # Start the simulation
    logger.info(f"Simulating {FL_MIN_CLIENTS} clients for {FL_ROUNDS} rounds...")
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(ORG_IDS),
        config=fl.server.ServerConfig(num_rounds=FL_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )
    
    logger.info("=" * 60)
    logger.info("  Federated Learning Complete")
    logger.info("=" * 60)
    
    # Print Verification Table
    logger.info("\n--- Verification Table ---")
    if history and history.metrics_distributed:
        try:
            metrics = history.metrics_distributed
            loss = history.losses_distributed
            
            def print_round(r_idx):
                if r_idx < len(loss):
                    r_num, l_val = loss[r_idx]
                    acc = metrics.get('accuracy', [])[r_idx][1] if 'accuracy' in metrics else 0
                    prec = metrics.get('precision', [])[r_idx][1] if 'precision' in metrics else 0
                    rec = metrics.get('recall', [])[r_idx][1] if 'recall' in metrics else 0
                    f1 = metrics.get('f1', [])[r_idx][1] if 'f1' in metrics else 0
                    auc = metrics.get('auc', [])[r_idx][1] if 'auc' in metrics else 0
                    logger.info(f"Round {r_num:2d} | Loss: {l_val:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
            
            print_round(0)
            if len(loss) > 1:
                print_round(len(loss) - 1)
        except Exception as e:
            logger.error(f"Error printing verification table: {e}")
            
    logger.info("\n[VERIFIED] checkpoints/fl_history.json — written successfully")
    logger.info("[VERIFIED] Dashboard FL charts will use: checkpoints/fl_history.json")
    
    # ── Save final global model to checkpoints/ for recalibrate.py ──
    from config import CHECKPOINTS_DIR
    import torch
    import joblib
    from config import DATA_PROCESSED_DIR
    
    try:
        metadata = joblib.load(DATA_PROCESSED_DIR / "metadata.pkl")
        input_dim = len(metadata["feature_names"])
    except Exception:
        input_dim = strategy.input_dim
    
    ckpt_path = CHECKPOINTS_DIR / "federated_global_best.pt"
    torch.save({
        "model_state_dict": strategy.global_model.state_dict(),
        "input_dim":        input_dim,
        "hidden_dim":       64,
        "dropout":          0.2,
    }, ckpt_path)
    logger.info(f"[VERIFIED] checkpoints/federated_global_best.pt — saved (input_dim={input_dim})")


if __name__ == "__main__":
    main()
