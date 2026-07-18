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
    
    # Compare with baselines
    logger.info("\n--- Final Global Model Metrics (Test Set) ---")
    if "test_f1" in history.metrics_distributed:
        final_round, final_f1 = history.metrics_distributed["test_f1"][-1]
        logger.info(f"Global F1 Score: {final_f1:.4f}")
    if "test_auc" in history.metrics_distributed:
        final_round, final_auc = history.metrics_distributed["test_auc"][-1]
        logger.info(f"Global AUC:      {final_auc:.4f}")
        

if __name__ == "__main__":
    main()
