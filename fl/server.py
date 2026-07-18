"""
SupplyShield — Standalone Flower Server

Runs the global Flower server for actual distributed deployments.
Uses the custom SaveModelStrategy.

Usage:
    python -m fl.server
"""

import sys
from pathlib import Path

import flwr as fl

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FL_SERVER_ADDRESS, FL_ROUNDS, FL_MIN_CLIENTS
from utils import logger, set_seed
from fl.strategy import SaveModelStrategy


def main():
    logger.info("=" * 60)
    logger.info(f"  Starting SupplyShield FL Server on {FL_SERVER_ADDRESS}")
    logger.info("=" * 60)
    
    set_seed()
    
    # Initialize the custom strategy
    strategy = SaveModelStrategy(
        fraction_fit=1.0,           
        fraction_evaluate=1.0,      
        min_fit_clients=FL_MIN_CLIENTS,
        min_evaluate_clients=FL_MIN_CLIENTS,
        min_available_clients=FL_MIN_CLIENTS,
    )
    
    # Start the server
    fl.server.start_server(
        server_address=FL_SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=FL_ROUNDS),
        strategy=strategy,
    )
    
    logger.info("Server stopped.")


if __name__ == "__main__":
    main()
