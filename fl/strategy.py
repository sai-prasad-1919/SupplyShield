"""
SupplyShield — Custom Federated Learning Strategy

Extends standard FedAvg to:
1. Aggregate custom metrics (F1, AUC, Accuracy)
2. Save the global model checkpoint after each round
"""

import sys
from pathlib import Path
from collections import OrderedDict
import json

import torch
import flwr as fl
from flwr.common import NDArrays, Scalar, Parameters, FitRes, EvaluateRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODEL_DIR
from utils import logger
from models.delay_predictor import create_model


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy that aggregates metrics and saves the global model."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = {"rounds": []}
        
        import joblib
        from config import DATA_PROCESSED_DIR
        from utils import get_device
        try:
            metadata = joblib.load(DATA_PROCESSED_DIR / "metadata.pkl")
            self.input_dim = len(metadata["feature_names"])
        except FileNotFoundError:
            self.input_dim = 4 # fallback

        self.global_model = create_model(input_dim=self.input_dim).to(get_device())
        
    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """Aggregate evaluation metrics from clients."""
        if not results:
            return None, {}
            
        # Call standard FedAvg aggregate_evaluate to get aggregated loss
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        # Calculate custom weighted metrics (weighted by number of validation examples)
        total_examples = sum([res.num_examples for _, res in results])
        
        agg_metrics = {}
        # Keys to aggregate
        keys = ["accuracy", "precision", "recall", "f1", "auc", "test_acc", "test_precision", "test_recall", "test_f1", "test_auc"]
        
        for key in keys:
            weighted_sum = sum([res.metrics.get(key, 0.0) * res.num_examples for _, res in results])
            agg_metrics[key] = weighted_sum / total_examples
            
        logger.info(f"--- Round {server_round} Evaluation ---")
        logger.info(f"Aggregated Loss: {loss_aggregated:.4f}")
        logger.info(f"Aggregated F1:   {agg_metrics['f1']:.4f}")
        logger.info(f"Aggregated AUC:  {agg_metrics['auc']:.4f}")
        
        # Save to history
        round_data = {
            "round": server_round,
            "loss": loss_aggregated,
            **agg_metrics
        }
        self.history["rounds"].append(round_data)
        
        # Save history to disk (models/saved/ - legacy format)
        with open(MODEL_DIR / "fl_history.json", "w") as f:
            json.dump(self.history, f, indent=4)
            
        # Transform history into the format required by dashboard and save in checkpoints/
        dashboard_history = {
            "loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "auc": []
        }
        for rd in self.history["rounds"]:
            rnd = rd["round"]
            dashboard_history["loss"].append({"round": rnd, "value": rd["loss"]})
            for k in ["accuracy", "precision", "recall", "f1", "auc"]:
                dashboard_history[k].append({"round": rnd, "value": rd.get(k, 0.0)})
                
        from config import CHECKPOINTS_DIR
        with open(CHECKPOINTS_DIR / "fl_history.json", "w") as f:
            json.dump(dashboard_history, f, indent=4)
            
        return loss_aggregated, agg_metrics

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results and save global model."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Convert parameters back to NDArrays
            ndarrays: NDArrays = parameters_to_ndarrays(aggregated_parameters)
            
            # Load into model to save PyTorch checkpoint
            params_dict = zip(self.global_model.state_dict().keys(), ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.global_model.load_state_dict(state_dict, strict=True)
            
            # Save global model
            torch.save(self.global_model.state_dict(), MODEL_DIR / f"global_model_r{server_round}.pt")
            
            # Keep a 'latest' symlink/copy
            torch.save(self.global_model.state_dict(), MODEL_DIR / "global_model_latest.pt")
            logger.info(f"Saved global model for round {server_round}")
            
        return aggregated_parameters, aggregated_metrics
