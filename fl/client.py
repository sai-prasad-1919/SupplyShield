"""
SupplyShield — Flower Federated Learning Client

Wraps the PyTorch local model in a Flower NumPyClient.
"""

import sys
from pathlib import Path
from collections import OrderedDict
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from flwr.common import NDArrays, Scalar

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LEARNING_RATE, LOCAL_EPOCHS
from utils import logger, get_device
from models.delay_predictor import create_model
from training.dataset import get_dataloaders
from training.trainer import evaluate

warnings.filterwarnings("ignore", category=UserWarning)


class SupplyChainClient(fl.client.NumPyClient):
    """Flower client for supply chain delay prediction."""

    def __init__(self, org_id: str):
        self.org_id = org_id
        self.device = get_device()
        
        # Load dataloaders
        loaders = get_dataloaders(org_id)
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]
        self.test_loader = loaders["test"]
        
        # Initialize        
        # Infer input_dim from the first sample
        features, _ = self.train_loader.dataset[0]
        self.input_dim = len(features)
        
        self.model = create_model(input_dim=self.input_dim).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        
    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Extract model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Apply parameters from the server to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Train the local model using the provided parameters."""
        self.set_parameters(parameters)
        
        epochs = config.get("local_epochs", LOCAL_EPOCHS)
        lr = config.get("learning_rate", LEARNING_RATE)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            for features, targets in self.train_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                logits = self.model(features)
                loss = self.criterion(logits, targets)
                loss.backward()
                optimizer.step()
                
        # Calculate local metrics after training
        metrics = evaluate(self.model, self.val_loader, self.device)
        
        # Return updated parameters, dataset size, and metrics
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"f1": metrics["f1"]}

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluate the provided parameters on the local validation set."""
        self.set_parameters(parameters)
        metrics = evaluate(self.model, self.val_loader, self.device)
        
        # We also evaluate on test set here just to log its performance, though standard FL
        # evaluates on the validation set for aggregation weighting.
        test_metrics = evaluate(self.model, self.test_loader, self.device)
        
        return metrics["loss"], len(self.val_loader.dataset), {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
            "test_f1": test_metrics["f1"],
            "test_auc": test_metrics["auc"],
            "test_acc": test_metrics["accuracy"]
        }


def client_fn(cid: str) -> fl.client.Client:
    """Factory function to create a client for Flower simulation."""
    # In a simulation, cid is passed as a string index, e.g., "0", "1", "2"
    from config import ORG_IDS
    
    org_id = ORG_IDS[int(cid)]
    return SupplyChainClient(org_id).to_client()
