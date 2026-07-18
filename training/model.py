"""
SupplyShield - DelayPredictor Model Architecture
Feedforward neural network for binary delay prediction.
"""
import torch
import torch.nn as nn


class DelayPredictor(nn.Module):
    """
    Feedforward network for binary delay prediction.
    Includes BatchNorm and Dropout to prevent overfitting.
    """
    def __init__(self, input_dim, hidden_dims=None, dropout_rate=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
