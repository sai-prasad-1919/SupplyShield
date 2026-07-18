"""
SupplyShield — Delay Prediction Model

A lightweight feedforward neural network for binary classification
(on-time vs. delayed delivery). Designed to be small enough (~5K params)
for efficient federated learning on CPU.

Architecture:
    Input → Linear(H) → ReLU → Dropout → Linear(H) → ReLU → Dropout → Linear(1) → Sigmoid

Where H = hidden_dim (default 64).
"""

import torch
import torch.nn as nn


class DelayPredictor(nn.Module):
    """
    Feedforward network for delivery delay prediction.

    Args:
        input_dim: Number of input features (set after preprocessing).
        hidden_dim: Width of hidden layers (default: 64).
        dropout: Dropout probability (default: 0.2).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, input_dim]
        # output shape: [batch_size, 1] -> [batch_size]
        return self.network(x).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return delay probability (0-1)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary predictions (0 = on-time, 1 = delayed)."""
        proba = self.predict_proba(x)
        return (proba >= threshold).long()

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(input_dim: int, hidden_dim: int = 64, dropout: float = 0.2) -> DelayPredictor:
    """Factory function to create a DelayPredictor model."""
    model = DelayPredictor(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    print(f"[Model] Created DelayPredictor: {model.count_parameters():,} parameters")
    print(f"        input_dim={input_dim}, hidden_dim={hidden_dim}, dropout={dropout}")
    return model


if __name__ == "__main__":
    # Quick sanity check
    demo_input_dim = 15  # placeholder
    model = create_model(demo_input_dim)
    x = torch.randn(4, demo_input_dim)
    logits = model(x)
    proba = model.predict_proba(x)
    preds = model.predict(x)
    print(f"\n[Demo] Input shape: {x.shape}")
    print(f"[Demo] Logits: {logits}")
    print(f"[Demo] Probabilities: {proba}")
    print(f"[Demo] Predictions: {preds}")
