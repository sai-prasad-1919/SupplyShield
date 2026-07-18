"""
SupplyShield — PyTorch Dataset and DataLoader Utilities

Provides:
- SupplyChainDataset: PyTorch Dataset for org-partitioned CSV data
- get_dataloaders: Factory function to create train/val/test DataLoaders per org
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_PARTITIONS_DIR, BATCH_SIZE, TARGET_COLUMN, RANDOM_SEED


class SupplyChainDataset(Dataset):
    """
    PyTorch Dataset for supply chain delivery data.

    Loads a CSV partition and provides (features, target) tensors.
    """

    def __init__(self, csv_path: str | Path, target_column: str = None):
        """
        Args:
            csv_path: Path to the CSV file (train.csv, val.csv, or test.csv).
            target_column: Name of the target column. Uses config default if None.
        """
        self.csv_path = Path(csv_path)
        self.target_column = target_column or TARGET_COLUMN

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Separate features and target
        self.feature_names = [c for c in df.columns if c != self.target_column]
        self.features = torch.tensor(
            df[self.feature_names].values, dtype=torch.float32
        )
        self.targets = torch.tensor(
            df[self.target_column].values, dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

    @property
    def input_dim(self) -> int:
        """Number of input features."""
        return self.features.shape[1]

    @property
    def num_positive(self) -> int:
        """Count of positive (delayed) samples."""
        return int(self.targets.sum().item())

    @property
    def class_balance(self) -> dict:
        """Return class distribution."""
        pos = self.num_positive
        neg = len(self) - pos
        return {
            "total": len(self),
            "delayed": pos,
            "on_time": neg,
            "delay_rate": pos / len(self) if len(self) > 0 else 0,
        }


def get_dataloaders(
    org_id: str,
    batch_size: int = None,
    target_column: str = None,
) -> dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders for a specific organization.

    Args:
        org_id: Organization identifier (novamart, titanelec, swiftlog).
        batch_size: Batch size (defaults to config value).
        target_column: Target column name (defaults to config value).

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders.
    """
    bs = batch_size or BATCH_SIZE
    partition_dir = DATA_PARTITIONS_DIR / org_id

    loaders = {}
    for split in ["train", "val", "test"]:
        csv_path = partition_dir / f"{split}.csv"
        dataset = SupplyChainDataset(csv_path, target_column=target_column)
        loaders[split] = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=(split == "train"),
            drop_last=False,
            generator=torch.Generator().manual_seed(RANDOM_SEED),
        )

    return loaders


def get_all_org_dataloaders(
    batch_size: int = None,
    target_column: str = None,
) -> dict[str, dict[str, DataLoader]]:
    """
    Create DataLoaders for all organizations.

    Returns:
        Dict mapping org_id -> {train, val, test} DataLoaders.
    """
    from config import ORG_IDS

    return {
        org_id: get_dataloaders(org_id, batch_size, target_column)
        for org_id in ORG_IDS
    }
