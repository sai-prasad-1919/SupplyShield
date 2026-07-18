"""
SupplyShield — Shared Utilities

Common helper functions used across the project:
- Reproducibility (seed setting)
- Logging setup
- Device detection
- Data loading helpers
"""

import os
import random
import logging
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Get the best available compute device (CPU for this project)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def setup_logging(
    name: str = "supplyshield",
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """
    Configure and return a logger.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional file path to also write logs to.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist. Returns the Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_csv_rows(filepath: str | Path) -> int:
    """Quick row count for a CSV file (excluding header)."""
    with open(filepath, "r") as f:
        return sum(1 for _ in f) - 1


# Pre-configured project logger
logger = setup_logging()
