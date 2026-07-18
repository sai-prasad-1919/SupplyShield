"""
SupplyShield — Centralized Configuration

All paths, hyperparameters, and organization definitions live here.
Loads from .env when available, falls back to sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# ──────────────────────────────────────────────
# Paths (relative to project root)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW_DIR = PROJECT_ROOT / os.getenv("DATA_RAW_DIR", "data/raw")
DATA_PROCESSED_DIR = PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/processed")
DATA_PARTITIONS_DIR = PROJECT_ROOT / os.getenv("DATA_PARTITIONS_DIR", "data/partitions")
MODEL_DIR = PROJECT_ROOT / os.getenv("MODEL_DIR", "models/saved")

# Ensure directories exist
for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_PARTITIONS_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Organization Definitions
# ──────────────────────────────────────────────
ORGANIZATIONS = {
    "novamart": {
        "name": "NovaMart",
        "role": "Retailer",
        "delivery_partner": "Partner 1",  # Will be updated after data exploration
        "partition_dir": DATA_PARTITIONS_DIR / "novamart",
    },
    "titanelec": {
        "name": "TitanElec",
        "role": "Electronics Manufacturer",
        "delivery_partner": "Partner 2",
        "partition_dir": DATA_PARTITIONS_DIR / "titanelec",
    },
    "swiftlog": {
        "name": "SwiftLog",
        "role": "Logistics Distributor",
        "delivery_partner": "Partner 3",
        "partition_dir": DATA_PARTITIONS_DIR / "swiftlog",
    },
}

ORG_IDS = list(ORGANIZATIONS.keys())

# ──────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", "5"))
FL_ROUNDS = int(os.getenv("FL_ROUNDS", "10"))
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "64"))
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────
# Dynamically determined in preprocess.py

# ──────────────────────────────────────────────
# Federated Learning
# ──────────────────────────────────────────────
# ==========================================
# FLOWER SERVER & FEDERATED LEARNING
# ==========================================
FL_SERVER_ADDRESS = "127.0.0.1:8080"
FL_MIN_CLIENTS = 3

# ==========================================
# LLM GUIDANCE
# ==========================================
USE_LLM = True
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
FL_MIN_AVAILABLE = 3

# ──────────────────────────────────────────────
# API Configuration
# ──────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'supplyshield.db'}")

# ──────────────────────────────────────────────
# LLM Configuration (Phase 6)
# ──────────────────────────────────────────────
# Using GEMINI_API_KEY above

# ──────────────────────────────────────────────
# Feature Configuration (set in Phase 2)
# ──────────────────────────────────────────────
# These will be populated after data exploration
CATEGORICAL_FEATURES = ["weather", "traffic", "vehicle_type"]
NUMERICAL_FEATURES = ["distance_km"]
TARGET_COLUMN = "is_late"  # Assumed; will verify in Phase 2

# ──────────────────────────────────────────────
# Dataset file names
# ──────────────────────────────────────────────
INDIA_DATASET_FILENAME = "delivery_logistics.csv"   # Will rename after download
DATACO_DATASET_FILENAME = "dataco_supply_chain.csv"
WALMART_DATASET_FILENAME = "walmart_sales.csv"
