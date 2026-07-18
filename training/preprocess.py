"""
SupplyShield — Data Preprocessing Pipeline

Handles the India Multi-Partner Delivery Logistics dataset:
1. Load raw CSV
2. Explore and understand features
3. Clean (handle missing values, outliers)
4. Feature engineering (encode categoricals, scale numericals)
5. Partition by delivery_partner into org-specific folders
6. Save train/val/test splits per org

Usage:
    python -m training.preprocess
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_PARTITIONS_DIR,
    ORGANIZATIONS,
    INDIA_DATASET_FILENAME,
    RANDOM_SEED,
)
from utils import logger, set_seed


def load_raw_data() -> pd.DataFrame:
    """Load the India Multi-Partner delivery dataset."""
    filepath = DATA_RAW_DIR / INDIA_DATASET_FILENAME
    if not filepath.exists():
        # Try to find any CSV in raw directory
        csvs = list(DATA_RAW_DIR.glob("*.csv"))
        if csvs:
            filepath = csvs[0]
            logger.info(f"Using found CSV: {filepath.name}")
        else:
            raise FileNotFoundError(
                f"No dataset found in {DATA_RAW_DIR}. "
                f"Please download the India Multi-Partner dataset from Kaggle "
                f"and place the CSV in {DATA_RAW_DIR}/"
            )

    logger.info(f"Loading dataset from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def explore_data(df: pd.DataFrame) -> dict:
    """
    Print dataset summary and return key stats.
    Call this first to understand the data before cleaning.
    """
    logger.info("=" * 60)
    logger.info("DATASET EXPLORATION")
    logger.info("=" * 60)

    stats = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
    }

    print(f"\nShape: {df.shape}")
    print(f"\nColumns:\n{df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nBasic statistics:\n{df.describe()}")

    # Identify the delivery partner column for FL partitioning
    partner_candidates = [
        c for c in df.columns if "partner" in c.lower() or "delivery" in c.lower()
    ]
    print(f"\nPotential partition columns: {partner_candidates}")

    for col in partner_candidates:
        if df[col].dtype == "object" or df[col].nunique() < 20:
            print(f"\n{col} value counts:")
            print(df[col].value_counts())

    # Identify potential target column
    target_candidates = [
        c for c in df.columns if "late" in c.lower() or "delay" in c.lower() or "on_time" in c.lower()
    ]
    print(f"\nPotential target columns: {target_candidates}")
    for col in target_candidates:
        print(f"\n{col} value counts:")
        print(df[col].value_counts())

    return stats


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Drop duplicates
    - Handle missing values
    - Remove obvious outliers
    - Standardize column names
    """
    logger.info("Cleaning data...")
    initial_rows = len(df)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Drop duplicates
    df = df.drop_duplicates()
    logger.info(f"Dropped {initial_rows - len(df)} duplicate rows")

    # Handle missing values
    missing = df.isnull().sum()
    if missing.any():
        logger.info(f"Missing values found:\n{missing[missing > 0]}")
        # For numerical columns: fill with median
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # For categorical columns: fill with mode
        cat_cols = df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])

    logger.info(f"Cleaned data: {len(df):,} rows remaining")
    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Feature engineering:
    - Encode categorical variables (label encoding for tree-based, one-hot for NN)
    - Scale numerical features
    - Create derived features if applicable

    Returns:
        (processed_df, metadata_dict) where metadata contains encoders and scalers
    """
    logger.info("Engineering features...")
    metadata = {"encoders": {}, "scaler": None, "feature_names": []}

    # Identify feature types
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target and partition columns from features
    exclude_cols = ["is_late", "delivery_id", "delivery_partner"]

    # Label encode categorical features
    for col in cat_cols:
        if col not in exclude_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            metadata["encoders"][col] = le
            logger.info(f"  Encoded '{col}': {len(le.classes_)} classes")

    # Scale numerical features
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    num_feature_cols = [c for c in feature_cols if c in num_cols]

    # 3. Numerical Scaling
    # [!] KNOWN SIMPLIFICATION (H1): 
    # Scaling the entire dataset before partitioning leaks global distribution 
    # statistics across organizations. In a strict FL production environment, 
    # each client must fit its own scaler locally.
    if num_feature_cols:
        scaler = StandardScaler()
        df[num_feature_cols] = scaler.fit_transform(df[num_feature_cols])
        metadata["scaler"] = scaler
        logger.info(f"  Scaled {len(num_feature_cols)} numerical features")

    metadata["feature_names"] = feature_cols
    return df, metadata


def partition_by_org(
    df: pd.DataFrame,
    partner_column: str,
    target_column: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> dict:
    """
    Partition dataset by delivery partner and create train/val/test splits per org.

    Args:
        df: Preprocessed DataFrame
        partner_column: Column name to partition by
        target_column: Column name for the prediction target
        test_size: Fraction for test set
        val_size: Fraction for validation set (from remaining after test)

    Returns:
        Dict mapping org_id -> {train, val, test} DataFrames
    """
    logger.info(f"Partitioning by '{partner_column}'...")
    partners = sorted(df[partner_column].unique())
    logger.info(f"Found {len(partners)} unique partners: {partners}")

    org_data = {}
    org_mapping = list(zip(partners, ORGANIZATIONS.keys()))

    for partner_val, org_id in org_mapping:
        org_df = df[df[partner_column] == partner_val].copy()
        org_df = org_df.drop(columns=[partner_column, "delivery_id"], errors="ignore")

        logger.info(f"  {org_id} ({ORGANIZATIONS[org_id]['name']}): {len(org_df):,} rows")

        # Split: test first, then val from remaining
        X = org_df.drop(columns=[target_column])
        y = org_df[target_column]

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
        )
        relative_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=relative_val_size, random_state=RANDOM_SEED, stratify=y_temp
        )

        # Save to partition directory
        partition_dir = DATA_PARTITIONS_DIR / org_id
        partition_dir.mkdir(parents=True, exist_ok=True)

        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        train_df.to_csv(partition_dir / "train.csv", index=False)
        val_df.to_csv(partition_dir / "val.csv", index=False)
        test_df.to_csv(partition_dir / "test.csv", index=False)

        logger.info(f"    Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

        org_data[org_id] = {"train": train_df, "val": val_df, "test": test_df}

    return org_data


def run_pipeline():
    """Run the full preprocessing pipeline."""
    set_seed(RANDOM_SEED)

    # Step 1: Load
    df = load_raw_data()

    # Step 2: Explore
    stats = explore_data(df)

    # Step 3: Clean
    df = clean_data(df)

    # Save cleaned data
    df.to_csv(DATA_PROCESSED_DIR / "cleaned.csv", index=False)
    logger.info(f"Saved cleaned data to {DATA_PROCESSED_DIR / 'cleaned.csv'}")

    # Step 4: Feature Engineering
    df, metadata = engineer_features(df)
    
    import joblib
    joblib.dump(metadata, DATA_PROCESSED_DIR / "metadata.pkl")

    # Step 5: Partition by org
    org_data = partition_by_org(
        df,
        partner_column="delivery_partner",
        target_column="is_late"
    )
    logger.info("Pipeline complete. Partitions created successfully.")


if __name__ == "__main__":
    run_pipeline()
