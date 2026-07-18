"""
SupplyShield - Step 2: Data Preprocessing & Feature Engineering
Cleans data, engineers features, partitions by delivery partner, saves processed data.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA = os.path.join(ROOT, 'data', 'raw')

print("=" * 60)
print("  STEP 2: Data Preprocessing & Feature Engineering")
print("=" * 60)

# ── Load ───────────────────────────────────────────────────────
csv_path = os.path.join(RAW_DATA, 'Delivery_Logistics.csv')
df = pd.read_csv(csv_path)
print(f"Loaded: {df.shape}")

# ── Column inventory ──────────────────────────────────────────
# Columns: delivery_id, delivery_partner, package_type, vehicle_type,
#           delivery_mode, region, weather_condition, distance_km,
#           package_weight_kg, delivery_time_hours, expected_time_hours,
#           delayed, delivery_status, delivery_rating, delivery_cost

TARGET_COL = 'delayed'
PARTITION_COL = 'delivery_partner'

# ── Step 1: Clean ──────────────────────────────────────────────
print("\n--- Cleaning ---")

# Drop delivery_id (not informative — all identical 250.99)
if 'delivery_id' in df.columns:
    df = df.drop(columns=['delivery_id'])
    print("Dropped: delivery_id (constant value)")

# Drop delivery_status — it leaks the target (delivered/delayed/failed)
if 'delivery_status' in df.columns:
    df = df.drop(columns=['delivery_status'])
    print("Dropped: delivery_status (target leakage)")

# Handle the time columns — they are in epoch format
# delivery_time_hours and expected_time_hours appear to be encoded oddly
# They look like 1970-01-01 00:00:00.000000008 which is nanoseconds from epoch
# The actual numeric value (8, 3, 16, etc.) is the number of hours
for time_col in ['delivery_time_hours', 'expected_time_hours']:
    if time_col in df.columns:
        try:
            # Try parsing as datetime and extracting nanosecond component as hours
            ts = pd.to_datetime(df[time_col], errors='coerce')
            # The nanoseconds represent the actual hour value
            hour_vals = ts.dt.nanosecond
            # Validate: should be small positive integers (1-48 range for hours)
            if hour_vals.max() <= 100 and hour_vals.min() >= 0:
                df[time_col] = hour_vals.astype(float)
                print(f"Converted {time_col} from timestamp to numeric hours (range: {hour_vals.min()}-{hour_vals.max()})")
            else:
                # Fallback: try numeric conversion
                df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
        except Exception:
            df[time_col] = pd.to_numeric(df[time_col], errors='coerce')

# Create time difference feature
if 'delivery_time_hours' in df.columns and 'expected_time_hours' in df.columns:
    df['time_diff_hours'] = df['delivery_time_hours'] - df['expected_time_hours']
    print(f"Created: time_diff_hours (delivery - expected)")

# Missing values
print(f"\nMissing values before cleaning:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

cat_cols = df.select_dtypes(include=['object', 'string']).columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Remove duplicates
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate rows")

# ── Step 2: Encode target ─────────────────────────────────────
print("\n--- Encoding target ---")
# delayed: 'yes' -> 1, 'no' -> 0
df[TARGET_COL] = (df[TARGET_COL].str.lower().str.strip() == 'yes').astype(int)
print(f"Target encoded: {df[TARGET_COL].value_counts().to_dict()}")

# ── Step 3: Encode categoricals ───────────────────────────────
print("\n--- Encoding categoricals ---")
label_encoders = {}
cat_features = df.select_dtypes(include=['object', 'string']).columns.tolist()
# Remove partition column from encoding
cat_features_to_encode = [c for c in cat_features if c != PARTITION_COL]

for col in cat_features_to_encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded '{col}': {len(le.classes_)} classes")

# Save encoders
os.makedirs(os.path.join(ROOT, 'checkpoints'), exist_ok=True)
joblib.dump(label_encoders, os.path.join(ROOT, 'checkpoints', 'label_encoders.pkl'))
print(f"Saved encoders to checkpoints/label_encoders.pkl")

# ── Step 4: Partition by delivery partner ─────────────────────
print("\n--- Partitioning by delivery partner ---")

partner_counts = df[PARTITION_COL].value_counts()
print(f"Partner distribution:\n{partner_counts}\n")

top_3 = partner_counts.head(3).index.tolist()
org_mapping = {
    top_3[0]: 'novamart',
    top_3[1]: 'titanelec',
    top_3[2]: 'swiftlog',
}

print(f"Partner -> Org mapping:")
for partner, org in org_mapping.items():
    print(f"  {partner} -> {org}")

for partner, org_name in org_mapping.items():
    partition = df[df[PARTITION_COL] == partner].copy()
    partition = partition.drop(columns=[PARTITION_COL])

    out_dir = os.path.join(ROOT, 'data', 'partitions', org_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'data.csv')
    partition.to_csv(out_path, index=False)

    delay_rate = partition[TARGET_COL].mean()
    print(f"  {org_name} ({partner}): {len(partition)} records, {delay_rate:.2%} delay rate -> {out_path}")

# Save combined dataset (for centralized training)
full = df[df[PARTITION_COL].isin(top_3)].drop(columns=[PARTITION_COL])
combined_dir = os.path.join(ROOT, 'data', 'processed')
os.makedirs(combined_dir, exist_ok=True)
combined_path = os.path.join(combined_dir, 'combined.csv')
full.to_csv(combined_path, index=False)
print(f"\nCombined dataset: {len(full)} records -> {combined_path}")

# Save org mapping for later use
joblib.dump(org_mapping, os.path.join(ROOT, 'checkpoints', 'org_mapping.pkl'))

# Print final feature set
feature_cols = [c for c in full.columns if c != TARGET_COL]
print(f"\nFinal features ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i}. {col}")

print(f"\nFinal shape: {full.shape}")
print("\n[OK] Preprocessing complete.")
