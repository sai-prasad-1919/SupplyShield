"""
SupplyShield - Step 2: Data Preprocessing & Feature Engineering
Cleans data, engineers features, partitions by delivery partner, saves processed data.

Leakage fix (2026-08-10):
  DROPPED post-delivery columns that directly encode the target:
    - delivery_time_hours  (actual time -- only known AFTER delivery)
    - time_diff_hours      (= delivery_time - expected_time = target definition)
    - delivery_rating      (customer rating given AFTER delivery)
  Scaler is now fit on training data ONLY (no test/val contamination).
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA = os.path.join(ROOT, 'data', 'raw')

print("=" * 60)
print("  STEP 2: Data Preprocessing & Feature Engineering")
print("=" * 60)

csv_path = os.path.join(RAW_DATA, 'Delivery_Logistics.csv')
df = pd.read_csv(csv_path)
print(f"Loaded: {df.shape}")

TARGET_COL    = 'delayed'
PARTITION_COL = 'delivery_partner'

# ── Step 1: Drop leaking & irrelevant columns ─────────────────
print("\n--- Dropping leaking / irrelevant columns ---")
ALWAYS_DROP = [
    'delivery_id',
    'delivery_status',
    'delivery_time_hours',
    'time_diff_hours',
    'delivery_rating',
]
for col in ALWAYS_DROP:
    if col in df.columns:
        df = df.drop(columns=[col])
        print(f"  Dropped: {col}")

# ── Step 2: Clean ─────────────────────────────────────────────
print("\n--- Cleaning ---")
for time_col in ['expected_time_hours']:
    if time_col in df.columns:
        try:
            ts = pd.to_datetime(df[time_col], errors='coerce')
            hour_vals = ts.dt.nanosecond
            if hour_vals.max() <= 100 and hour_vals.min() >= 0:
                df[time_col] = hour_vals.astype(float)
                print(f"Converted {time_col} from timestamp to numeric hours")
            else:
                df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
        except Exception:
            df[time_col] = pd.to_numeric(df[time_col], errors='coerce')

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

cat_cols_all = df.select_dtypes(include=['object', 'string']).columns
for col in cat_cols_all:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate rows. Remaining: {len(df)}")

# ── Step 3: Encode target ─────────────────────────────────────
print("\n--- Encoding target ---")
# Handle 'yes'/'no' strings (works for both object and pandas StringDtype)
# and plain numeric 0/1 columns
try:
    unique_vals = df[TARGET_COL].dropna().unique()
    # If any value looks like a string label, do string mapping
    if any(str(v).lower() in ('yes', 'no', 'true', 'false') for v in unique_vals):
        df[TARGET_COL] = (df[TARGET_COL].astype(str).str.lower().str.strip() == 'yes').astype(int)
    else:
        df[TARGET_COL] = df[TARGET_COL].astype(int)
except Exception as e:
    # Last resort: coerce
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce').fillna(0).astype(int)
    print(f"  Warning: target coerced ({e})")
print(f"Target encoded: {df[TARGET_COL].value_counts().to_dict()}")
print(f"Delay rate: {df[TARGET_COL].mean():.2%}")


# ── Step 4: Encode categoricals ───────────────────────────────
print("\n--- Encoding categoricals ---")
label_encoders = {}
cat_features = df.select_dtypes(include=['object', 'string']).columns.tolist()
cat_features_to_encode = [c for c in cat_features if c != PARTITION_COL]

for col in cat_features_to_encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded '{col}': {len(le.classes_)} classes")

os.makedirs(os.path.join(ROOT, 'checkpoints'), exist_ok=True)
joblib.dump(label_encoders, os.path.join(ROOT, 'checkpoints', 'label_encoders.pkl'))
print(f"Saved encoders -> checkpoints/label_encoders.pkl")

# ── Step 5: Partition by delivery partner ─────────────────────
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

# ── Step 6: Per-org train/val/test splits ─────────────────────
FL_TARGET_COL = 'is_late'
print("\n--- Creating per-org train/val/test splits ---")
print("  Scaler fitted on TRAIN ONLY per org -- no leakage\n")

for partner, org_name in org_mapping.items():
    partition = df[df[PARTITION_COL] == partner].copy()
    partition = partition.drop(columns=[PARTITION_COL])

    feature_cols = [c for c in partition.columns if c != TARGET_COL]
    X = partition[feature_cols].values.astype(np.float32)
    y = partition[TARGET_COL].values.astype(np.float32)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    val_ratio = 0.10 / 0.80
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s   = scaler.transform(X_val).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)

    out_dir = os.path.join(ROOT, 'data', 'partitions', org_name)
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.DataFrame(X_train_s, columns=feature_cols)
    train_df[FL_TARGET_COL] = y_train.astype(int)

    val_df = pd.DataFrame(X_val_s, columns=feature_cols)
    val_df[FL_TARGET_COL] = y_val.astype(int)

    test_df = pd.DataFrame(X_test_s, columns=feature_cols)
    test_df[FL_TARGET_COL] = y_test.astype(int)

    train_df.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(out_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(out_dir, 'test.csv'), index=False)

    full_df_org = pd.DataFrame(X, columns=feature_cols)
    full_df_org[TARGET_COL] = y.astype(int)
    full_df_org.to_csv(os.path.join(out_dir, 'data.csv'), index=False)

    delay_rate = y.mean()
    print(f"  {org_name}: {len(partition)} records, {delay_rate:.2%} delay rate")
    print(f"    Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── Step 7: Combined dataset ─────────────────────────────────
print("\n--- Saving combined dataset ---")
full = df[df[PARTITION_COL].isin(top_3)].drop(columns=[PARTITION_COL])
combined_dir = os.path.join(ROOT, 'data', 'processed')
os.makedirs(combined_dir, exist_ok=True)
combined_path = os.path.join(combined_dir, 'combined.csv')
full.to_csv(combined_path, index=False)
print(f"Combined dataset: {len(full)} records -> {combined_path}")

# ── Step 8: Global scaler (train-only) ────────────────────────
print("\n--- Saving global scaler ---")
feature_cols_global = [c for c in full.columns if c != TARGET_COL]
X_all = full[feature_cols_global].values.astype(np.float32)
y_all = full[TARGET_COL].values.astype(np.float32)

X_temp_g, _, y_temp_g, _ = train_test_split(
    X_all, y_all, test_size=0.15, random_state=42, stratify=y_all
)
val_ratio_g = 0.15 / 0.85
X_train_g, _, _, _ = train_test_split(
    X_temp_g, y_temp_g, test_size=val_ratio_g, random_state=42, stratify=y_temp_g
)

global_scaler = StandardScaler()
global_scaler.fit(X_train_g)
joblib.dump(global_scaler, os.path.join(ROOT, 'checkpoints', 'scaler.pkl'))
print("Saved global scaler (train-only fit) -> checkpoints/scaler.pkl")

# ── Step 9: Metadata ──────────────────────────────────────────
metadata = {
    'feature_names': feature_cols_global,
    'target_col': TARGET_COL,
    'n_features': len(feature_cols_global),
    'encoders': label_encoders,
    'scaler': global_scaler,
}
joblib.dump(metadata, os.path.join(combined_dir, 'metadata.pkl'))
print(f"Saved metadata -> data/processed/metadata.pkl")

joblib.dump(org_mapping, os.path.join(ROOT, 'checkpoints', 'org_mapping.pkl'))

# ── Summary ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  PREPROCESSING COMPLETE")
print(f"{'='*60}")
print(f"\nFinal clean features ({len(feature_cols_global)}):")
for i, col in enumerate(feature_cols_global, 1):
    print(f"  {i}. {col}")
print(f"\nFinal combined shape: {full.shape}")
print(f"Delay rate: {full[TARGET_COL].mean():.2%}")
print("\n[OK] No leaking columns. Scaler fitted on train only.")