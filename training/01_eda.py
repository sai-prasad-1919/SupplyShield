"""
SupplyShield - Step 1: Exploratory Data Analysis
Loads raw data, explores distributions, saves EDA plots.
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA = os.path.join(ROOT, 'data', 'raw')
NOTEBOOKS = os.path.join(ROOT, 'notebooks')
os.makedirs(NOTEBOOKS, exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────
print("=" * 60)
print("  STEP 1: Exploratory Data Analysis")
print("=" * 60)

csv_path = os.path.join(RAW_DATA, 'Delivery_Logistics.csv')
if not os.path.exists(csv_path):
    print(f"ERROR: {csv_path} not found.")
    print("Please copy the CSV into data/raw/ first.")
    sys.exit(1)

df = pd.read_csv(csv_path)

print(f"\nShape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nBasic Stats:\n{df.describe()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")

# ── Target column ──────────────────────────────────────────────
# The target is 'delayed' with values 'yes'/'no'
target_col = 'delayed'
print(f"\nTarget Distribution ('{target_col}'):")
print(df[target_col].value_counts())
print(df[target_col].value_counts(normalize=True))

# ── Plot 1: Overview ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Delay distribution
df[target_col].value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Delay Distribution')
axes[0].set_xlabel('Delayed?')
axes[0].set_ylabel('Count')

# Delay by delivery partner
partner_col = 'delivery_partner'
delay_binary = (df[target_col] == 'yes').astype(int)
df['_delay_binary'] = delay_binary
df.groupby(partner_col)['_delay_binary'].mean().sort_values().plot(
    kind='barh', ax=axes[1], color='#3498db'
)
axes[1].set_title('Delay Rate by Partner')
axes[1].set_xlabel('Delay Rate')

# Delay by weather
weather_col = 'weather_condition'
df.groupby(weather_col)['_delay_binary'].mean().sort_values().plot(
    kind='barh', ax=axes[2], color='#e67e22'
)
axes[2].set_title('Delay Rate by Weather')
axes[2].set_xlabel('Delay Rate')

plt.tight_layout()
plt.savefig(os.path.join(NOTEBOOKS, 'eda_overview.png'), dpi=150)
print(f"\nSaved: {os.path.join(NOTEBOOKS, 'eda_overview.png')}")

# ── Plot 2: Correlation Matrix ─────────────────────────────────
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(NOTEBOOKS, 'correlation_matrix.png'), dpi=150)
print(f"Saved: {os.path.join(NOTEBOOKS, 'correlation_matrix.png')}")

# ── Plot 3: Distribution of key features ───────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

if 'distance_km' in df.columns:
    df['distance_km'].hist(bins=50, ax=axes[0, 0], color='#9b59b6', alpha=0.7)
    axes[0, 0].set_title('Distance (km) Distribution')

if 'package_weight_kg' in df.columns:
    df['package_weight_kg'].hist(bins=50, ax=axes[0, 1], color='#e74c3c', alpha=0.7)
    axes[0, 1].set_title('Package Weight (kg) Distribution')

if 'delivery_cost' in df.columns:
    df['delivery_cost'].hist(bins=50, ax=axes[1, 0], color='#2ecc71', alpha=0.7)
    axes[1, 0].set_title('Delivery Cost Distribution')

if 'delivery_rating' in df.columns:
    df['delivery_rating'].value_counts().sort_index().plot(
        kind='bar', ax=axes[1, 1], color='#3498db'
    )
    axes[1, 1].set_title('Delivery Rating Distribution')

plt.tight_layout()
plt.savefig(os.path.join(NOTEBOOKS, 'feature_distributions.png'), dpi=150)
print(f"Saved: {os.path.join(NOTEBOOKS, 'feature_distributions.png')}")

# ── Check class balance ────────────────────────────────────────
delay_ratio = df[target_col].value_counts(normalize=True)
print(f"\nClass Balance:")
for label, ratio in delay_ratio.items():
    print(f"  {label}: {ratio:.2%}")

if min(delay_ratio) < 0.3:
    print("  [!]  WARNING: Class imbalance detected. Will need handling.")
else:
    print("  [OK] Classes are reasonably balanced.")

# ── Records per delivery partner ──────────────────────────────
print(f"\nRecords per delivery partner:")
partner_counts = df[partner_col].value_counts()
print(partner_counts)
print(f"\nTotal partners: {len(partner_counts)}")

top_3_partners = partner_counts.head(3).index.tolist()
print(f"\nSelected for FL: {top_3_partners}")
for p in top_3_partners:
    subset = df[df[partner_col] == p]
    delay_rate = (subset[target_col] == 'yes').mean()
    print(f"  {p}: {len(subset)} records, {delay_rate:.2%} delay rate")

# Cleanup temp column
df.drop(columns=['_delay_binary'], inplace=True)

print("\n[OK] EDA complete. Check notebooks/ folder for plots.")
