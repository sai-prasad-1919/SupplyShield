# SupplyShield Data

## Datasets

### Primary: India Multi-Partner Delivery Logistics
- **Purpose:** Delay prediction (binary classification)
- **Size:** ~25K records
- **Key columns:** delivery_partner, distance_km, weather, traffic, vehicle_type, is_late
- **Kaggle:** https://www.kaggle.com/datasets/kundanbedmutha/delivery-logistics-dataset-india-multi-partner
- **Partitioning:** Split by `delivery_partner` column → NovaMart / TitanElec / SwiftLog

### Supplementary: DataCo Smart Supply Chain
- **Purpose:** Additional delay experimentation
- **Size:** ~180K records, 53 features
- **Kaggle:** https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

### Demand Forecasting: Walmart Sales
- **Purpose:** Demand prediction (time-series)
- **Size:** 45 stores
- **Kaggle:** https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting

## Download Instructions

1. Download the India Multi-Partner dataset CSV and place it in `data/raw/`
2. Run preprocessing: `python -m training.preprocess`
3. Partitions will be created automatically in `data/partitions/{novamart,titanelec,swiftlog}/`

## Directory Layout

```
data/
├── raw/            # Original downloaded CSVs (gitignored)
├── processed/      # Cleaned, feature-engineered data
└── partitions/     # Per-org splits for FL training
    ├── novamart/
    ├── titanelec/
    └── swiftlog/
```
