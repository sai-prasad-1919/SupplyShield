# SupplyShield

Predict supply chain disruptions without sharing your data using Federated Learning.

## Setup

```bash
cd SupplyShield
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure

```
SupplyShield/
├── data/
│   ├── raw/                  # Raw CSV datasets (copied from source)
│   ├── processed/            # Cleaned and combined data
│   └── partitions/           # Per-org partitions for FL
│       ├── novamart/
│       ├── titanelec/
│       └── swiftlog/
├── notebooks/                # EDA plots
├── evaluation/               # Evaluation plots and metrics
├── checkpoints/              # Saved model weights
├── training/                 # Training pipeline scripts
│   ├── 01_eda.py
│   ├── 02_preprocess.py
│   ├── 03_train_local.py
│   ├── 04_train_centralized.py
│   ├── 05_train_federated.py
│   ├── 06_evaluate_compare.py
│   ├── 07_shap_explain.py
│   └── model.py
├── backend/                  # FastAPI backend
├── frontend/                 # React frontend (Vite)
├── requirements.txt
└── README.md
```
