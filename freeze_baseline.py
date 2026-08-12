import os
import json
import shutil
import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def freeze():
    print("Freezing Baseline v1...")
    baseline_dir = ROOT / "baseline" / "v1"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Collect Metadata
    meta = joblib.load(ROOT / "data/processed/metadata.pkl")
    thr = float(joblib.load(ROOT / "checkpoints/threshold.pkl"))
    
    data_splits = {}
    for org in ["novamart", "titanelec", "swiftlog"]:
        te = pd.read_csv(ROOT / "data/partitions" / org / "test.csv")
        target = "is_late" if "is_late" in te.columns else "delayed"
        data_splits[org] = {
            "test_samples": len(te),
            "delay_rate": float(te[target].mean())
        }
    
    with open(ROOT / "reports/metrics_summary.json", "r") as f:
        metrics = json.load(f)
        
    import sys; sys.path.insert(0, str(ROOT))
    from config import FL_ROUNDS, RANDOM_SEED
    
    baseline_data = {
        "version": "v1",
        "features": meta["feature_names"],
        "random_seed": RANDOM_SEED,
        "fl_rounds": FL_ROUNDS,
        "fl_aggregation": "FedAvg",
        "fl_clients": ["novamart", "titanelec", "swiftlog"],
        "decision_threshold": thr,
        "data_splits": data_splits,
        "final_test_metrics": metrics
    }
    
    with open(baseline_dir / "baseline_v1.json", "w") as f:
        json.dump(baseline_data, f, indent=4)
        
    print("  [OK] Saved baseline_v1.json")
    
    # 2. Copy Checkpoints
    ckpt_dir = ROOT / "checkpoints"
    for ext in ["*.pt", "*.pkl", "*.json"]:
        for fpath in ckpt_dir.glob(ext):
            shutil.copy2(fpath, baseline_dir / fpath.name)
            
    print(f"  [OK] Copied checkpoints to {baseline_dir.relative_to(ROOT)}")
    print("Baseline v1 frozen successfully.")

if __name__ == "__main__":
    freeze()
