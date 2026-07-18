"""
SupplyShield - Step 3: Train Local Baseline Models
Trains one model per organization (non-federated).
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from training.utils import prepare_splits, train_model, evaluate_model, plot_training_history

print("=" * 60)
print("  STEP 3: Train Local Baseline Models")
print("=" * 60)

os.makedirs(os.path.join(ROOT, 'evaluation'), exist_ok=True)
os.makedirs(os.path.join(ROOT, 'checkpoints'), exist_ok=True)

local_results = {}

for org_name in ['novamart', 'titanelec', 'swiftlog']:
    data_path = os.path.join(ROOT, 'data', 'partitions', org_name, 'data.csv')
    if not os.path.exists(data_path):
        print(f"  [!]  Skipping {org_name}: {data_path} not found")
        continue

    print(f"\n{'='*60}")
    print(f"  TRAINING: {org_name.upper()} (Local-Only Baseline)")
    print(f"{'='*60}")

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_splits(data_path)

    model, history = train_model(
        X_train, y_train, X_val, y_val,
        input_dim=X_train.shape[1],
        hidden_dims=[128, 64, 32],
        dropout_rate=0.3,
        lr=0.001,
        epochs=200,
        patience=15,
        model_save_path=os.path.join(ROOT, 'checkpoints', f'{org_name}_local_best.pt'),
    )

    plot_training_history(history, title=f"{org_name} Local Training")
    results = evaluate_model(model, X_test, y_test, title=f"{org_name} Local Model")
    local_results[org_name] = results

    # Diagnose fit quality
    train_f1 = history['train_f1'][-1] if history['train_f1'] else 0
    val_f1 = history['val_f1'][-1] if history['val_f1'] else 0
    gap = abs(train_f1 - val_f1)

    if train_f1 < 0.60:
        print(f"  [!]  UNDERFITTING detected (Train F1={train_f1:.4f})")
    elif gap > 0.15:
        print(f"  [!]  OVERFITTING detected (Gap={gap:.4f})")
    else:
        print(f"  [OK] Good fit (Train F1={train_f1:.4f}, Val F1={val_f1:.4f}, Gap={gap:.4f})")

# Summary
print(f"\n{'='*60}")
print(f"  LOCAL-ONLY BASELINE SUMMARY")
print(f"{'='*60}")
for org, res in local_results.items():
    print(f"  {org}: F1={res['f1']:.4f}, AUC={res['auc']:.4f}")
if local_results:
    avg_f1 = np.mean([r['f1'] for r in local_results.values()])
    avg_auc = np.mean([r['auc'] for r in local_results.values()])
    print(f"  Average F1:  {avg_f1:.4f}")
    print(f"  Average AUC: {avg_auc:.4f}")

print("\n[OK] Local training complete.")
