"""
SupplyShield - Step 4: Train Centralized Model (Upper Bound Baseline)
Trains one model on all combined data from all 3 orgs.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from training.utils import prepare_splits, train_model, evaluate_model, plot_training_history

print("=" * 60)
print("  STEP 4: Train Centralized Model (All Data Combined)")
print("=" * 60)

data_path = os.path.join(ROOT, 'data', 'processed', 'combined.csv')
if not os.path.exists(data_path):
    print(f"ERROR: {data_path} not found. Run 02_preprocess.py first.")
    sys.exit(1)

X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_splits(data_path)

centralized_model, cent_history = train_model(
    X_train, y_train, X_val, y_val,
    input_dim=X_train.shape[1],
    hidden_dims=[128, 64, 32],
    dropout_rate=0.3,
    lr=0.001,
    epochs=200,
    patience=15,
    model_save_path=os.path.join(ROOT, 'checkpoints', 'centralized_best.pt'),
)

plot_training_history(cent_history, title="Centralized Training")
centralized_results = evaluate_model(
    centralized_model, X_test, y_test, title="Centralized Model"
)

print(f"\n  Centralized Results: F1={centralized_results['f1']:.4f}, AUC={centralized_results['auc']:.4f}")
print("\n[OK] Centralized training complete.")
