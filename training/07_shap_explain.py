"""
SupplyShield - Step 7: SHAP Explainability
Computes SHAP values for model predictions and generates explanation plots.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from training.model import DelayPredictor
from training.utils import prepare_splits

print("=" * 60)
print("  STEP 7: SHAP Explainability")
print("=" * 60)

EVAL_DIR = os.path.join(ROOT, 'evaluation')
os.makedirs(EVAL_DIR, exist_ok=True)

# Load feature names from combined data
combined_path = os.path.join(ROOT, 'data', 'processed', 'combined.csv')
df = pd.read_csv(combined_path)
feature_names = [c for c in df.columns if c != 'delayed']

# Load federated model (or centralized as fallback)
fed_ckpt = os.path.join(ROOT, 'checkpoints', 'federated_global_best.pt')
cent_ckpt = os.path.join(ROOT, 'checkpoints', 'centralized_best.pt')

if os.path.exists(fed_ckpt):
    ckpt_path = fed_ckpt
    model_label = "Federated"
elif os.path.exists(cent_ckpt):
    ckpt_path = cent_ckpt
    model_label = "Centralized"
else:
    print("ERROR: No trained model found. Run training steps first.")
    sys.exit(1)

checkpoint = torch.load(ckpt_path, weights_only=False)
model = DelayPredictor(
    input_dim=checkpoint['input_dim'],
    hidden_dims=checkpoint['hidden_dims'],
    dropout_rate=checkpoint.get('dropout_rate', 0.3),
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"  Loaded {model_label} model from {ckpt_path}")

# Prepare data
X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_splits(combined_path)

# ── SHAP ──────────────────────────────────────────────────────
try:
    import shap

    def predict_fn(x):
        model.eval()
        with torch.no_grad():
            return model(torch.tensor(x, dtype=torch.float32)).numpy()

    # Use a small background set for KernelExplainer
    background = X_train[:100]
    explain_set = X_test[:50]

    print(f"\n  Computing SHAP values...")
    print(f"  Background samples: {len(background)}")
    print(f"  Explain samples: {len(explain_set)}")

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(explain_set)

    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, explain_set, feature_names=feature_names, show=False)
    plt.title(f'SHAP Feature Importance ({model_label} Model)')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: evaluation/shap_summary.png")

    # Bar plot (mean absolute SHAP)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, explain_set, feature_names=feature_names,
                      plot_type='bar', show=False)
    plt.title(f'Mean |SHAP| Feature Importance ({model_label} Model)')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'shap_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: evaluation/shap_bar.png")

    # Individual explanation force plot skipped due to matplotlib NotImplementedError in SHAP 
    print(f"  Skipped force plot (unsupported with matplotlib=True)")

    # Save SHAP values for backend use
    import joblib
    joblib.dump({
        'shap_values': shap_values,
        'feature_names': feature_names,
        'expected_value': explainer.expected_value,
    }, os.path.join(ROOT, 'checkpoints', 'shap_data.pkl'))
    print(f"  Saved: checkpoints/shap_data.pkl")

except ImportError:
    print("\n  [!]  SHAP not installed. Computing feature importance via permutation instead.")

    # Fallback: simple permutation importance
    from sklearn.metrics import f1_score as sk_f1

    def get_f1(X, y):
        with torch.no_grad():
            preds = model(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()
            return sk_f1(y, (preds > 0.5).astype(int), zero_division=0)

    base_f1 = get_f1(X_test, y_test)
    importances = []

    for i, name in enumerate(feature_names):
        X_perm = X_test.copy()
        np.random.seed(42)
        X_perm[:, i] = np.random.permutation(X_perm[:, i])
        perm_f1 = get_f1(X_perm, y_test)
        importances.append(base_f1 - perm_f1)

    # Plot
    sorted_idx = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)),
             [importances[i] for i in sorted_idx],
             color='#3498db')
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('F1 Drop (higher = more important)')
    plt.title(f'Permutation Feature Importance ({model_label} Model)')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'feature_importance.png'), dpi=150)
    plt.close()
    print(f"  Saved: evaluation/feature_importance.png")

print("\n[OK] Explainability analysis complete.")
