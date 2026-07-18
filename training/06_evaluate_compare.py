"""
SupplyShield - Step 6: Compare All Three Approaches
Generates comparison bar chart: Local vs Federated vs Centralized.
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from training.model import DelayPredictor
from training.utils import prepare_splits, evaluate_model

print("=" * 60)
print("  STEP 6: Compare Local vs Federated vs Centralized")
print("=" * 60)

ORG_NAMES = ['novamart', 'titanelec', 'swiftlog']
EVAL_DIR = os.path.join(ROOT, 'evaluation')
os.makedirs(EVAL_DIR, exist_ok=True)

# ── Load and evaluate all models on test sets ─────────────────

def load_model(checkpoint_path):
    """Load a saved model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = DelayPredictor(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims'],
        dropout_rate=checkpoint.get('dropout_rate', 0.3),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


# Local models
print("\n--- Local Models ---")
local_results = {}
for org in ORG_NAMES:
    ckpt = os.path.join(ROOT, 'checkpoints', f'{org}_local_best.pt')
    if not os.path.exists(ckpt):
        print(f"  [!]  {org}: no local checkpoint found")
        continue
    model = load_model(ckpt)
    _, _, X_test, _, _, y_test, _ = prepare_splits(
        os.path.join(ROOT, 'data', 'partitions', org, 'data.csv')
    )
    results = evaluate_model(model, X_test, y_test, title=f"{org} Local Comparison")
    local_results[org] = results

# Centralized model
print("\n--- Centralized Model ---")
cent_ckpt = os.path.join(ROOT, 'checkpoints', 'centralized_best.pt')
centralized_results = {}
if os.path.exists(cent_ckpt):
    cent_model = load_model(cent_ckpt)
    _, _, X_test_c, _, _, y_test_c, _ = prepare_splits(
        os.path.join(ROOT, 'data', 'processed', 'combined.csv')
    )
    centralized_results = evaluate_model(cent_model, X_test_c, y_test_c,
                                          title="Centralized Comparison")
else:
    print("  [!]  No centralized checkpoint found")

# Federated model
print("\n--- Federated Model ---")
fed_ckpt = os.path.join(ROOT, 'checkpoints', 'federated_global_best.pt')
fed_results = {}
if os.path.exists(fed_ckpt):
    fed_model = load_model(fed_ckpt)
    for org in ORG_NAMES:
        _, _, X_test, _, _, y_test, _ = prepare_splits(
            os.path.join(ROOT, 'data', 'partitions', org, 'data.csv')
        )
        results = evaluate_model(fed_model, X_test, y_test,
                                  title=f"{org} Federated Comparison")
        fed_results[org] = results
else:
    print("  [!]  No federated checkpoint found")

# ── Comparison Bar Chart ──────────────────────────────────────
if local_results and fed_results and centralized_results:
    orgs = [o for o in ORG_NAMES if o in local_results and o in fed_results]
    x = np.arange(len(orgs))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # F1 Score comparison
    local_f1s = [local_results[o]['f1'] for o in orgs]
    fed_f1s = [fed_results[o]['f1'] for o in orgs]
    cent_f1 = [centralized_results['f1']] * len(orgs)

    axes[0].bar(x - width, local_f1s, width, label='Local-Only', color='#e74c3c')
    axes[0].bar(x, fed_f1s, width, label='Federated (FedAvg)', color='#2ecc71')
    axes[0].bar(x + width, cent_f1, width, label='Centralized', color='#3498db')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('F1 Score: Local vs Federated vs Centralized')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([o.upper() for o in orgs])
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    # Add value labels
    for i, (l, f, c) in enumerate(zip(local_f1s, fed_f1s, cent_f1)):
        axes[0].text(i - width, l + 0.02, f'{l:.3f}', ha='center', va='bottom', fontsize=8)
        axes[0].text(i, f + 0.02, f'{f:.3f}', ha='center', va='bottom', fontsize=8)
        axes[0].text(i + width, c + 0.02, f'{c:.3f}', ha='center', va='bottom', fontsize=8)

    # AUC comparison
    local_aucs = [local_results[o]['auc'] for o in orgs]
    fed_aucs = [fed_results[o]['auc'] for o in orgs]
    cent_auc = [centralized_results['auc']] * len(orgs)

    axes[1].bar(x - width, local_aucs, width, label='Local-Only', color='#e74c3c')
    axes[1].bar(x, fed_aucs, width, label='Federated (FedAvg)', color='#2ecc71')
    axes[1].bar(x + width, cent_auc, width, label='Centralized', color='#3498db')
    axes[1].set_ylabel('AUC-ROC')
    axes[1].set_title('AUC-ROC: Local vs Federated vs Centralized')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([o.upper() for o in orgs])
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    for i, (l, f, c) in enumerate(zip(local_aucs, fed_aucs, cent_auc)):
        axes[1].text(i - width, l + 0.02, f'{l:.3f}', ha='center', va='bottom', fontsize=8)
        axes[1].text(i, f + 0.02, f'{f:.3f}', ha='center', va='bottom', fontsize=8)
        axes[1].text(i + width, c + 0.02, f'{c:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'comparison.png'), dpi=150)
    plt.close()
    print(f"\n  Saved: evaluation/comparison.png")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'F1 Score':>10} {'AUC-ROC':>10}")
    print(f"  {'-'*45}")
    for org in orgs:
        print(f"  {org+' (Local)':<25} {local_results[org]['f1']:>10.4f} {local_results[org]['auc']:>10.4f}")
    print(f"  {'-'*45}")
    for org in orgs:
        print(f"  {org+' (Federated)':<25} {fed_results[org]['f1']:>10.4f} {fed_results[org]['auc']:>10.4f}")
    print(f"  {'-'*45}")
    print(f"  {'Centralized':<25} {centralized_results['f1']:>10.4f} {centralized_results['auc']:>10.4f}")
    print(f"  {'-'*45}")

    # Improvement analysis
    avg_local = np.mean([local_results[o]['f1'] for o in orgs])
    avg_fed = np.mean([fed_results[o]['f1'] for o in orgs])
    improvement = ((avg_fed - avg_local) / avg_local) * 100 if avg_local > 0 else 0
    print(f"\n  Avg Local F1:      {avg_local:.4f}")
    print(f"  Avg Federated F1:  {avg_fed:.4f}")
    print(f"  Improvement:       {improvement:+.2f}%")
    print(f"  Centralized F1:    {centralized_results['f1']:.4f}")

else:
    print("\n  [!]  Cannot create comparison — missing one or more model results.")

print("\n[OK] Comparison complete.")
