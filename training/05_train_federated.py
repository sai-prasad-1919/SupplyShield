"""
SupplyShield - Step 5: Federated Learning Training (Flower Simulation)
Simulates FL across 3 orgs using Flower's simulation API.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from training.model import DelayPredictor
from training.utils import prepare_splits, evaluate_model

print("=" * 60)
print("  STEP 5: Federated Learning (Flower Simulation)")
print("=" * 60)

# ── Preload all org data ──────────────────────────────────────
ORG_NAMES = ['novamart', 'titanelec', 'swiftlog']
org_data = {}

for org in ORG_NAMES:
    path = os.path.join(ROOT, 'data', 'partitions', org, 'data.csv')
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run 02_preprocess.py first.")
        sys.exit(1)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_splits(path)
    org_data[org] = {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'scaler': scaler
    }

INPUT_DIM = org_data[ORG_NAMES[0]]['X_train'].shape[1]
HIDDEN_DIMS = [128, 64, 32]
DROPOUT = 0.3
LOCAL_EPOCHS = 3
NUM_ROUNDS = 20
BATCH_SIZE = 64
LR = 0.001

# ── Try Flower simulation ────────────────────────────────────
try:
    import flwr as fl
    from flwr.common import ndarrays_to_parameters

    class SupplyShieldClient(fl.client.NumPyClient):
        def __init__(self, org_name):
            self.org_name = org_name
            d = org_data[org_name]
            self.X_train = d['X_train']
            self.X_val = d['X_val']
            self.y_train = d['y_train']
            self.y_val = d['y_val']
            self.model = DelayPredictor(INPUT_DIM, HIDDEN_DIMS, DROPOUT)
            self.criterion = nn.BCELoss()

        def get_parameters(self, config):
            return [val.cpu().numpy() for val in self.model.state_dict().values()]

        def set_parameters(self, parameters):
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train_dataset = TensorDataset(
                torch.tensor(self.X_train, dtype=torch.float32),
                torch.tensor(self.y_train, dtype=torch.float32).unsqueeze(1)
            )
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)

            for _ in range(LOCAL_EPOCHS):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    loss = self.criterion(self.model(batch_X), batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

            return self.get_parameters(config={}), len(self.X_train), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            self.model.eval()

            val_dataset = TensorDataset(
                torch.tensor(self.X_val, dtype=torch.float32),
                torch.tensor(self.y_val, dtype=torch.float32).unsqueeze(1)
            )
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

            total_loss, correct, total = 0, 0, 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    preds = self.model(batch_X)
                    total_loss += self.criterion(preds, batch_y).item()
                    pred_labels = (preds > 0.5).int().squeeze().tolist()
                    true_labels = batch_y.int().squeeze().tolist()
                    if isinstance(pred_labels, int):
                        pred_labels = [pred_labels]
                        true_labels = [true_labels]
                    all_preds.extend(pred_labels)
                    all_labels.extend(true_labels)
                    correct += sum(1 for a, b in zip(pred_labels, true_labels) if a == b)
                    total += len(true_labels)

            f1 = f1_score(all_labels, all_preds, zero_division=0)
            return total_loss / max(len(val_loader), 1), total, {
                "accuracy": correct / max(total, 1),
                "f1": f1
            }

    def client_fn(cid):
        return SupplyShieldClient(ORG_NAMES[int(cid)]).to_client()

    # Custom strategy that captures the final global parameters
    class SaveModelFedAvg(fl.server.strategy.FedAvg):
        """FedAvg strategy that saves the latest aggregated parameters."""
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.final_parameters = None

        def aggregate_fit(self, server_round, results, failures):
            aggregated = super().aggregate_fit(server_round, results, failures)
            if aggregated is not None:
                self.final_parameters = aggregated[0]  # Parameters object
            return aggregated

    # Create initial parameters
    init_model = DelayPredictor(INPUT_DIM, HIDDEN_DIMS, DROPOUT)
    init_params = ndarrays_to_parameters(
        [val.cpu().numpy() for val in init_model.state_dict().values()]
    )

    strategy = SaveModelFedAvg(
        min_fit_clients=len(ORG_NAMES),
        min_evaluate_clients=len(ORG_NAMES),
        min_available_clients=len(ORG_NAMES),
        initial_parameters=init_params,
    )

    print(f"\n  Starting FL simulation: {NUM_ROUNDS} rounds, {len(ORG_NAMES)} clients")
    print(f"  Local epochs per round: {LOCAL_EPOCHS}")
    print(f"  Model: {INPUT_DIM}->{HIDDEN_DIMS}->1\n")

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(ORG_NAMES),
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print("\n  [OK] FL simulation complete!")

    # Extract final global model from the strategy's saved parameters
    from flwr.common import parameters_to_ndarrays
    final_ndarrays = parameters_to_ndarrays(strategy.final_parameters)
    global_model = DelayPredictor(INPUT_DIM, HIDDEN_DIMS, DROPOUT)
    params_dict = zip(global_model.state_dict().keys(), final_ndarrays)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    global_model.load_state_dict(state_dict, strict=True)
    global_model.eval()

    # Evaluate federated model on each org's test set
    print(f"\n{'='*60}")
    print(f"  FEDERATED MODEL EVALUATION")
    print(f"{'='*60}")

    fed_results = {}
    for org in ORG_NAMES:
        d = org_data[org]
        results = evaluate_model(global_model, d['X_test'], d['y_test'],
                                 title=f"{org} Federated Model")
        fed_results[org] = results

    # Save federated model
    torch.save({
        'model_state_dict': global_model.state_dict(),
        'input_dim': INPUT_DIM,
        'hidden_dims': HIDDEN_DIMS,
        'dropout_rate': DROPOUT,
        'num_rounds': NUM_ROUNDS,
    }, os.path.join(ROOT, 'checkpoints', 'federated_global_best.pt'))

    print(f"\n  Saved federated model to checkpoints/federated_global_best.pt")

    # Summary
    print(f"\n{'='*60}")
    print(f"  FEDERATED RESULTS SUMMARY")
    print(f"{'='*60}")
    for org, res in fed_results.items():
        print(f"  {org}: F1={res['f1']:.4f}, AUC={res['auc']:.4f}")
    avg_f1 = np.mean([r['f1'] for r in fed_results.values()])
    avg_auc = np.mean([r['auc'] for r in fed_results.values()])
    print(f"  Average F1:  {avg_f1:.4f}")
    print(f"  Average AUC: {avg_auc:.4f}")

except ImportError as e:
    print(f"\n  [!]  Flower not installed. Running manual FedAvg simulation instead.")
    print(f"  Error: {e}")

    # ── Manual FedAvg Implementation ──────────────────────────
    print(f"\n  Starting manual FedAvg: {NUM_ROUNDS} rounds, {len(ORG_NAMES)} clients")

    # Initialize global model
    global_model = DelayPredictor(INPUT_DIM, HIDDEN_DIMS, DROPOUT)

    for round_num in range(1, NUM_ROUNDS + 1):
        # Distribute global model to all clients
        global_state = global_model.state_dict()

        client_states = []
        client_sizes = []

        for org in ORG_NAMES:
            d = org_data[org]
            # Create local model with global weights
            local_model = DelayPredictor(INPUT_DIM, HIDDEN_DIMS, DROPOUT)
            local_model.load_state_dict({k: v.clone() for k, v in global_state.items()})

            # Train locally
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(local_model.parameters(), lr=LR)
            train_dataset = TensorDataset(
                torch.tensor(d['X_train'], dtype=torch.float32),
                torch.tensor(d['y_train'], dtype=torch.float32).unsqueeze(1)
            )
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

            local_model.train()
            for _ in range(LOCAL_EPOCHS):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(local_model(batch_X), batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                    optimizer.step()

            client_states.append(local_model.state_dict())
            client_sizes.append(len(d['X_train']))

        # FedAvg aggregation
        total_size = sum(client_sizes)
        new_state = OrderedDict()
        for key in global_state.keys():
            new_state[key] = sum(
                client_states[i][key] * (client_sizes[i] / total_size)
                for i in range(len(ORG_NAMES))
            )
        global_model.load_state_dict(new_state)

        # Evaluate
        if round_num % 5 == 0 or round_num == 1:
            global_model.eval()
            round_f1s = []
            for org in ORG_NAMES:
                d = org_data[org]
                with torch.no_grad():
                    preds = global_model(torch.tensor(d['X_val'], dtype=torch.float32))
                    pred_labels = (preds > 0.5).int().squeeze().numpy()
                    f1 = f1_score(d['y_val'], pred_labels, zero_division=0)
                    round_f1s.append(f1)
            avg_f1 = np.mean(round_f1s)
            print(f"  Round {round_num:2d}/{NUM_ROUNDS} | "
                  f"Avg Val F1: {avg_f1:.4f} | "
                  f"Per-org: {', '.join(f'{f:.3f}' for f in round_f1s)}")

    # Save and evaluate
    torch.save({
        'model_state_dict': global_model.state_dict(),
        'input_dim': INPUT_DIM,
        'hidden_dims': HIDDEN_DIMS,
        'dropout_rate': DROPOUT,
        'num_rounds': NUM_ROUNDS,
    }, os.path.join(ROOT, 'checkpoints', 'federated_global_best.pt'))

    print(f"\n{'='*60}")
    print(f"  FEDERATED MODEL EVALUATION")
    print(f"{'='*60}")

    fed_results = {}
    for org in ORG_NAMES:
        d = org_data[org]
        results = evaluate_model(global_model, d['X_test'], d['y_test'],
                                 title=f"{org} Federated Model")
        fed_results[org] = results

    print(f"\n{'='*60}")
    print(f"  FEDERATED RESULTS SUMMARY")
    print(f"{'='*60}")
    for org, res in fed_results.items():
        print(f"  {org}: F1={res['f1']:.4f}, AUC={res['auc']:.4f}")
    avg_f1 = np.mean([r['f1'] for r in fed_results.values()])
    avg_auc = np.mean([r['auc'] for r in fed_results.values()])
    print(f"  Average F1:  {avg_f1:.4f}")
    print(f"  Average AUC: {avg_auc:.4f}")

print("\n[OK] Federated training complete.")
