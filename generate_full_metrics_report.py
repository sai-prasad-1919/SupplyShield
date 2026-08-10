"""
SupplyShield - Full Model Performance Report
============================================

Generates a comprehensive metrics report covering:
  1. All model evaluations (Centralized, 3 x Local, Federated Global)
  2. SHAP feature importance (bar + beeswarm + waterfall for best model)
  3. Federated Learning round-by-round performance tracking
  4. Cross-model comparative analysis (Accuracy, F1, AUC, Precision, Recall)
  5. Self-contained HTML report combining all plots + metrics table

Usage:
    python generate_full_metrics_report.py

Output:
    reports/full_metrics_report.html   <- open in browser
    reports/figures/                   <- all generated PNG plots
    reports/metrics_summary.json       <- raw metrics JSON
"""

import sys
import os
import json
import base64
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# project root on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix, roc_curve, average_precision_score
)
from sklearn.model_selection import train_test_split

from models.delay_predictor import DelayPredictor, create_model
from config import DATA_PROCESSED_DIR

# ── global style
PALETTE = {
    "centralized": "#6C63FF",
    "novamart":    "#FF6584",
    "titanelec":   "#43B89C",
    "swiftlog":    "#F7B731",
    "federated":   "#FF9F43",
}

REPORT_DIR = ROOT / "reports"
FIG_DIR    = REPORT_DIR / "figures"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    csv_path = DATA_PROCESSED_DIR / "combined.csv"
    df = pd.read_csv(csv_path)
    target = "delayed"
    X = df.drop(columns=[target]).values.astype(np.float32)
    y = df[target].values.astype(np.float32)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    val_ratio = 0.15 / (1 - 0.15)
    X_train, _, y_train, _ = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )

    scaler = joblib.load(ROOT / "checkpoints" / "scaler.pkl")
    X_test_s  = scaler.transform(X_test).astype(np.float32)
    X_train_s = scaler.transform(X_train).astype(np.float32)

    try:
        meta = joblib.load(DATA_PROCESSED_DIR / "metadata.pkl")
        feature_names = meta.get("feature_names", [f"f{i}" for i in range(X.shape[1])])
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    return X_train_s, X_test_s, y_train, y_test, feature_names, df


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = ROOT / "checkpoints"

MODEL_CONFIGS = {
    "centralized": CHECKPOINT_DIR / "centralized_best.pt",
    "novamart":    CHECKPOINT_DIR / "novamart_local_best.pt",
    "titanelec":   CHECKPOINT_DIR / "titanelec_local_best.pt",
    "swiftlog":    CHECKPOINT_DIR / "swiftlog_local_best.pt",
    "federated":   CHECKPOINT_DIR / "federated_global_best.pt",
}

MODEL_LABELS = {
    "centralized": "Centralized",
    "novamart":    "NovaMart (Local)",
    "titanelec":   "TitanElec (Local)",
    "swiftlog":    "SwiftLog (Local)",
    "federated":   "Federated Global",
}


def load_checkpoint_model(path: Path, input_dim: int):
    if not path.exists():
        print(f"  [SKIP] Not found: {path.name}")
        return None
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
            # Detect architecture: FL model uses 'hidden_dim' (int, no BatchNorm),
            # centralized/local use 'hidden_dims' (list, with BatchNorm)
            if "hidden_dim" in ckpt and "hidden_dims" not in ckpt:
                # FL lightweight architecture (models.delay_predictor)
                hidden_dim = ckpt.get("hidden_dim", 64)
                dropout    = ckpt.get("dropout", 0.2)
                model = create_model(input_dim, hidden_dim=hidden_dim, dropout=dropout)
            else:
                # Centralized/local architecture (training.model.DelayPredictor)
                h_dims = ckpt.get("hidden_dims", [128, 64, 32])
                try:
                    from training.model import DelayPredictor as LegacyDP
                    model = LegacyDP(input_dim, h_dims, dropout_rate=0.3)
                except Exception:
                    model = create_model(input_dim)
            model.load_state_dict(state)
        else:
            model = create_model(input_dim)
            model.load_state_dict(ckpt)
        model.eval()
        print(f"  [OK] Loaded {path.name}")
        return model
    except Exception as e:
        print(f"  [ERR] {path.name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, threshold=0.5, apply_sigmoid=True):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        raw = model(X_t)
        if raw.dim() > 1:
            raw = raw.squeeze(1)
        # Only apply sigmoid if the model does NOT already have one built in
        probs = torch.sigmoid(raw).numpy() if apply_sigmoid else raw.numpy()

    preds = (probs >= threshold).astype(int)
    acc  = accuracy_score(y_test, preds)
    f1   = f1_score(y_test, preds, zero_division=0)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs)
        ap  = average_precision_score(y_test, probs)
    except ValueError:
        auc = ap = 0.0

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy":      round(acc,  4),
        "f1":            round(f1,   4),
        "precision":     round(prec, 4),
        "recall":        round(rec,  4),
        "specificity":   round(spec, 4),
        "auc_roc":       round(auc,  4),
        "avg_precision": round(ap,   4),
        "confusion_matrix": cm.tolist(),
        "probs":  probs,
        "preds":  preds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. SHAP
# ─────────────────────────────────────────────────────────────────────────────

def normalize_shap(shap_vals, n_features: int) -> np.ndarray:
    """Normalize shap output to shape (n_samples, n_features)."""
    # KernelExplainer binary returns list[2]: [class0, class1]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    shap_vals = np.asarray(shap_vals, dtype=np.float64)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 0]
    if shap_vals.ndim == 1:
        shap_vals = shap_vals[np.newaxis, :]
    # Clip to known feature count
    if shap_vals.shape[1] > n_features:
        shap_vals = shap_vals[:, :n_features]
    return shap_vals


def compute_shap(model, X_train, X_test, feature_names):
    """Return (shap_values [n,f], X_np [n,f], base_value float)."""
    model.eval()
    n_f = len(feature_names)

    class _Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            out = self.m(x)
            if out.dim() == 1:
                return out.unsqueeze(1)
            return torch.sigmoid(out)

    X_bg = torch.tensor(X_train[:200], dtype=torch.float32)
    X_ev = torch.tensor(X_test[:300], dtype=torch.float32)

    try:
        wrapped   = _Wrapper(model)
        explainer = shap.DeepExplainer(wrapped, X_bg)
        sv        = explainer.shap_values(X_ev)
        sv        = normalize_shap(sv, n_f)
        base      = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base_arr = np.asarray(base).flatten()
            base = float(base_arr[1]) if len(base_arr) > 1 else float(base_arr[0])
        else:
            base = float(base)
        print(f"  [SHAP] DeepExplainer shape: {sv.shape}  base={base:.4f}")
        return sv, X_ev.numpy()[:, :n_f], base
    except Exception as e:
        print(f"  [SHAP] DeepExplainer failed: {e}")
        print("  [SHAP] Falling back to KernelExplainer ...")

    def predict_fn(x):
        with torch.no_grad():
            t   = torch.tensor(x.astype(np.float32))
            out = model(t)
            if out.dim() > 1:
                out = out.squeeze(1)
            return torch.sigmoid(out).numpy()

    bg  = X_train[:50]
    ev  = X_test[:100]
    exp = shap.KernelExplainer(predict_fn, bg)
    sv  = exp.shap_values(ev, nsamples=200)
    sv  = normalize_shap(sv, n_f)
    base_raw = exp.expected_value
    if isinstance(base_raw, (list, np.ndarray)):
        base_arr = np.asarray(base_raw).flatten()
        base = float(base_arr[1]) if len(base_arr) > 1 else float(base_arr[0])
    else:
        base = float(base_raw)
    print(f"  [SHAP] KernelExplainer shape: {sv.shape}  base={base:.4f}")
    return sv, ev[:, :n_f], base


# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_and_b64(fig, filename: str) -> str:
    import io
    path = FIG_DIR / filename
    fig.savefig(str(path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def plot_confusion_matrices(all_metrics: dict) -> str:
    keys = list(all_metrics.keys())
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        cm = np.array(all_metrics[key]["confusion_matrix"])
        color = PALETTE.get(key, "#888")
        cmap = sns.light_palette(color, as_cmap=True)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                    xticklabels=["On-Time", "Delayed"],
                    yticklabels=["On-Time", "Delayed"],
                    linewidths=0.5, linecolor="white",
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(MODEL_LABELS[key], fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
    fig.suptitle("Confusion Matrices - All Models", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return save_and_b64(fig, "confusion_matrices.png")


def plot_roc_curves(all_metrics: dict, y_test: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1.5, label="Random (AUC=0.50)")
    for key, metrics in all_metrics.items():
        probs = metrics["probs"]
        try:
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc = metrics["auc_roc"]
            ax.plot(fpr, tpr, color=PALETTE.get(key, "#888"), lw=2.2,
                    label=f"{MODEL_LABELS[key]}  (AUC={auc:.4f})")
        except Exception:
            pass
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - All Models", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    fig.tight_layout()
    return save_and_b64(fig, "roc_curves.png")


def plot_metric_comparison(all_metrics: dict) -> str:
    metric_keys   = ["accuracy", "f1", "precision", "recall", "auc_roc"]
    metric_labels = ["Accuracy", "F1", "Precision", "Recall", "AUC-ROC"]
    n_models = len(all_metrics)
    x      = np.arange(len(metric_keys))
    bar_w  = 0.14

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, (key, metrics) in enumerate(all_metrics.items()):
        vals   = [metrics[m] for m in metric_keys]
        offset = (i - n_models / 2 + 0.5) * bar_w
        bars   = ax.bar(x + offset, vals, bar_w,
                        label=MODEL_LABELS[key],
                        color=PALETTE.get(key, "#888"),
                        edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison - Key Metrics", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, lw=1)
    fig.tight_layout()
    return save_and_b64(fig, "metric_comparison.png")


def plot_shap_bar(shap_values, feature_names: list) -> str:
    n_f = len(feature_names)
    sv  = normalize_shap(shap_values, n_f)
    importance = np.mean(np.abs(sv), axis=0)
    idx = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(8, max(4, n_f * 0.6)))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_f))[::-1]
    bars = ax.barh(
        [feature_names[i] for i in idx],
        importance[idx],
        color=[colors[j] for j in range(len(idx))],
        edgecolor="white", linewidth=0.4,
    )
    ax.set_xlabel("Mean |SHAP value| (average impact on model output)", fontsize=10)
    ax.set_title("SHAP Global Feature Importance\n(Best Model - Federated Global)", fontweight="bold")
    for bar, val in zip(bars, importance[idx]):
        ax.text(val + 0.0003, bar.get_y() + bar.get_height() / 2,
                f"  {val:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    return save_and_b64(fig, "shap_bar.png")


def plot_shap_beeswarm(shap_values, X_test_np, feature_names: list) -> str:
    n_f  = len(feature_names)
    sv   = normalize_shap(shap_values, n_f)
    Xnp  = np.asarray(X_test_np, dtype=np.float64)
    if Xnp.ndim == 2 and Xnp.shape[1] != n_f:
        Xnp = Xnp[:, :n_f]
    # Match rows
    n = min(sv.shape[0], Xnp.shape[0])
    sv   = sv[:n]
    Xnp  = Xnp[:n]

    plt.figure(figsize=(9, max(4, n_f * 0.65)))
    shap.summary_plot(sv, Xnp, feature_names=feature_names,
                      show=False, plot_size=None,
                      max_display=n_f, color_bar=True)
    plt.title("SHAP Beeswarm - Feature Impact Distribution\n(Best Model - Federated Global)",
              fontweight="bold")
    plt.tight_layout()
    fig = plt.gcf()
    return save_and_b64(fig, "shap_beeswarm.png")


def plot_shap_waterfall(shap_values, X_test_np, feature_names, base_value, instance_idx=0) -> str:
    n_f = len(feature_names)
    sv  = normalize_shap(shap_values, n_f)
    if instance_idx >= sv.shape[0]:
        instance_idx = 0
    row = sv[instance_idx]
    top_k = min(10, n_f)
    sorted_idx   = np.argsort(np.abs(row))[::-1][:top_k]
    names_sorted = [feature_names[i] for i in sorted_idx]
    vals_sorted  = row[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    running = float(base_value)
    starts, widths, clrs = [], [], []
    for v in vals_sorted[::-1]:
        starts.append(running)
        widths.append(float(v))
        clrs.append("#E74C3C" if v > 0 else "#2ECC71")
        running += float(v)

    y_pos = list(range(len(names_sorted)))
    ax.barh(y_pos, widths, left=starts[::-1],
            color=clrs[::-1], edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted[::-1], fontsize=9)
    ax.axvline(float(base_value), color="#888", linestyle="--", lw=1.2,
               alpha=0.6, label=f"Base value={base_value:.3f}")
    ax.set_xlabel("SHAP value contribution", fontsize=10)
    ax.set_title(f"SHAP Waterfall - Instance #{instance_idx}\n"
                 "Red = increases delay risk | Green = reduces risk", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return save_and_b64(fig, f"shap_waterfall_{instance_idx}.png")


def plot_shap_dependency(shap_values, X_test_np, feature_names: list) -> str:
    n_f  = len(feature_names)
    sv   = normalize_shap(shap_values, n_f)
    Xnp  = np.asarray(X_test_np, dtype=np.float64)
    if Xnp.ndim == 2 and Xnp.shape[1] != n_f:
        Xnp = Xnp[:, :n_f]
    n = min(sv.shape[0], Xnp.shape[0])
    sv  = sv[:n]
    Xnp = Xnp[:n]

    importance = np.mean(np.abs(sv), axis=0)
    top_feat   = int(np.argmax(importance))

    fig, ax = plt.subplots(figsize=(7, 5))
    feat_vals  = Xnp[:, top_feat]
    shap_feat  = sv[:, top_feat]
    sc = ax.scatter(feat_vals, shap_feat, c=feat_vals,
                    cmap="RdYlGn_r", alpha=0.6, s=20, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Feature value")
    ax.axhline(0, color="gray", lw=1, linestyle="--")
    ax.set_xlabel(feature_names[top_feat], fontsize=11)
    ax.set_ylabel("SHAP value", fontsize=11)
    ax.set_title(f"SHAP Dependency - {feature_names[top_feat]}\n(Top feature by mean |SHAP|)",
                 fontweight="bold")
    plt.tight_layout()
    return save_and_b64(fig, "shap_dependency.png")


def plot_fl_rounds() -> str:
    hist_candidates = [
        ROOT / "checkpoints" / "fl_history.json",
        ROOT / "models" / "saved" / "fl_history.json",
    ]
    data = None
    for p in hist_candidates:
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            break

    if data is None:
        raise FileNotFoundError(
            "\n[ERROR] Federated training history not found.\n"
            "Run FL training first:\n"
            "  python -m fl.simulate\n"
            "Expected: checkpoints/fl_history.json"
        )

    # Support both old flat format and new metric-keyed array format
    if "rounds" in data:
        # Legacy format from models/saved/
        df_r = pd.DataFrame(data["rounds"])
    else:
        # New format from checkpoints/
        # Build DataFrame from metric arrays
        df_r = pd.DataFrame({"round": [x["round"] for x in data["loss"]]})
        for metric in ["loss", "accuracy", "f1", "auc"]:
            if metric in data:
                df_r[metric] = [x["value"] for x in data[metric]]

    metrics_to_plot = [
        ("f1",       "F1 Score",  PALETTE["federated"]),
        ("auc",      "AUC-ROC",   PALETTE["centralized"]),
        ("accuracy", "Accuracy",  PALETTE["novamart"]),
        ("loss",     "Loss",      PALETTE["swiftlog"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (col, label, color) in zip(axes, metrics_to_plot):
        if col not in df_r.columns:
            continue
        ax.plot(df_r["round"], df_r[col], "o-", color=color,
                lw=2.2, ms=7, markeredgecolor="white", markeredgewidth=1.5)
        ax.fill_between(df_r["round"], df_r[col], alpha=0.12, color=color)
        ax.set_xlabel("FL Round", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"FL Validation {label} by Round", fontweight="bold")
        ax.set_xticks(df_r["round"])
        last_val = df_r[col].iloc[-1]
        ax.annotate(f"Final: {last_val:.4f}",
                    xy=(df_r["round"].iloc[-1], last_val),
                    xytext=(-30, 12), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    fig.suptitle("Federated Learning - Round-by-Round Training Progress",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return save_and_b64(fig, "fl_rounds.png")


def plot_client_comparison(all_metrics: dict) -> str:
    orgs       = ["novamart", "titanelec", "swiftlog"]
    org_labels = ["NovaMart", "TitanElec", "SwiftLog"]
    fed_m      = all_metrics.get("federated", {})
    metric_list   = ["accuracy", "f1", "precision", "recall", "auc_roc"]
    metric_labels = ["Acc", "F1", "Prec", "Recall", "AUC"]
    N = len(metric_list)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(polar=True))
    for ax, org, label in zip(axes, orgs, org_labels):
        loc_v = [all_metrics.get(org, {}).get(m, 0) for m in metric_list] + [all_metrics.get(org, {}).get(metric_list[0], 0)]
        fed_v = [fed_m.get(m, 0) for m in metric_list] + [fed_m.get(metric_list[0], 0)]

        ax.plot(angles, loc_v, "o-", color=PALETTE[org],        lw=2, ms=6, label="Local")
        ax.fill(angles, loc_v, alpha=0.15, color=PALETTE[org])
        ax.plot(angles, fed_v, "s-", color=PALETTE["federated"], lw=2, ms=6, label="Federated")
        ax.fill(angles, fed_v, alpha=0.10, color=PALETTE["federated"])
        ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(label, fontweight="bold", pad=16, fontsize=12)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Local vs. Federated - Per-Client Radar Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return save_and_b64(fig, "client_comparison.png")


def plot_class_distribution(y_test: np.ndarray) -> str:
    counts = pd.Series(y_test).value_counts().sort_index()
    labels = ["On-Time (0)", "Delayed (1)"]
    colors = ["#43B89C", "#FF6584"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].pie(counts.values, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=140,
                wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[0].set_title("Test Set Class Distribution", fontweight="bold")

    axes[1].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=2, width=0.5)
    for i, val in enumerate(counts.values):
        axes[1].text(i, val + 5, str(val), ha="center", va="bottom", fontweight="bold", fontsize=12)
    axes[1].set_ylabel("Count")
    axes[1].set_title("Sample Counts by Class", fontweight="bold")
    plt.tight_layout()
    return save_and_b64(fig, "class_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. HTML REPORT
# ─────────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>SupplyShield - Full Model Performance Report</title>
  <style>
    :root {{
      --bg:#0f1117;--card:#1a1d27;--border:#2a2d3a;
      --accent:#6C63FF;--text:#e2e8f0;--muted:#8892a4;
      --green:#43B89C;--red:#FF6584;--yellow:#F7B731;--orange:#FF9F43;
    }}
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;line-height:1.6;}}
    .hero{{background:linear-gradient(135deg,#1a1d27 0%,#0f1117 60%,#1a0a3a 100%);
      padding:60px 48px 40px;border-bottom:1px solid var(--border);position:relative;overflow:hidden;}}
    .hero::before{{content:'';position:absolute;top:-80px;right:-80px;width:320px;height:320px;
      border-radius:50%;background:radial-gradient(circle,rgba(108,99,255,0.15) 0%,transparent 70%);}}
    .hero h1{{font-size:2.4rem;font-weight:800;letter-spacing:-0.5px;
      background:linear-gradient(135deg,#6C63FF,#FF6584);-webkit-background-clip:text;
      -webkit-text-fill-color:transparent;background-clip:text;}}
    .hero p{{color:var(--muted);margin-top:8px;font-size:1rem;}}
    .badge{{display:inline-block;padding:3px 10px;border-radius:99px;font-size:0.78rem;
      font-weight:600;margin-right:8px;margin-top:14px;}}
    .bp{{background:rgba(108,99,255,0.2);color:#6C63FF;border:1px solid rgba(108,99,255,0.4);}}
    .bg{{background:rgba(67,184,156,0.2);color:#43B89C;border:1px solid rgba(67,184,156,0.4);}}
    .bo{{background:rgba(255,159,67,0.2);color:#FF9F43;border:1px solid rgba(255,159,67,0.4);}}
    main{{max-width:1400px;margin:0 auto;padding:40px 32px;}}
    section{{margin-bottom:52px;}}
    h2{{font-size:1.45rem;font-weight:700;margin-bottom:6px;color:var(--text);
      border-left:4px solid var(--accent);padding-left:14px;}}
    .sub{{color:var(--muted);font-size:0.88rem;margin-bottom:20px;padding-left:18px;}}
    .card{{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:24px;margin-bottom:20px;}}
    .mgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:14px;}}
    .mc{{background:var(--card);border:1px solid var(--border);border-radius:14px;
      padding:18px 16px;text-align:center;transition:transform 0.2s,border-color 0.2s;position:relative;overflow:hidden;}}
    .mc:hover{{transform:translateY(-3px);border-color:var(--accent);}}
    .mc .lbl{{font-size:0.76rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.8px;}}
    .mc .val{{font-size:1.9rem;font-weight:800;margin:6px 0;}}
    .mc::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;}}
    .mp::before{{background:var(--accent);}} .mr::before{{background:var(--red);}}
    .mg::before{{background:var(--green);}} .my::before{{background:var(--yellow);}}
    .mo::before{{background:var(--orange);}}
    table{{width:100%;border-collapse:collapse;font-size:0.88rem;}}
    th{{background:rgba(108,99,255,0.15);color:var(--accent);font-weight:700;
      padding:12px 16px;text-align:left;border-bottom:2px solid var(--border);}}
    td{{padding:10px 16px;border-bottom:1px solid var(--border);}}
    tr:hover td{{background:rgba(255,255,255,0.03);}}
    .best{{font-weight:700;color:#43B89C;}}
    img.chart{{width:100%;border-radius:12px;border:1px solid var(--border);background:#fff;}}
    .g2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;}}
    .g3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;}}
    @media(max-width:900px){{.g2,.g3{{grid-template-columns:1fr;}}}}
    .tab-nav{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px;}}
    .tab-btn{{padding:8px 18px;border-radius:99px;border:1px solid var(--border);
      background:transparent;color:var(--muted);cursor:pointer;font-size:0.84rem;transition:all 0.2s;}}
    .tab-btn.active,.tab-btn:hover{{background:var(--accent);color:#fff;border-color:var(--accent);}}
    .tab-pane{{display:none;}} .tab-pane.active{{display:block;}}
    footer{{text-align:center;padding:28px;color:var(--muted);font-size:0.8rem;
      border-top:1px solid var(--border);margin-top:40px;}}
    .pills{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;}}
    .pill{{padding:5px 14px;border-radius:99px;font-size:0.78rem;font-weight:600;
      display:flex;align-items:center;gap:6px;}}
    .dot{{width:8px;height:8px;border-radius:50%;}}
  </style>
</head>
<body>
<div class="hero">
  <h1>SupplyShield - Model Performance Report</h1>
  <p>Comprehensive evaluation of centralized, local, and federated learning models with SHAP explainability</p>
  <div>
    <span class="badge bp">Federated Learning</span>
    <span class="badge bg">SHAP Explainability</span>
    <span class="badge bo">Best: {best_model}</span>
    <span class="badge bp">Generated {timestamp}</span>
  </div>
</div>
<main>

  <section>
    <h2>Performance at a Glance</h2>
    <p class="sub">Best scores across all 5 models on the held-out test set</p>
    <div class="mgrid">{kpi_cards}</div>
  </section>

  <section>
    <h2>Full Metrics Comparison</h2>
    <p class="sub">All models evaluated on the same 15% stratified test split</p>
    <div class="card">
      <div class="pills">{model_pills}</div>
      {metrics_table}
    </div>
  </section>

  <section>
    <h2>Confusion Matrices</h2>
    <p class="sub">True positives (Delayed correctly detected) vs false negatives (missed delays)</p>
    <div class="card"><img class="chart" src="data:image/png;base64,{confusion_b64}"/></div>
  </section>

  <section>
    <h2>ROC Curves &amp; Metric Comparison</h2>
    <p class="sub">Receiver Operating Characteristic curves and grouped metric bars</p>
    <div class="g2">
      <div class="card"><img class="chart" src="data:image/png;base64,{roc_b64}"/></div>
      <div class="card"><img class="chart" src="data:image/png;base64,{compare_b64}"/></div>
    </div>
  </section>

  <section>
    <h2>SHAP Feature Explainability</h2>
    <p class="sub">SHapley Additive exPlanations - which features drive delay predictions most (Federated Global model)</p>
    <div class="tab-nav">
      <button class="tab-btn active" onclick="showTab('sb','shap-bar',this)">Feature Importance</button>
      <button class="tab-btn" onclick="showTab('sb','shap-bee',this)">Beeswarm</button>
      <button class="tab-btn" onclick="showTab('sb','shap-wf',this)">Waterfall</button>
      <button class="tab-btn" onclick="showTab('sb','shap-dep',this)">Dependency</button>
    </div>
    <div id="shap-bar" class="tab-pane active card"><img class="chart" src="data:image/png;base64,{shap_bar_b64}"/></div>
    <div id="shap-bee" class="tab-pane card"><img class="chart" src="data:image/png;base64,{shap_bee_b64}"/></div>
    <div id="shap-wf"  class="tab-pane card"><img class="chart" src="data:image/png;base64,{shap_wf_b64}"/></div>
    <div id="shap-dep" class="tab-pane card"><img class="chart" src="data:image/png;base64,{shap_dep_b64}"/></div>
  </section>

  <section>
    <h2>Federated Learning Progress</h2>
    <p class="sub">Round-by-round aggregated metrics across 3 organizations (NovaMart, TitanElec, SwiftLog)</p>
    <div style="background:rgba(108,99,255,0.05);border:1px solid rgba(108,99,255,0.2);border-radius:8px;padding:12px 16px;margin-bottom:16px;font-size:0.85rem;">
      <strong>Note:</strong> Round-level validation metrics monitor federated convergence across 10 rounds. Final model metrics (shown in the table above) are computed separately on the held-out test set after training completes.
    </div>
    <div class="card"><img class="chart" src="data:image/png;base64,{fl_rounds_b64}"/></div>
  </section>

  <section>
    <h2>Local vs. Federated - Per-Client Analysis</h2>
    <p class="sub">Radar comparison showing federated model benefit over each local model</p>
    <div class="card"><img class="chart" src="data:image/png;base64,{radar_b64}"/></div>
  </section>

  <section>
    <h2>Dataset Class Distribution</h2>
    <p class="sub">Test set composition - class imbalance context for interpreting metrics</p>
    <div class="card"><img class="chart" src="data:image/png;base64,{dist_b64}"/></div>
  </section>

</main>
<footer>
  SupplyShield - Federated Learning for Supply Chain Delay Prediction |
  Report generated {timestamp} | Models: {n_models} | Test samples: {n_test}
</footer>
<script>
function showTab(group, id, btn) {{
  document.querySelectorAll('#shap-bar,#shap-bee,#shap-wf,#shap-dep').forEach(p => p.classList.remove('active'));
  btn.closest('.tab-nav').querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}}
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 68)
    print("  SupplyShield - Full Model Performance Report Generator")
    print("=" * 68 + "\n")

    print("[1/7] Loading data ...")
    X_train, X_test, y_train, y_test, feature_names, df = load_data()
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}  Features: {len(feature_names)}")
    print(f"  Features: {feature_names}")
    print(f"  Test delay rate: {y_test.mean():.2%}")

    thresh_path = CHECKPOINT_DIR / "threshold.pkl"
    threshold = float(joblib.load(thresh_path)) if thresh_path.exists() else 0.5
    if isinstance(threshold, np.ndarray):
        threshold = float(threshold.flat[0])
    print(f"  Decision threshold: {threshold:.4f}")

    print("\n[2/7] Loading models ...")
    input_dim = X_test.shape[1]
    loaded_models = {}
    for key, path in MODEL_CONFIGS.items():
        m = load_checkpoint_model(path, input_dim)
        if m is not None:
            loaded_models[key] = m

    if not loaded_models:
        print("\n[ERROR] No models could be loaded.")
        return

    print("\n[3/7] Evaluating models ...")
    all_metrics = {}
    # training.model.DelayPredictor (centralized/local) has built-in Sigmoid — no double-sigmoid
    # models.delay_predictor.DelayPredictor (federated) outputs raw logits — apply sigmoid
    NEEDS_SIGMOID = {"federated"}
    for key, model in loaded_models.items():
        print(f"\n  -- {MODEL_LABELS[key]} --")
        model_threshold = threshold if key == "federated" else 0.5
        apply_sig = key in NEEDS_SIGMOID
        metrics = evaluate_model(model, X_test, y_test, threshold=model_threshold, apply_sigmoid=apply_sig)
        all_metrics[key] = metrics
        print(f"     Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
              f"AUC={metrics['auc_roc']:.4f}  Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}")

    json_path = REPORT_DIR / "metrics_summary.json"
    summary = {k: {m: v for m, v in met.items() if m not in ("probs", "preds")}
               for k, met in all_metrics.items()}
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {json_path}")

    print("\n[4/7] Computing SHAP values ...")
    shap_key = "federated" if "federated" in loaded_models else \
        max(all_metrics.keys(), key=lambda k: all_metrics[k]["f1"])
    print(f"  Model: {MODEL_LABELS[shap_key]}")
    shap_values, X_shap_np, base_value = compute_shap(
        loaded_models[shap_key], X_train, X_test, feature_names)

    print("\n[5/7] Generating plots ...")
    confusion_b64 = plot_confusion_matrices(all_metrics)
    print("  [OK] Confusion matrices")

    roc_b64 = plot_roc_curves(all_metrics, y_test)
    print("  [OK] ROC curves")

    compare_b64 = plot_metric_comparison(all_metrics)
    print("  [OK] Metric comparison bars")

    shap_bar_b64 = plot_shap_bar(shap_values, feature_names)
    print("  [OK] SHAP bar chart")

    shap_bee_b64 = plot_shap_beeswarm(shap_values, X_shap_np, feature_names)
    print("  [OK] SHAP beeswarm")

    delayed_idx = next((i for i, v in enumerate(y_test) if v == 1), 0)
    shap_wf_b64 = plot_shap_waterfall(shap_values, X_shap_np, feature_names, base_value, delayed_idx)
    print(f"  [OK] SHAP waterfall (instance #{delayed_idx})")

    shap_dep_b64 = plot_shap_dependency(shap_values, X_shap_np, feature_names)
    print("  [OK] SHAP dependency plot")

    fl_rounds_b64 = plot_fl_rounds()
    print("  [OK] FL round progress")

    radar_b64 = plot_client_comparison(all_metrics)
    print("  [OK] Per-client radar")

    dist_b64 = plot_class_distribution(y_test)
    print("  [OK] Class distribution")

    print("\n[6/7] Building HTML report ...")
    best_key   = max(all_metrics.keys(), key=lambda k: all_metrics[k]["f1"])
    best_label = MODEL_LABELS[best_key]

    # KPI cards
    kpi_data = [
        ("Best Accuracy",  max(m["accuracy"]  for m in all_metrics.values()), "mg", "%"),
        ("Best F1 Score",  max(m["f1"]        for m in all_metrics.values()), "mp", ""),
        ("Best AUC-ROC",   max(m["auc_roc"]   for m in all_metrics.values()), "mo", ""),
        ("Best Precision", max(m["precision"] for m in all_metrics.values()), "mr", ""),
        ("Best Recall",    max(m["recall"]    for m in all_metrics.values()), "my", ""),
        ("Test Samples",   len(y_test),                                        "mg", ""),
        ("Delay Rate",     float(y_test.mean()),                               "mr", "%"),
        ("Features Used",  len(feature_names),                                 "mp", ""),
    ]
    kpi_html = ""
    for label, val, cls, suf in kpi_data:
        disp = f"{val:.1%}" if suf == "%" else (f"{val:.4f}" if isinstance(val, float) else str(val))
        kpi_html += f'<div class="mc {cls}"><div class="lbl">{label}</div><div class="val">{disp}</div></div>'

    # Table
    metric_cols   = ["accuracy", "f1", "precision", "recall", "specificity", "auc_roc", "avg_precision"]
    metric_labels = ["Accuracy", "F1", "Precision", "Recall", "Specificity", "AUC-ROC", "Avg Precision"]
    best_vals = {m: max(all_metrics[k][m] for k in all_metrics) for m in metric_cols}

    rows = ""
    for key in all_metrics:
        row = f"<tr><td><strong>{MODEL_LABELS[key]}</strong></td>"
        for m in metric_cols:
            v = all_metrics[key][m]
            cls = ' class="best"' if abs(v - best_vals[m]) < 1e-9 else ""
            row += f"<td{cls}>{v:.4f}</td>"
        row += "</tr>"
        rows += row
    header = "<tr><th>Model</th>" + "".join(f"<th>{l}</th>" for l in metric_labels) + "</tr>"
    table_html = f"<table>{header}{rows}</table>"

    # Pills
    pill_colors = {"centralized": "#6C63FF", "novamart": "#FF6584",
                   "titanelec": "#43B89C",   "swiftlog": "#F7B731", "federated": "#FF9F43"}
    pills = ""
    for key in all_metrics:
        c = pill_colors.get(key, "#888")
        pills += (f'<span class="pill" style="background:rgba(0,0,0,0.3);border:1px solid {c};color:{c};">'
                  f'<span class="dot" style="background:{c};"></span>{MODEL_LABELS[key]}</span>')

    html = HTML_TEMPLATE.format(
        best_model    = best_label,
        timestamp     = datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_models      = len(all_metrics),
        n_test        = len(y_test),
        kpi_cards     = kpi_html,
        metrics_table = table_html,
        model_pills   = pills,
        confusion_b64 = confusion_b64,
        roc_b64       = roc_b64,
        compare_b64   = compare_b64,
        shap_bar_b64  = shap_bar_b64,
        shap_bee_b64  = shap_bee_b64,
        shap_wf_b64   = shap_wf_b64,
        shap_dep_b64  = shap_dep_b64,
        fl_rounds_b64 = fl_rounds_b64,
        radar_b64     = radar_b64,
        dist_b64      = dist_b64,
    )

    report_path = REPORT_DIR / "full_metrics_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("\n[7/7] Done!")
    print("\n" + "=" * 68)
    print(f"  HTML Report: {report_path}")
    print(f"  JSON:        {json_path}")
    print(f"  Figures:     {FIG_DIR}")
    print("=" * 68)
    bm = all_metrics[best_key]
    print(f"\n  Best Model:  {best_label}")
    print(f"  Accuracy:    {bm['accuracy']:.4f}")
    print(f"  F1 Score:    {bm['f1']:.4f}")
    print(f"  AUC-ROC:     {bm['auc_roc']:.4f}")
    print(f"  Precision:   {bm['precision']:.4f}")
    print(f"  Recall:      {bm['recall']:.4f}")
    print("\n  Open full_metrics_report.html in your browser!")


if __name__ == "__main__":
    main()
