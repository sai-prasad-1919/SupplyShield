"""
SupplyShield — Main API v2
Data served from per-org PostgreSQL databases (not CSV files).
Per-org DB connection pools are initialised once at startup.
"""
import os
import sys

# Ensure the project root (parent of 'backend/') is on sys.path so that
# top-level packages (models, explainability, guidance, api, etc.) are found
# regardless of which directory uvicorn is launched from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor
from typing import Optional
from pydantic import BaseModel
import passlib.hash as phash
from models.delay_predictor import create_model
from models.xgboost_model import XGBoostPredictor
from explainability.shap_explainer import XGBoostExplainer, generate_explanation_dict
from guidance.llm_agent import get_guidance
from api.demand_forecast import forecast_demand

from backend.guidance.rules import evaluate_rules
from backend.auth.routes import router as auth_router, get_current_org

# --------------- Paths ---------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")

app = FastAPI(title="SupplyShield API v2", version="2.0.0")

# CORS must be registered BEFORE routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include auth routes
app.include_router(auth_router)

# --------------- Org Config ---------------
ORG_CONFIG = {
    "novamart":  {"display_name": "Novamart Retail",          "type": "Retail",          "prefix": "NVM", "postgres_db": "supplyshield_novamart"},
    "titanelec": {"display_name": "TitanElec Manufacturing",  "type": "Manufacturing",   "prefix": "TIT", "postgres_db": "supplyshield_titanelec"},
    "swiftlog":  {"display_name": "SwiftLog Warehouse",       "type": "Logistics",       "prefix": "SWL", "postgres_db": "supplyshield_swiftlog"},
}

PG_USER     = "postgres"
PG_PASSWORD = "postgres1919"
PG_HOST     = "localhost"
PG_PORT     = 5432

# --------------- Model Architecture ---------------
class DelayPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, dropout_rate=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# --------------- Globals ---------------
model = None
label_encoders = {}
scaler = None
global_threshold = 0.5
# One SimpleConnectionPool per org-db, keyed by db name.
# NOTE: ORDER BY RANDOM() is fine at ~2,800 rows/table; revisit if tables grow > 100k rows.
_pg_pools: dict[str, pg_pool.SimpleConnectionPool] = {}


@app.on_event("startup")
def load_assets():
    global model, label_encoders, scaler, global_threshold, _pg_pools
    print("Loading assets for SupplyShield v2...")

    # --- ML model (FL architecture: hidden_dim=64, no BatchNorm, outputs logits) ---
    model_path = os.path.join(CHECKPOINTS_DIR, "federated_global_best.pt")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    from models.delay_predictor import DelayPredictor as FLDelayPredictor
    model = FLDelayPredictor(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint.get('hidden_dim', 64),
        dropout=checkpoint.get('dropout', 0.2),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    encoders_path = os.path.join(CHECKPOINTS_DIR, "label_encoders.pkl")
    if os.path.exists(encoders_path):
        label_encoders = joblib.load(encoders_path)

    scaler_path = os.path.join(CHECKPOINTS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    thresh_path = os.path.join(CHECKPOINTS_DIR, "threshold.pkl")
    if os.path.exists(thresh_path):
        global_threshold = joblib.load(thresh_path)

    # --- PostgreSQL connection pools (one per org) ---
    for org_key, cfg in ORG_CONFIG.items():
        db_name = cfg["postgres_db"]
        try:
            _pg_pools[db_name] = pg_pool.SimpleConnectionPool(
                minconn=1, maxconn=5,
                dbname=db_name, user=PG_USER, password=PG_PASSWORD,
                host=PG_HOST, port=PG_PORT,
            )
            print(f"  [OK] Pool created for {db_name}")
        except Exception as e:
            print(f"  [WARN] Could not connect to {db_name}: {e}")

    # --- XGBoost model + SHAP TreeExplainer ---
    xgb_path = os.path.join(CHECKPOINTS_DIR, "xgboost_model.json")
    if os.path.exists(xgb_path):
        try:
            xgb_predictor = XGBoostPredictor()
            xgb_predictor.load(xgb_path)
            app.state.xgb_model    = xgb_predictor.model
            app.state.xgb_explainer = XGBoostExplainer(xgb_predictor.model, FEATURE_COLS)
            print(f"  [OK] XGBoost model loaded from {xgb_path}")
        except Exception as e:
            print(f"  [WARN] XGBoost load failed: {e}")
            app.state.xgb_model     = None
            app.state.xgb_explainer = None
    else:
        print(f"  [WARN] XGBoost checkpoint not found at {xgb_path}")
        app.state.xgb_model     = None
        app.state.xgb_explainer = None

    print("Assets loaded successfully.")



@app.on_event("shutdown")
def close_pools():
    for p in _pg_pools.values():
        p.closeall()


# 9 clean features (post leakage-fix — no delivery_time_hours / time_diff_hours / delivery_rating)
CAT_COLS = ['package_type', 'vehicle_type', 'delivery_mode', 'region', 'weather_condition']
NUM_COLS = ['distance_km', 'package_weight_kg', 'expected_time_hours', 'delivery_cost']
FEATURE_COLS = CAT_COLS + NUM_COLS


def predict_from_pg_row(row: dict) -> dict:
    """
    Run inference on a row fetched from PostgreSQL.
    PostgreSQL stores categorical columns as decoded strings (e.g. 'electronics').
    We re-encode them to ints via label_encoders before passing to the model.
    Encoders are fit on the full combined dataset so all orgs share the same classes.
    """
    encoded_features = []
    decoded_features = {}

    for col in CAT_COLS:
        raw_val = str(row.get(col, "")).lower().strip()
        decoded_features[col] = raw_val
        if col in label_encoders:
            le = label_encoders[col]
            if raw_val in le.classes_:
                encoded_features.append(float(le.transform([raw_val])[0]))
            else:
                # Unseen label fallback: use class index 0 (most stable default)
                print(f"  [WARN] Unseen label '{raw_val}' in column '{col}' — using fallback 0")
                encoded_features.append(0.0)
        else:
            encoded_features.append(0.0)

    for col in NUM_COLS:
        val = row.get(col, 0.0)
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = 0.0
        decoded_features[col] = val
        encoded_features.append(val)

    if scaler is not None:
        features_scaled = scaler.transform(np.array([encoded_features]))
        tensor_features = torch.tensor(features_scaled, dtype=torch.float32)
    else:
        tensor_features = torch.tensor([encoded_features], dtype=torch.float32)

    # Pass through analytics-only post-delivery columns (NOT fed to model)
    # time_diff_hours = actual - expected delivery time; used for delay impact display only
    for analytics_col in ['time_diff_hours', 'delivery_time_hours', 'delivery_rating']:
        if analytics_col in row:
            try:
                decoded_features[analytics_col] = float(row[analytics_col] or 0.0)
            except (TypeError, ValueError):
                decoded_features[analytics_col] = 0.0

    with torch.no_grad():
        # FL model outputs logits — apply sigmoid to get probability
        prob = torch.sigmoid(model(tensor_features)).item()

    is_delayed = prob > global_threshold
    risk_level = "High" if prob > (global_threshold + 0.15) else "Medium" if prob > (global_threshold - 0.1) else "Low"
    shipment_state = {**decoded_features, "delay_probability": prob, "risk_level": risk_level}

    return {
        "delay_probability": prob,
        "prediction": "Delayed" if is_delayed else "On-Time",
        "risk_level": risk_level,
        "features": decoded_features,
        "shipment_state": shipment_state,
    }


# --------------- PostgreSQL Data Fetch ---------------
def get_pg_shipments(org_key: str, postgres_db: str) -> list[dict]:
    """
    Fetch 30 random shipment rows from the org's PostgreSQL database,
    run ML inference on each, and return the enriched shipment list.
    Returns [] on any DB error (with a console warning) rather than crashing.
    NOTE: ORDER BY RANDOM() is efficient at current table sizes (~2800 rows).
    """
    if postgres_db not in _pg_pools:
        print(f"  [WARN] No connection pool for db '{postgres_db}'")
        return []

    pool = _pg_pools[postgres_db]
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # ORDER BY RANDOM() for a fresh sample each call — fine at ~2800 rows
            cur.execute(
                """
                SELECT id, package_type, vehicle_type, delivery_mode, region,
                       weather_condition, distance_km, package_weight_kg,
                       delivery_time_hours, expected_time_hours, delivery_rating,
                       delivery_cost, time_diff_hours
                FROM shipments
                ORDER BY RANDOM()
                LIMIT 30
                """)
            rows = cur.fetchall()
    except Exception as e:
        print(f"  [ERROR] DB query failed for {postgres_db}: {e}")
        return []
    finally:
        if conn:
            pool.putconn(conn)

    prefix = ORG_CONFIG.get(org_key, {}).get("prefix", "SHP")
    dest_map = {"north": "Delhi", "south": "Bangalore", "east": "Kolkata",
                "west": "Mumbai", "central": "Nagpur"}

    shipments = []
    for row in rows:
        row = dict(row)
        pg_id = row.get("id", 0)
        try:
            pred = predict_from_pg_row(row)
        except Exception as e:
            print(f"  [WARN] Inference failed for row id={pg_id}: {e}")
            continue

        dest = dest_map.get(pred["features"].get("region", ""), "Hub")
        guidance = evaluate_rules(pred["shipment_state"])

        pkg = pred["features"].get("package_type", "")
        shipments.append({
            "id": f"{prefix}-{pg_id:04d}",
            "route": f"Supplier to {dest}",
            "package_type": pkg.title(),
            "prediction": {
                "delay_probability": pred["delay_probability"],
                "prediction": pred["prediction"],
                "risk_level": pred["risk_level"],
            },
            "features": pred["features"],
            "guidance": {
                "text": guidance["message"],
                "rule_fired": guidance["reason"],
            },
        })

    risk_order = {"High": 0, "Medium": 1, "Low": 2}
    shipments.sort(key=lambda x: (
        risk_order[x["prediction"]["risk_level"]],
        -x["prediction"]["delay_probability"]
    ))
    return shipments


# --------------- Public Endpoints ---------------
@app.get("/api/orgs")
def get_orgs():
    return [{"id": k, **{kk: vv for kk, vv in v.items() if kk != "postgres_db"}}
            for k, v in ORG_CONFIG.items()]


# --------------- JWT-Protected Endpoints ---------------
@app.get("/api/kpis")
def get_kpis(current_org: dict = Depends(get_current_org)):
    org_key    = current_org.get("org_key", "")
    postgres_db = current_org.get("postgres_db", "")

    if not org_key or not postgres_db:
        raise HTTPException(status_code=400, detail="Invalid org token")

    try:
        shipments = get_pg_shipments(org_key, postgres_db)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")

    if not shipments:
        raise HTTPException(status_code=503, detail="Could not load shipment data from database")

    high_risk = [s for s in shipments if s["prediction"]["risk_level"] == "High"]
    delayed   = [s for s in shipments if s["prediction"]["risk_level"] in ("High", "Medium")]
    on_time   = [s for s in shipments if s["prediction"]["risk_level"] == "Low"]
    
    total_delay = sum(
        s["features"].get("time_diff_hours", 0)
        for s in delayed
        if (s["features"].get("time_diff_hours") or 0) > 0
    )
    avg_delay_hours = (total_delay / len(delayed)) if delayed else 0
    delay_str = f"{avg_delay_hours / 24:.1f} days" if avg_delay_hours > 24 else f"{avg_delay_hours:.1f} hours"

    return {
        "total_shipments": len(shipments),
        "on_time_count": len(on_time),
        "delayed_count": len(delayed),
        "high_risk_count": len(high_risk),
        "avg_delay": delay_str,
        "units_in_transit": f"{len(shipments) * 450:,}",
    }

@app.get("/api/federated/status")
def get_federated_status():
    """Returns FL training metadata and model sizes.
    
    fl_history.json has the shape:
      {"loss": [{"round":1,"value":0.25},...], "auc": [...], "f1": [...], ...}
    We zip them into per-round objects for the frontend chart.
    """
    import os
    import json

    fl_history_path = os.path.join(CHECKPOINTS_DIR, "fl_history.json")
    if not os.path.exists(fl_history_path):
        raise HTTPException(status_code=404, detail="FL history not found")

    with open(fl_history_path, "r", encoding="utf-8") as f:
        raw = json.load(f)  # dict of {metric: [{round, value}, ...]}

    def get_kb(filename):
        path = os.path.join(CHECKPOINTS_DIR, filename)
        return round(os.path.getsize(path) / 1024, 2) if os.path.exists(path) else 0

    # Build per-round list: [{"round":1, "loss":0.25, "auc":0.94, ...}, ...]
    n_rounds = len(raw.get("loss", []))
    convergence = []
    for i in range(n_rounds):
        entry = {"round": i + 1}
        for metric, values in raw.items():
            if i < len(values):
                entry[metric] = values[i].get("value", 0)
        convergence.append(entry)

    final_metrics = convergence[-1] if convergence else {}

    return {
        "organizations": ["novamart", "titanelec", "swiftlog"],
        "fl_rounds_completed": n_rounds,
        "local_epochs_per_round": 5,
        "final_metrics": final_metrics,
        "convergence": convergence,
        "model_sizes": {
            "fl_model_kb": get_kb("federated_global_best.pt"),
            "xgboost_kb":  get_kb("xgboost_model.json"),
            "lstm_kb_per_org": get_kb("novamart_lstm.pt"),
        },
        "fl_threshold": 0.28604,
        "xgb_threshold": 0.19732,
    }


@app.get("/api/shipments")
def get_shipments_api(current_org: dict = Depends(get_current_org)):
    org_key    = current_org.get("org_key", "")
    postgres_db = current_org.get("postgres_db", "")
    if not org_key or not postgres_db:
        raise HTTPException(status_code=400, detail="Invalid org token")
    try:
        return get_pg_shipments(org_key, postgres_db)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")


@app.get("/api/shipment/{shipment_id}")
def get_shipment_api(shipment_id: str, current_org: dict = Depends(get_current_org)):
    """
    Single-shipment lookup — queries PostgreSQL directly by row ID.
    The shipment_id format is '{PREFIX}-{row_id:04d}' e.g. 'NVM-0198' → id=198.
    This avoids the 404 bug that occurred when ORDER BY RANDOM() returned a
    different set of rows than the previous /api/shipments call.
    """
    org_key     = current_org.get("org_key", "")
    postgres_db = current_org.get("postgres_db", "")
    if not org_key or not postgres_db:
        raise HTTPException(status_code=400, detail="Invalid org token")

    # Parse the numeric row ID from the shipment_id string (e.g. "NVM-0198" → 198)
    try:
        row_id = int(shipment_id.split("-")[-1])
    except (ValueError, IndexError):
        raise HTTPException(status_code=400, detail=f"Invalid shipment ID format: '{shipment_id}'")

    if postgres_db not in _pg_pools:
        raise HTTPException(status_code=503, detail="Database connection unavailable")

    pool = _pg_pools[postgres_db]
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, package_type, vehicle_type, delivery_mode, region,
                       weather_condition, distance_km, package_weight_kg,
                       delivery_time_hours, expected_time_hours, delivery_rating,
                       delivery_cost, time_diff_hours
                FROM shipments
                WHERE id = %s
                """,
                (row_id,)
            )
            row = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")
    finally:
        if conn:
            pool.putconn(conn)

    if row is None:
        raise HTTPException(status_code=404, detail=f"Shipment '{shipment_id}' not found in database")

    row = dict(row)
    try:
        pred = predict_from_pg_row(row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    dest_map = {"north": "Delhi", "south": "Bangalore", "east": "Kolkata",
                "west": "Mumbai", "central": "Nagpur"}
    dest     = dest_map.get(pred["features"].get("region", ""), "Hub")
    guidance = evaluate_rules(pred["shipment_state"])
    prefix   = ORG_CONFIG.get(org_key, {}).get("prefix", "SHP")
    pkg      = pred["features"].get("package_type", "")

    return {
        "id": f"{prefix}-{row_id:04d}",
        "route": f"Supplier to {dest}",
        "package_type": pkg.title(),
        "prediction": {
            "delay_probability": pred["delay_probability"],
            "prediction":        pred["prediction"],
            "risk_level":        pred["risk_level"],
        },
        "features": pred["features"],
        "guidance": {
            "text":       guidance["message"],
            "rule_fired": guidance["reason"],
        },
    }



# --------------- Supplier Intelligence Endpoints ---------------

def _get_pool(postgres_db: str):
    """Return a connection from the org pool, or raise 503."""
    if postgres_db not in _pg_pools:
        raise HTTPException(status_code=503, detail="Database connection unavailable")
    return _pg_pools[postgres_db]


def _compute_supplier_metrics(cur, supplier_id: int) -> dict:
    """
    Compute analytical metrics for a supplier from their linked shipments.
    Returns a dict with aggregated statistics.
    """
    cur.execute("""
        SELECT
            COUNT(*)                                        AS shipment_count,
            AVG(delivery_rating)                            AS avg_delivery_rating,
            AVG(delivery_cost)                              AS avg_delivery_cost,
            AVG(delivery_time_hours)                        AS avg_delivery_time_hours,
            AVG(CASE WHEN time_diff_hours > 0
                     THEN time_diff_hours ELSE 0 END)       AS avg_delay_hours
        FROM shipments
        WHERE supplier_id = %s
    """, (supplier_id,))
    row = cur.fetchone()
    shipment_count        = int(row[0] or 0)
    avg_rating            = float(row[1] or 0.0)
    avg_cost              = float(row[2] or 0.0)
    avg_delivery_hrs      = float(row[3] or 0.0)
    avg_delay_hrs         = float(row[4] or 0.0)

    # On-time = delivery_time_hours <= expected_time_hours
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE delivery_time_hours <= expected_time_hours) AS on_time,
            COUNT(*) FILTER (WHERE delivery_time_hours >  expected_time_hours) AS delayed
        FROM shipments
        WHERE supplier_id = %s
    """, (supplier_id,))
    ot_row       = cur.fetchone()
    on_time      = int(ot_row[0] or 0)
    delayed      = int(ot_row[1] or 0)
    on_time_rate = round(on_time / shipment_count, 4) if shipment_count > 0 else 0.0
    delay_pct    = round(delayed / shipment_count, 4) if shipment_count > 0 else 0.0

    return {
        "shipment_count":        shipment_count,
        "on_time_count":         on_time,
        "delayed_count":         delayed,
        "on_time_rate":          on_time_rate,
        "delay_percentage":      delay_pct,
        "avg_delivery_rating":   round(avg_rating, 2),
        "avg_delivery_cost":     round(avg_cost, 2),
        "avg_delivery_time_hrs": round(avg_delivery_hrs, 2),
        "avg_delay_hours":       round(avg_delay_hrs, 2),
    }


def _risk_level_from_delay(delay_pct: float) -> str:
    """Risk tiers calibrated to the dataset's ~26.6% average delay rate."""
    if delay_pct >= 0.35:
        return "High"
    elif delay_pct >= 0.20:
        return "Medium"
    return "Low"


@app.get("/api/suppliers/kpis")
def get_supplier_kpis(current_org: dict = Depends(get_current_org)):
    """KPI summary strip: total suppliers, high-risk count, avg on-time rate."""
    org_key     = current_org.get("org_key", "")
    postgres_db = current_org.get("postgres_db", "")
    if not org_key or not postgres_db:
        raise HTTPException(status_code=400, detail="Invalid org token")

    pool = _get_pool(postgres_db)
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) AS total FROM suppliers WHERE active = TRUE")
            total = int(cur.fetchone()["total"])

            cur.execute("SELECT id FROM suppliers WHERE active = TRUE")
            sup_ids = [r["id"] for r in cur.fetchall()]

        high_risk = 0
        total_on_time_rate = 0.0
        for sid in sup_ids:
            with conn.cursor() as cur:
                metrics = _compute_supplier_metrics(cur, sid)
            if _risk_level_from_delay(metrics["delay_percentage"]) == "High":
                high_risk += 1
            total_on_time_rate += metrics["on_time_rate"]

        avg_on_time = round(total_on_time_rate / total, 2) if total > 0 else 0.0
    finally:
        if conn:
            pool.putconn(conn)

    return {
        "total_suppliers":   total,
        "high_risk_count":   high_risk,
        "avg_on_time_rate":  avg_on_time,
    }


@app.get("/api/suppliers")
def get_suppliers(current_org: dict = Depends(get_current_org)):
    """
    Returns all active suppliers for the logged-in org with computed risk metrics.
    """
    org_key     = current_org.get("org_key", "")
    postgres_db = current_org.get("postgres_db", "")
    if not org_key or not postgres_db:
        raise HTTPException(status_code=400, detail="Invalid org token")

    pool = _get_pool(postgres_db)
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, name, contact_name, contact_email, contact_phone,
                       region, category, lead_time_days, active, joined_at
                FROM suppliers
                WHERE active = TRUE
                ORDER BY name
            """)
            rows = [dict(r) for r in cur.fetchall()]

        result = []
        for row in rows:
            with conn.cursor() as cur:
                metrics = _compute_supplier_metrics(cur, row["id"])
            result.append({
                "id":              row["id"],
                "name":            row["name"],
                "category":        row["category"],
                "region":          row["region"],
                "lead_time_days":  row["lead_time_days"],
                "contact_name":    row["contact_name"],
                "contact_email":   row["contact_email"],
                "contact_phone":   row["contact_phone"],
                "active":          row["active"],
                "joined_at":       row["joined_at"].isoformat() if row.get("joined_at") else None,
                "metrics":         metrics,
                "risk_level":      _risk_level_from_delay(metrics["delay_percentage"]),
            })
    finally:
        if conn:
            pool.putconn(conn)

    return {"suppliers": result, "total": len(result)}


@app.get("/api/supplier/{supplier_id}")
def get_supplier(supplier_id: int, current_org: dict = Depends(get_current_org)):
    """Full supplier profile with computed performance metrics."""
    org_key     = current_org.get("org_key", "")
    postgres_db = current_org.get("postgres_db", "")
    if not org_key or not postgres_db:
        raise HTTPException(status_code=400, detail="Invalid org token")

    pool = _get_pool(postgres_db)
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, name, contact_name, contact_email, contact_phone,
                       region, category, lead_time_days, active, joined_at
                FROM suppliers WHERE id = %s
            """, (supplier_id,))
            row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail=f"Supplier {supplier_id} not found")

        row = dict(row)
        with conn.cursor() as cur:
            metrics = _compute_supplier_metrics(cur, supplier_id)

    finally:
        if conn:
            pool.putconn(conn)

    return {
        "id":             row["id"],
        "name":           row["name"],
        "category":       row["category"],
        "region":         row["region"],
        "lead_time_days": row["lead_time_days"],
        "contact_name":   row["contact_name"],
        "contact_email":  row["contact_email"],
        "contact_phone":  row["contact_phone"],
        "active":         row["active"],
        "joined_at":      row["joined_at"].isoformat() if row.get("joined_at") else None,
        "metrics":        metrics,
        "risk_level":     _risk_level_from_delay(metrics["delay_percentage"]),
    }


@app.get("/api/supplier/{supplier_id}/shipments")
def get_supplier_shipments(supplier_id: int, current_org: dict = Depends(get_current_org)):
    """Returns the last 20 shipments linked to this supplier, with predictions."""
    org_key     = current_org.get("org_key", "")
    postgres_db = current_org.get("postgres_db", "")
    if not org_key or not postgres_db:
        raise HTTPException(status_code=400, detail="Invalid org token")

    pool   = _get_pool(postgres_db)
    prefix = ORG_CONFIG.get(org_key, {}).get("prefix", "SHP")
    dest_map = {"north": "Delhi", "south": "Bangalore", "east": "Kolkata",
                "west": "Mumbai", "central": "Nagpur"}
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, package_type, vehicle_type, delivery_mode, region,
                       weather_condition, distance_km, package_weight_kg,
                       delivery_time_hours, expected_time_hours, delivery_rating,
                       delivery_cost, time_diff_hours
                FROM shipments
                WHERE supplier_id = %s
                ORDER BY id DESC
                LIMIT 20
            """, (supplier_id,))
            rows = [dict(r) for r in cur.fetchall()]
    finally:
        if conn:
            pool.putconn(conn)

    shipments = []
    for row in rows:
        pg_id = row.get("id", 0)
        try:
            pred = predict_from_pg_row(row)
        except Exception:
            continue
        dest = dest_map.get(pred["features"].get("region", ""), "Hub")
        guidance = evaluate_rules(pred["shipment_state"])
        shipments.append({
            "id":           f"{prefix}-{pg_id:04d}",
            "route":        f"Supplier to {dest}",
            "package_type": pred["features"].get("package_type", "").title(),
            "prediction": {
                "delay_probability": pred["delay_probability"],
                "risk_level":        pred["risk_level"],
            },
            "guidance": {"text": guidance["message"]},
        })

    return {"shipments": shipments, "total": len(shipments)}


# ─── Demand Forecast ──────────────────────────────────────────────────────────

@app.get("/api/demand/forecast")
def get_demand_forecast(
    weeks: int = 4,
    current_org: dict = Depends(get_current_org),
):
    """
    Return the org's LSTM demand forecast.
    Org is determined entirely from the JWT — no org param accepted.
    weeks: 1–12 (validated).
    """
    org_key = current_org.get("org_key", "")
    if not org_key:
        raise HTTPException(status_code=400, detail="Invalid org token")

    if not (1 <= weeks <= 12):
        raise HTTPException(status_code=422, detail="weeks must be between 1 and 12")

    result = forecast_demand(org_key, n_weeks=weeks)
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    return result


# ─── Shipment Explanation (FL + XGBoost + SHAP + LLM) ───────────────────────

class ExplainRequest(BaseModel):
    shipment_id: str  # e.g. "NVM-0198"


@app.post("/api/predict/explain")
def predict_explain(req: ExplainRequest, current_org: dict = Depends(get_current_org)):
    """Full AI pipeline: FL + XGBoost + SHAP + LLM/rules for a single shipment."""
    org_key     = current_org.get("org_key", "")
    postgres_db = current_org.get("postgres_db", "")
    if not org_key or not postgres_db:
        raise HTTPException(status_code=400, detail="Invalid org token")

    # ── 1. Validate shipment_id format and verify org ownership ───────────────
    expected_prefix = ORG_CONFIG.get(org_key, {}).get("prefix", "")
    parts = req.shipment_id.split("-")
    if len(parts) < 2 or parts[0].upper() != expected_prefix:
        raise HTTPException(
            status_code=403,
            detail=f"Shipment '{req.shipment_id}' does not belong to your organization.",
        )
    try:
        row_id = int(parts[-1])
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid shipment ID format: '{req.shipment_id}'")

    # ── 2. Fetch raw row from PostgreSQL ──────────────────────────────────────
    if postgres_db not in _pg_pools:
        raise HTTPException(status_code=503, detail="Database connection unavailable")

    pool = _pg_pools[postgres_db]
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, package_type, vehicle_type, delivery_mode, region,
                       weather_condition, distance_km, package_weight_kg,
                       delivery_time_hours, expected_time_hours, delivery_rating,
                       delivery_cost, time_diff_hours
                FROM shipments WHERE id = %s
                """,
                (row_id,),
            )
            row = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")
    finally:
        if conn:
            pool.putconn(conn)

    if row is None:
        raise HTTPException(status_code=404, detail=f"Shipment '{req.shipment_id}' not found")

    row = dict(row)

    # ── 3. FL prediction (feature engineering done inside predict_from_pg_row) ─
    fl_pred = predict_from_pg_row(row)
    fl_prob = fl_pred["delay_probability"]

    # ── 4. Encode features for XGBoost (same pipeline as FL) ─────────────────
    encoded_features = []
    for col in CAT_COLS:
        raw_val = str(row.get(col, "")).lower().strip()
        if col in label_encoders and raw_val in label_encoders[col].classes_:
            encoded_features.append(float(label_encoders[col].transform([raw_val])[0]))
        else:
            encoded_features.append(0.0)
    for col in NUM_COLS:
        try:
            encoded_features.append(float(row.get(col, 0.0)))
        except (TypeError, ValueError):
            encoded_features.append(0.0)

    features_scaled = scaler.transform(np.array([encoded_features]))

    # ── 5. XGBoost prediction ─────────────────────────────────────────────────
    xgb_prob = 0.0
    if hasattr(app.state, "xgb_model") and app.state.xgb_model is not None:
        xgb_prob = float(app.state.xgb_model.predict_proba(features_scaled)[0, 1])

    # ── 6. SHAP explanation ───────────────────────────────────────────────────
    top_drivers     = []
    guidance_result = {"recommendations": [], "guidance_source": "rule_based"}

    if hasattr(app.state, "xgb_explainer") and app.state.xgb_explainer is not None:
        shap_values = app.state.xgb_explainer.explain_instances(features_scaled)
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]

        feature_names = CAT_COLS + NUM_COLS
        raw_values    = [row.get(col, "") for col in CAT_COLS] + [row.get(col, 0.0) for col in NUM_COLS]

        explanation_dict = generate_explanation_dict(
            feature_names=feature_names,
            feature_values=raw_values,
            shap_values=shap_values,
            base_value=0.5,
            prediction_prob=xgb_prob,
        )
        explanation_dict["context"] = fl_pred["features"]

        # Fix: use rule_id key, not non-existent "flag" key
        rule_res = evaluate_rules(fl_pred["shipment_state"])
        explanation_dict["business_rules"] = (
            [rule_res["message"]] if rule_res.get("rule_id") != "fallback" else []
        )
        top_drivers = explanation_dict["explainability"]["top_drivers"]

        # ── 7. LLM / rule-based guidance ──────────────────────────────────────
        guidance_result = get_guidance(explanation_dict)

    # ── 8. Build consensus + final response ───────────────────────────────────
    fl_threshold  = global_threshold
    xgb_threshold = 0.1973  # calibrated during training/08_train_xgboost.py

    def _verdict(prob: float, threshold: float) -> str:
        return "Delayed" if prob >= threshold else "On-Time"

    return {
        "shipment_id": req.shipment_id,
        "prediction": {
            "risk_probability": round(fl_prob, 4),
            "risk_level":       fl_pred["risk_level"],
            "verdict":          _verdict(fl_prob, fl_threshold),
        },
        "consensus": [
            {
                "model":       "Federated Neural Network",
                "probability": round(fl_prob, 4),
                "verdict":     _verdict(fl_prob, fl_threshold),
            },
            {
                "model":       "XGBoost (TreeSHAP)",
                "probability": round(xgb_prob, 4),
                "verdict":     _verdict(xgb_prob, xgb_threshold),
            },
        ],
        "shap":            top_drivers,
        "recommendations": guidance_result["recommendations"],
        "guidance_source": guidance_result["guidance_source"],
    }


# ─── Human-in-the-Loop Feedback ──────────────────────────────────────────────

import json as _json
import uuid
from datetime import datetime, timezone


class FeedbackRequest(BaseModel):
    shipment_id:        str
    fl_probability:     float
    xgb_probability:    float
    risk_level:         str
    recommendations:    list
    guidance_source:    str
    decision:           str   # confirm | override | escalate
    override_reason:    str = ""
    alternative_action: str = ""


VALID_DECISIONS = {"confirm", "override", "escalate"}


@app.post("/api/guidance/feedback", status_code=201)
def submit_feedback(req: FeedbackRequest, current_org: dict = Depends(get_current_org)):
    """Log a manager's human decision about an AI recommendation to feedback_log.jsonl."""
    if req.decision not in VALID_DECISIONS:
        raise HTTPException(
            status_code=422,
            detail=f"decision must be one of: {', '.join(sorted(VALID_DECISIONS))}",
        )

    record = {
        "analysis_id":        str(uuid.uuid4()),
        "org":                current_org.get("org_key", ""),
        "shipment_id":        req.shipment_id,
        "fl_probability":     req.fl_probability,
        "xgb_probability":    req.xgb_probability,
        "risk_level":         req.risk_level,
        "recommendations":    req.recommendations,
        "guidance_source":    req.guidance_source,
        "decision":           req.decision,
        "override_reason":    req.override_reason,
        "alternative_action": req.alternative_action,
        "timestamp":          datetime.now(timezone.utc).isoformat(),  # server-generated UTC
    }

    log_path = os.path.join(CHECKPOINTS_DIR, "feedback_log.jsonl")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(record) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write feedback: {e}")

    return {"status": "logged", "analysis_id": record["analysis_id"]}

@app.get("/api/guidance/feedback/history")
def get_feedback_history(current_org: dict = Depends(get_current_org)):
    """Return paginated feedback log entries for the current org."""
    org_key = current_org.get("org_key", "")
    log_path = os.path.join(CHECKPOINTS_DIR, "feedback_log.jsonl")
    
    entries = []
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = _json.loads(line)
                        if record.get("org") == org_key:
                            entries.append(record)
                    except:
                        pass
                        
    # Sort newest first
    entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    summary = {
        "confirm": sum(1 for e in entries if e.get("decision") == "confirm"),
        "override": sum(1 for e in entries if e.get("decision") == "override"),
        "escalate": sum(1 for e in entries if e.get("decision") == "escalate"),
        "gemini_count": sum(1 for e in entries if e.get("guidance_source") == "gemini"),
        "rule_based_count": sum(1 for e in entries if e.get("guidance_source") != "gemini"),
    }
    
    return {
        "total": len(entries),
        "entries": entries,
        "summary": summary
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8001, reload=True)


