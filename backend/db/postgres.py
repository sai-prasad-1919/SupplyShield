"""
SupplyShield — PostgreSQL connection helpers.
One connection per org, resolved at login via MongoDB's postgres_db field.
"""
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from typing import Optional

PG_USER = "postgres"
PG_PASSWORD = "postgres1919"
PG_HOST = "localhost"
PG_PORT = 5432


def get_admin_conn():
    """Connect to the postgres admin DB (for CREATE DATABASE)."""
    conn = psycopg2.connect(
        dbname="postgres",
        user=PG_USER,
        password=PG_PASSWORD,
        host=PG_HOST,
        port=PG_PORT
    )
    conn.autocommit = True
    return conn


def get_org_conn(db_name: str):
    """Connect to a specific org's PostgreSQL database."""
    return psycopg2.connect(
        dbname=db_name,
        user=PG_USER,
        password=PG_PASSWORD,
        host=PG_HOST,
        port=PG_PORT,
        cursor_factory=RealDictCursor
    )


def db_exists(admin_conn, db_name: str) -> bool:
    """Check if a database already exists."""
    cur = admin_conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
    exists = cur.fetchone() is not None
    cur.close()
    return exists


def create_org_db(admin_conn, db_name: str):
    """Create a new org database if it doesn't exist."""
    if not db_exists(admin_conn, db_name):
        cur = admin_conn.cursor()
        cur.execute(f'CREATE DATABASE "{db_name}"')
        cur.close()
        print(f"  [OK] Created database: {db_name}")
    else:
        print(f"  [SKIP] Database already exists: {db_name}")


CREATE_SHIPMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS shipments (
    id              SERIAL PRIMARY KEY,
    package_type    VARCHAR(50),
    vehicle_type    VARCHAR(50),
    delivery_mode   VARCHAR(50),
    region          VARCHAR(50),
    weather_condition VARCHAR(50),
    distance_km     FLOAT,
    package_weight_kg FLOAT,
    delivery_time_hours FLOAT,
    expected_time_hours FLOAT,
    delayed         INTEGER,
    delivery_rating FLOAT,
    delivery_cost   FLOAT,
    time_diff_hours FLOAT,
    created_at      TIMESTAMP DEFAULT NOW()
);
"""


def load_csv_to_postgres(db_name: str, csv_path: str, label_encoders: dict):
    """
    Load a partition CSV into the org's shipments table.
    Re-decodes integer-encoded columns back to string labels.
    """
    df = pd.read_csv(csv_path)

    # Decode categorical columns back to string labels
    cat_cols = ['package_type', 'vehicle_type', 'delivery_mode', 'region', 'weather_condition']
    for col in cat_cols:
        if col in label_encoders and col in df.columns:
            le = label_encoders[col]
            df[col] = le.inverse_transform(df[col].astype(int))

    conn = get_org_conn(db_name)
    cur = conn.cursor()
    cur.execute(CREATE_SHIPMENTS_TABLE)

    # Check if already loaded
    cur.execute("SELECT COUNT(*) FROM shipments")
    count = cur.fetchone()
    row_count = count['count'] if isinstance(count, dict) else count[0]

    if row_count > 0:
        print(f"  [SKIP] {db_name}.shipments already has {row_count} rows - skipping load")
        cur.close()
        conn.close()
        return

    # Bulk insert
    records = []
    for _, row in df.iterrows():
        records.append((
            row.get('package_type'), row.get('vehicle_type'),
            row.get('delivery_mode'), row.get('region'),
            row.get('weather_condition'), row.get('distance_km'),
            row.get('package_weight_kg'), row.get('delivery_time_hours'),
            row.get('expected_time_hours'), int(row.get('delayed', 0)),
            row.get('delivery_rating'), row.get('delivery_cost'),
            row.get('time_diff_hours')
        ))

    cur.executemany(
        """INSERT INTO shipments
           (package_type, vehicle_type, delivery_mode, region, weather_condition,
            distance_km, package_weight_kg, delivery_time_hours, expected_time_hours,
            delayed, delivery_rating, delivery_cost, time_diff_hours)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        records
    )
    conn.commit()
    print(f"  [OK] Loaded {len(records)} rows into {db_name}.shipments")
    cur.close()
    conn.close()
