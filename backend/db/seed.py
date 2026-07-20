"""
SupplyShield — Seed Script
Bootstraps PostgreSQL databases and MongoDB organization documents.
Run ONCE from the SupplyShield/ project root:
    venv\Scripts\python.exe -m backend.db.seed

Prints each org's generated Org ID — save these, they are shown only once.
"""
import os
import sys
import uuid
import joblib
from datetime import datetime
from passlib.context import CryptContext

# Force UTF-8 output on Windows (avoids cp1252 encoding errors)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


# Ensure the project root is on the path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

from backend.db.postgres import get_admin_conn, create_org_db, load_csv_to_postgres
from backend.db.mongo import get_orgs_collection, org_id_exists

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------------------------------------------------------------
# Org definitions
# ---------------------------------------------------------------------------
ORGS = [
    {
        "prefix": "NVM",
        "company_name": "Novamart Retail",
        "owner_name": "Arjun Raghavendra Menon",
        "password": "Novamart@1919",
        "postgres_db": "supplyshield_novamart",
        "data_csv": os.path.join(ROOT_DIR, "data", "partitions", "novamart", "data.csv"),
        "org_key": "novamart",
    },
    {
        "prefix": "TIT",
        "company_name": "TitanElec Manufacturing",
        "owner_name": "Meera Lakshmi Srinivasan",
        "password": "TitanElec@1919",
        "postgres_db": "supplyshield_titanelec",
        "data_csv": os.path.join(ROOT_DIR, "data", "partitions", "titanelec", "data.csv"),
        "org_key": "titanelec",
    },
    {
        "prefix": "SWL",
        "company_name": "SwiftLog Warehouse",
        "owner_name": "Vikram Aditya Sharma",
        "password": "SwiftLog@1919",
        "postgres_db": "supplyshield_swiftlog",
        "data_csv": os.path.join(ROOT_DIR, "data", "partitions", "swiftlog", "data.csv"),
        "org_key": "swiftlog",
    },
]

# ---------------------------------------------------------------------------
# Load label encoders for decoding CSV integer columns
# ---------------------------------------------------------------------------
ENCODERS_PATH = os.path.join(ROOT_DIR, "checkpoints", "label_encoders.pkl")
label_encoders = joblib.load(ENCODERS_PATH) if os.path.exists(ENCODERS_PATH) else {}


def generate_org_id(prefix: str) -> str:
    """Generate a unique Org ID like NVM-A1B2C3D4."""
    uid = uuid.uuid4().hex[:8].upper()
    return f"{prefix}-{uid}"


def seed():
    print("\n" + "=" * 60)
    print("  SupplyShield — Database Seed Script")
    print("=" * 60)

    # Step 1: Create PostgreSQL databases
    print("\n[DB] Step 1: Setting up PostgreSQL databases...\n")
    admin_conn = get_admin_conn()

    for org in ORGS:
        print(f"  [{org['company_name']}]")
        create_org_db(admin_conn, org["postgres_db"])

    admin_conn.close()

    # Step 2: Load CSV data into each database
    print("\n[CSV] Step 2: Loading partition data into PostgreSQL...\n")
    for org in ORGS:
        print(f"  [{org['company_name']}] -> {org['postgres_db']}")
        if os.path.exists(org["data_csv"]):
            load_csv_to_postgres(org["postgres_db"], org["data_csv"], label_encoders)
        else:
            print(f"  [ERROR] CSV not found: {org['data_csv']}")

    # Step 3: Seed MongoDB org documents
    print("\n[AUTH] Step 3: Seeding MongoDB organization records...\n")
    orgs_col = get_orgs_collection()

    generated_ids = []
    for org in ORGS:
        # Check if org already exists by company name
        existing = orgs_col.find_one({"company_name": org["company_name"]}, {"_id": 0})
        if existing:
            print(f"  [SKIP] [{org['company_name']}] already exists -- Org ID: {existing['org_id']}")
            generated_ids.append((org["company_name"], existing["org_id"]))
            continue

        # Generate unique org_id
        org_id = generate_org_id(org["prefix"])
        while org_id_exists(org_id):
            org_id = generate_org_id(org["prefix"])

        # Hash password
        password_hash = pwd_context.hash(org["password"])

        doc = {
            "org_id": org_id,
            "company_name": org["company_name"],
            "owner_name": org["owner_name"],
            "password_hash": password_hash,
            "postgres_db": org["postgres_db"],
            "org_key": org["org_key"],
            "created_at": datetime.utcnow().isoformat(),
        }
        orgs_col.insert_one(doc)
        generated_ids.append((org["company_name"], org_id))
        print(f"  [OK] [{org['company_name']}] created -- Org ID: {org_id}")

    # Final summary — IMPORTANT: save these IDs
    print("\n" + "=" * 60)
    print("  SEED COMPLETE -- SAVE THESE ORG IDs (shown only once):")
    print("=" * 60)
    for company_name, org_id in generated_ids:
        print(f"  {company_name:<35}  =>  {org_id}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    seed()
