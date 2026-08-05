"""
SupplyShield - Suppliers Module Migration
Creates `suppliers` table in all 3 org PostgreSQL databases and seeds mock data.
Also adds nullable `supplier_id` FK to `shipments` (backward-compatible).
Usage: python -m backend.db.migrate_suppliers
"""
import random
import psycopg2

PG_USER     = "postgres"
PG_PASSWORD = "postgres1919"
PG_HOST     = "localhost"
PG_PORT     = 5432

ORG_DATABASES = {
    "novamart":  "supplyshield_novamart",
    "titanelec": "supplyshield_titanelec",
    "swiftlog":  "supplyshield_swiftlog",
}

SUPPLIER_NAMES = {
    "novamart": [
        "FreshPack Distributors", "QuickLoad Retail Suppliers", "BrightGoods Co.",
        "SafeBox Logistics", "StellarPack Industries", "VerdantRoute Suppliers",
        "NovaStar Retail Partners", "TrustBridge Goods", "PrimeLine Distributors",
        "GreenCart Wholesale", "SwiftBox Retailers", "UltraFast Delivery Co.",
        "CoreSupply Network", "MarketEdge Distributors", "GridPath Logistics",
        "FastFill Retail", "BlueHorizon Goods", "NextStep Distributors",
    ],
    "titanelec": [
        "CircuitPro Suppliers", "MetalCore Components", "PrecisionParts Ltd.",
        "HighVolt Electronics", "SmartChip Industries", "TechBridge Manufacturing",
        "SolidState Components", "PowerGrid Suppliers", "NanoParts India",
        "EliteElec Wholesale", "FusionTech Parts", "UltraVolt Distributors",
        "CoreCircuit Co.", "PureWire Industries", "TerraElec Manufacturing",
        "MicroPath Components", "Apex Parts and Supplies", "HorizonElec Ltd.",
    ],
    "swiftlog": [
        "FastFreight Partners", "SecureLoad Logistics", "RapidRoute Warehousing",
        "GridLine Transport", "SkyPath Cargo", "TrustFreight Solutions",
        "BridgeWay Logistics", "CrossRoute Partners", "ClearPath Warehousing",
        "NightRun Freight", "SolidRoute Transport", "MaxLoad Logistics",
        "SwiftDock Cargo", "AllRoute Solutions", "CargoBridge India",
        "PrimeFreight Co.", "ZeroDelay Logistics", "FarReach Cargo",
    ],
}

CATEGORIES = {
    "novamart":  ["FMCG", "Electronics", "Apparel", "Pharma", "Food & Beverage", "Home & Garden"],
    "titanelec": ["Electronics", "Raw Materials", "Industrial Components", "Semiconductors", "Metals", "Cables & Wiring"],
    "swiftlog":  ["Logistics", "Cold Chain", "Bulk Cargo", "Express Freight", "Hazardous Materials", "Auto Parts"],
}

REGIONS = ["north", "south", "east", "west", "central"]

CONTACTS = [
    ("Arjun Menon",     "arjun.menon@supplier.in",     "+91-9800001001"),
    ("Priya Sharma",    "priya.sharma@supplier.in",    "+91-9800002002"),
    ("Rahul Verma",     "rahul.verma@supplier.in",     "+91-9800003003"),
    ("Sunita Gupta",    "sunita.gupta@supplier.in",    "+91-9800004004"),
    ("Vikram Nair",     "vikram.nair@supplier.in",     "+91-9800005005"),
    ("Kavya Reddy",     "kavya.reddy@supplier.in",     "+91-9800006006"),
    ("Aman Srivastava", "aman.srivastava@supplier.in", "+91-9800007007"),
    ("Divya Pillai",    "divya.pillai@supplier.in",    "+91-9800008008"),
    ("Rohit Kumar",     "rohit.kumar@supplier.in",     "+91-9800009009"),
    ("Meera Joshi",     "meera.joshi@supplier.in",     "+91-9800010010"),
    ("Suresh Bhat",     "suresh.bhat@supplier.in",     "+91-9800011011"),
    ("Anjali Das",      "anjali.das@supplier.in",      "+91-9800012012"),
    ("Nikhil Tiwari",   "nikhil.tiwari@supplier.in",   "+91-9800013013"),
    ("Pooja Agarwal",   "pooja.agarwal@supplier.in",   "+91-9800014014"),
    ("Sanjay Iyer",     "sanjay.iyer@supplier.in",     "+91-9800015015"),
    ("Rekha Pandey",    "rekha.pandey@supplier.in",    "+91-9800016016"),
    ("Manoj Singh",     "manoj.singh@supplier.in",     "+91-9800017017"),
    ("Deepa Nambiar",   "deepa.nambiar@supplier.in",   "+91-9800018018"),
]

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS suppliers (
    id             SERIAL PRIMARY KEY,
    name           VARCHAR(150) NOT NULL,
    contact_name   VARCHAR(100),
    contact_email  VARCHAR(150),
    contact_phone  VARCHAR(30),
    region         VARCHAR(50)  NOT NULL DEFAULT 'north',
    category       VARCHAR(80)  NOT NULL DEFAULT 'General',
    lead_time_days INTEGER      NOT NULL DEFAULT 7,
    active         BOOLEAN      NOT NULL DEFAULT TRUE,
    joined_at      TIMESTAMP    DEFAULT NOW()
);
"""

ADD_FK_SQL   = "ALTER TABLE shipments ADD COLUMN IF NOT EXISTS supplier_id INTEGER REFERENCES suppliers(id) ON DELETE SET NULL;"
CLEAR_FK_SQL = "UPDATE shipments SET supplier_id = NULL;"
# Use DELETE + sequence reset instead of TRUNCATE to avoid FK constraint errors
DELETE_SQL   = "DELETE FROM suppliers;"
RESET_SEQ_SQL = "ALTER SEQUENCE suppliers_id_seq RESTART WITH 1;"

INSERT_SQL = """
INSERT INTO suppliers (name, contact_name, contact_email, contact_phone, region, category, lead_time_days, active)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
RETURNING id;
"""

UPDATE_SQL  = "UPDATE shipments SET supplier_id = %s WHERE id = %s;"
GET_IDS_SQL = "SELECT id FROM shipments ORDER BY id;"


def seed_org(org_key: str, db_name: str):
    print(f"\n  Migrating {db_name} ({org_key})...")
    conn = psycopg2.connect(
        dbname=db_name, user=PG_USER, password=PG_PASSWORD,
        host=PG_HOST, port=PG_PORT,
    )
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute(CREATE_TABLE_SQL)
    cur.execute(ADD_FK_SQL)
    cur.execute(CLEAR_FK_SQL)    # NULL out all FK refs first
    cur.execute(DELETE_SQL)      # Now safe to delete suppliers
    cur.execute(RESET_SEQ_SQL)   # Reset auto-increment to 1

    names    = SUPPLIER_NAMES[org_key]
    cats     = CATEGORIES[org_key]
    contacts = CONTACTS[:]
    random.shuffle(contacts)

    supplier_ids = []
    for i, name in enumerate(names):
        c = contacts[i % len(contacts)]
        cur.execute(INSERT_SQL, (
            name, c[0], c[1], c[2],
            random.choice(REGIONS),
            cats[i % len(cats)],
            random.randint(3, 14),
            True,
        ))
        supplier_ids.append(cur.fetchone()[0])

    cur.execute(GET_IDS_SQL)
    shp_ids  = [r[0] for r in cur.fetchall()]
    assigned = 0
    for shp_id in shp_ids:
        if random.random() < 0.75:
            cur.execute(UPDATE_SQL, (random.choice(supplier_ids), shp_id))
            assigned += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"  OK  {len(supplier_ids)} suppliers seeded, {assigned}/{len(shp_ids)} shipments linked")


def main():
    print("SupplyShield - Suppliers Module Migration")
    print("=" * 45)
    random.seed(42)
    for org_key, db_name in ORG_DATABASES.items():
        try:
            seed_org(org_key, db_name)
        except Exception as e:
            print(f"  ERROR {db_name}: {e}")
    print("\nDone.")


if __name__ == "__main__":
    main()
