"""
SupplyShield — MongoDB connection and org collection helpers.
Stores: org_id, company_name, owner_name, password_hash, postgres_db.
"""
import pymongo
from typing import Optional

MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "supplyshield_auth"
ORGS_COLLECTION = "organizations"

_client: Optional[pymongo.MongoClient] = None


def get_client() -> pymongo.MongoClient:
    global _client
    if _client is None:
        _client = pymongo.MongoClient(MONGO_URI)
    return _client


def get_db():
    return get_client()[MONGO_DB]


def get_orgs_collection():
    return get_db()[ORGS_COLLECTION]


def find_org_by_id(org_id: str) -> Optional[dict]:
    """Look up org document by org_id."""
    return get_orgs_collection().find_one({"org_id": org_id}, {"_id": 0})


def org_id_exists(org_id: str) -> bool:
    return get_orgs_collection().count_documents({"org_id": org_id}) > 0


def insert_org(org_doc: dict) -> str:
    """Insert a new org document. Returns the inserted org_id."""
    result = get_orgs_collection().insert_one(org_doc)
    return org_doc["org_id"]


def list_all_orgs() -> list:
    """Return all orgs (without password_hash)."""
    return list(get_orgs_collection().find(
        {}, {"_id": 0, "password_hash": 0}
    ))
