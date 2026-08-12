"""
SupplyShield — Auth API Routes
POST /api/auth/login  → {org_id, password} → JWT access_token
POST /api/auth/register → 501 Coming Soon
GET  /api/auth/me     → org details from JWT
"""
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

from backend.db.mongo import find_org_by_id

router = APIRouter(prefix="/api/auth", tags=["auth"])

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SECRET_KEY = "SUPPLYSHIELD_JWT_SECRET_2026_SECURE"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class LoginRequest(BaseModel):
    org_id: str
    password: str


class RegisterRequest(BaseModel):
    company_name: str
    owner_name: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    org_id: str
    company_name: str
    owner_name: str


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------
def create_access_token(org_doc: dict) -> str:
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": org_doc["org_id"],
        "company_name": org_doc["company_name"],
        "owner_name": org_doc["owner_name"],
        "postgres_db": org_doc["postgres_db"],
        "org_key": org_doc.get("org_key", ""),
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Dependency: get current org from JWT
# ---------------------------------------------------------------------------
def get_current_org(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)) -> dict:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_token(credentials.credentials)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    """
    Authenticate with Org ID + password.
    Returns a JWT valid for 24 hours.
    """
    try:
        org = find_org_by_id(req.org_id.strip().upper())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is temporarily unavailable. Please ensure MongoDB is running.",
        )

    if not org:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Organization ID not found. Check your credentials.",
        )

    if not pwd_context.verify(req.password, org["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password.",
        )

    token = create_access_token(org)

    return TokenResponse(
        access_token=token,
        org_id=org["org_id"],
        company_name=org["company_name"],
        owner_name=org["owner_name"],
    )


@router.post("/register", status_code=501)
def register(req: RegisterRequest):
    """Registration endpoint — reserved for future development."""
    raise HTTPException(
        status_code=501,
        detail={
            "message": "Self-registration is coming soon.",
            "note": "Please contact the SupplyShield team to onboard your organization.",
        },
    )


@router.get("/me")
def get_me(current_org: dict = Depends(get_current_org)):
    """Return current org details from the JWT."""
    return {
        "org_id": current_org["sub"],
        "company_name": current_org["company_name"],
        "owner_name": current_org["owner_name"],
        "org_key": current_org.get("org_key", ""),
    }
