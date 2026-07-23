import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

// ---------- Auth helpers ----------
const getToken = (): string | null => localStorage.getItem('supplyshield_jwt');

const authHeader = () => {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
};

// ---------- Auth endpoints ----------
export const loginOrg = async (orgId: string, password: string) => {
  const res = await axios.post(`${API_URL}/auth/login`, {
    org_id: orgId.trim().toUpperCase(),
    password,
  });
  return res.data; // { access_token, org_id, company_name, owner_name }
};

export const getMe = async () => {
  const res = await axios.get(`${API_URL}/auth/me`, { headers: authHeader() });
  return res.data;
};

// ---------- Org/analytics endpoints (all JWT-protected) ----------
export const fetchOrgs = async () => {
  const res = await axios.get(`${API_URL}/orgs`);
  return res.data;
};

export const fetchShipments = async () => {
  const res = await axios.get(`${API_URL}/shipments`, { headers: authHeader() });
  return res.data;
};

export const fetchShipment = async (id: string) => {
  const res = await axios.get(`${API_URL}/shipment/${id}`, { headers: authHeader() });
  return res.data;
};

export const fetchKpis = async () => {
  const res = await axios.get(`${API_URL}/kpis`, { headers: authHeader() });
  return res.data;
};

// ---------- Session helpers ----------
export const saveSession = (token: string, orgId: string, orgName: string, ownerName: string) => {
  localStorage.setItem('supplyshield_jwt', token);
  localStorage.setItem('supplyshield_org_id', orgId);
  localStorage.setItem('supplyshield_org_name', orgName);
  localStorage.setItem('supplyshield_owner_name', ownerName);
};

export const clearSession = () => {
  localStorage.removeItem('supplyshield_jwt');
  localStorage.removeItem('supplyshield_org_id');
  localStorage.removeItem('supplyshield_org_name');
  localStorage.removeItem('supplyshield_owner_name');
};

export const isAuthenticated = (): boolean => !!getToken();
