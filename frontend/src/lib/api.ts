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

// ---------- Supplier Intelligence endpoints ----------
export const fetchSupplierKpis = async () => {
  const res = await axios.get(`${API_URL}/suppliers/kpis`, { headers: authHeader() });
  return res.data;
};

export const fetchSuppliers = async () => {
  const res = await axios.get(`${API_URL}/suppliers`, { headers: authHeader() });
  return res.data; // { suppliers: [...], total: N }
};

export const fetchSupplier = async (id: number | string) => {
  const res = await axios.get(`${API_URL}/supplier/${id}`, { headers: authHeader() });
  return res.data;
};

export const fetchSupplierShipments = async (id: number | string) => {
  const res = await axios.get(`${API_URL}/supplier/${id}/shipments`, { headers: authHeader() });
  return res.data; // { shipments: [...], total: N }
};

// ---------- Demand Forecast ----------
export const fetchDemandForecast = async (weeks = 4) => {
  const res = await axios.get(`${API_URL}/demand/forecast`, {
    params: { weeks },
    headers: authHeader(),
  });
  return res.data; // { org, weeks_ahead, history, forecast, forecast_start_date, model_info }
};

// ---------- Shipment Explanation (AI + SHAP + LLM) ----------
export const explainShipment = async (shipmentId: string) => {
  const res = await axios.post(
    `${API_URL}/predict/explain`,
    { shipment_id: shipmentId },
    { headers: authHeader() }
  );
  return res.data; // { shipment_id, prediction, consensus, shap, recommendations, guidance_source }
};

// ---------- Human-in-the-Loop Feedback ----------
export interface FeedbackPayload {
  shipment_id: string;
  fl_probability: number;
  xgb_probability: number;
  risk_level: string;
  recommendations: any[];
  guidance_source: string;
  decision: 'confirm' | 'override' | 'escalate';
  override_reason?: string;
  alternative_action?: string;
}

export const submitFeedback = async (payload: FeedbackPayload) => {
  const res = await axios.post(`${API_URL}/guidance/feedback`, payload, {
    headers: authHeader(),
  });
  return res.data; // { status: "logged", analysis_id: "..." }
};

