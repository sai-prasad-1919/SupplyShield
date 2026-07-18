import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Truck, AlertTriangle, ShieldCheck, Activity, Package, Clock, DollarSign } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import './index.css';

const API_URL = 'http://localhost:8000/api';

function App() {
  const [stats, setStats] = useState(null);
  const [shapData, setShapData] = useState([]);
  
  // Form State
  const [formData, setFormData] = useState({
    package_type: 'electronics',
    vehicle_type: 'truck',
    delivery_mode: 'express',
    region: 'north',
    weather_condition: 'clear',
    distance_km: 150,
    package_weight_kg: 5.5,
    delivery_time_hours: 14,
    expected_time_hours: 12,
    delivery_rating: 4.5,
    delivery_cost: 1500
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchStats();
    fetchShap();
  }, []);

  const fetchStats = async () => {
    try {
      const res = await axios.get(`${API_URL}/stats`);
      setStats(res.data);
    } catch (err) {
      console.error("Failed to fetch stats", err);
    }
  };

  const fetchShap = async () => {
    try {
      const res = await axios.get(`${API_URL}/shap`);
      if (res.data.feature_importance) {
        setShapData(res.data.feature_importance.slice(0, 6)); // Top 6 features
      }
    } catch (err) {
      console.error("Failed to fetch SHAP", err);
    }
  };

  const handleInputChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post(`${API_URL}/predict`, formData);
      setPrediction(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-6 md:p-12">
      {/* Header */}
      <header className="flex items-center justify-between mb-10">
        <div className="flex items-center gap-3">
          <div className="bg-indigo-600 p-3 rounded-xl shadow-lg shadow-indigo-500/20">
            <ShieldCheck size={28} className="text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white">SupplyShield</h1>
            <p className="text-sm text-slate-400">Federated Supply Chain Intelligence</p>
          </div>
        </div>
        
        {stats && (
          <div className="hidden md:flex items-center gap-6 glass-card px-6 py-3">
            <div className="flex items-center gap-2">
              <Activity size={18} className="text-indigo-400" />
              <span className="text-sm font-medium">F1 Score: <span className="text-white">{(stats.metrics.f1_score * 100).toFixed(1)}%</span></span>
            </div>
            <div className="w-px h-6 bg-slate-700"></div>
            <div className="text-sm font-medium">
              <span className="text-slate-400">Nodes: </span>
              <span className="text-white">{stats.processed_organizations} Orgs</span>
            </div>
          </div>
        )}
      </header>

      <main className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Form */}
        <div className="lg:col-span-1 space-y-6">
          <div className="glass-card p-6">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2 mb-6">
              <Package size={20} className="text-indigo-400" />
              Simulate Delivery
            </h2>
            
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="premium-label">Package Type</label>
                  <select name="package_type" value={formData.package_type} onChange={handleInputChange} className="premium-input">
                    <option value="electronics">Electronics</option>
                    <option value="clothing">Clothing</option>
                    <option value="documents">Documents</option>
                    <option value="fragile items">Fragile Items</option>
                    <option value="groceries">Groceries</option>
                    <option value="pharmacy">Pharmacy</option>
                  </select>
                </div>
                <div>
                  <label className="premium-label">Vehicle Type</label>
                  <select name="vehicle_type" value={formData.vehicle_type} onChange={handleInputChange} className="premium-input">
                    <option value="truck">Truck</option>
                    <option value="van">Van</option>
                    <option value="bike">Bike</option>
                    <option value="scooter">Scooter</option>
                    <option value="ev van">EV Van</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="premium-label">Weather</label>
                  <select name="weather_condition" value={formData.weather_condition} onChange={handleInputChange} className="premium-input">
                    <option value="clear">Clear</option>
                    <option value="rainy">Rainy</option>
                    <option value="stormy">Stormy</option>
                    <option value="foggy">Foggy</option>
                    <option value="cold">Cold</option>
                    <option value="hot">Hot</option>
                  </select>
                </div>
                <div>
                  <label className="premium-label">Region</label>
                  <select name="region" value={formData.region} onChange={handleInputChange} className="premium-input">
                    <option value="north">North</option>
                    <option value="south">South</option>
                    <option value="east">East</option>
                    <option value="west">West</option>
                    <option value="central">Central</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="premium-label">Delivery Mode</label>
                  <select name="delivery_mode" value={formData.delivery_mode} onChange={handleInputChange} className="premium-input">
                    <option value="express">Express</option>
                    <option value="same day">Same Day</option>
                    <option value="standard">Standard</option>
                    <option value="two day">Two Day</option>
                  </select>
                </div>
                <div>
                  <label className="premium-label">Distance (km)</label>
                  <input type="number" name="distance_km" value={formData.distance_km} onChange={handleInputChange} className="premium-input" />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="premium-label">Weight (kg)</label>
                  <input type="number" name="package_weight_kg" value={formData.package_weight_kg} onChange={handleInputChange} className="premium-input" />
                </div>
                <div>
                  <label className="premium-label">Rating</label>
                  <input type="number" step="0.1" name="delivery_rating" value={formData.delivery_rating} onChange={handleInputChange} className="premium-input" />
                </div>
                <div>
                  <label className="premium-label">Cost ($)</label>
                  <input type="number" name="delivery_cost" value={formData.delivery_cost} onChange={handleInputChange} className="premium-input" />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="premium-label">Exp. Time (hrs)</label>
                  <input type="number" name="expected_time_hours" value={formData.expected_time_hours} onChange={handleInputChange} className="premium-input" />
                </div>
                <div>
                  <label className="premium-label">Act. Time (hrs)</label>
                  <input type="number" name="delivery_time_hours" value={formData.delivery_time_hours} onChange={handleInputChange} className="premium-input" />
                </div>
              </div>

              <button type="submit" disabled={loading} className="btn-primary mt-6">
                {loading ? 'Analyzing...' : 'Predict Delay Risk'}
              </button>
            </form>
          </div>
        </div>

        {/* Right Column: Results & Insights */}
        <div className="lg:col-span-2 space-y-6">
          {/* Prediction Result */}
          <div className={`glass-card p-8 relative overflow-hidden transition-all duration-500 ${prediction ? (prediction.prediction === 'Delayed' ? 'border-rose-500/50 shadow-rose-500/10' : 'border-emerald-500/50 shadow-emerald-500/10') : ''}`}>
            
            {/* Background Glow */}
            {prediction && (
              <div className={`absolute -top-24 -right-24 w-64 h-64 rounded-full blur-3xl opacity-20 ${prediction.prediction === 'Delayed' ? 'bg-rose-500' : 'bg-emerald-500'}`}></div>
            )}

            <h2 className="text-lg font-semibold text-white mb-8">Prediction Engine</h2>
            
            {!prediction && !loading && (
              <div className="flex flex-col items-center justify-center py-12 text-slate-500">
                <Truck size={48} className="mb-4 opacity-50" />
                <p>Run a simulation to see federated predictions.</p>
              </div>
            )}

            {loading && (
              <div className="flex flex-col items-center justify-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500 mb-4"></div>
                <p className="text-indigo-400">Querying Global Model...</p>
              </div>
            )}

            {error && (
              <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-lg text-rose-400 flex items-start gap-3">
                <AlertTriangle size={20} className="shrink-0 mt-0.5" />
                <p>{error}</p>
              </div>
            )}

            {prediction && !loading && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
                <div>
                  <div className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-2">Status Forecast</div>
                  <div className={`text-5xl font-bold tracking-tight mb-2 ${prediction.prediction === 'Delayed' ? 'text-rose-400' : 'text-emerald-400'}`}>
                    {prediction.prediction}
                  </div>
                  <div className="text-slate-400">
                    Confidence: <span className="text-white font-medium">{((prediction.prediction === 'Delayed' ? prediction.delay_probability : 1 - prediction.delay_probability) * 100).toFixed(1)}%</span>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-slate-400">Risk Probability</span>
                      <span className="text-white">{(prediction.delay_probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-2 w-full bg-slate-700 rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full transition-all duration-1000 ${prediction.prediction === 'Delayed' ? 'bg-rose-500' : 'bg-emerald-500'}`}
                        style={{ width: `${prediction.delay_probability * 100}%` }}
                      ></div>
                    </div>
                  </div>
                  <div className="flex gap-4 pt-2">
                    <div className="bg-slate-900/50 rounded-lg p-3 flex-1 border border-slate-700">
                      <div className="text-xs text-slate-400 mb-1">Risk Level</div>
                      <div className={`font-semibold ${prediction.risk_level === 'High' ? 'text-rose-400' : prediction.risk_level === 'Medium' ? 'text-amber-400' : 'text-emerald-400'}`}>
                        {prediction.risk_level}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Model Insights (SHAP) */}
          <div className="glass-card p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <Activity size={20} className="text-indigo-400" />
                Global Feature Importance (SHAP)
              </h2>
            </div>
            
            <div className="h-64">
              {shapData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={shapData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                    <XAxis type="number" hide />
                    <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} width={120} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '0.5rem', color: '#f8fafc' }}
                      itemStyle={{ color: '#818cf8' }}
                      formatter={(value) => value.toFixed(4)}
                    />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={24}>
                      {shapData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={index === 0 ? '#6366f1' : '#4f46e5'} opacity={1 - index * 0.15} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex items-center justify-center h-full text-slate-500">
                  Loading insights...
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
