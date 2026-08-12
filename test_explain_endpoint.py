import json
from fastapi.testclient import TestClient
from backend.main import app

def test_explain():
    client = TestClient(app)
    
    # We must trigger startup event manually using TestClient by using it as a context manager
    with client:
        req_data = {
            "features": {
                "package_type": "Standard",
                "vehicle_type": "Truck",
                "delivery_mode": "Ground",
                "region": "North",
                "weather_condition": "Rainy",
                "distance_km": 847,
                "package_weight_kg": 12.3,
                "expected_time_hours": 48.0,
                "delivery_cost": 2340,
            }
        }
        
        # We also need to mock the current_org dependency. 
        # But for a quick test, maybe we can override the dependency.
        from backend.main import get_current_org
        app.dependency_overrides[get_current_org] = lambda: {"org_key": "novamart", "postgres_db": "supplyshield_novamart"}
        
        response = client.post("/api/predict/explain", json=req_data)
        
        if response.status_code == 200:
            print("SUCCESS:", json.dumps(response.json(), indent=2))
        else:
            print("ERROR:", response.status_code, response.text)

if __name__ == "__main__":
    test_explain()
