"""
SupplyShield — Mock Data Generator

Generates a synthetic version of the India Multi-Partner dataset for development.
Replace this with the real Kaggle dataset when available.
"""
import pandas as pd
import numpy as np
import random
from pathlib import Path

def generate_mock_data(num_records=25000, output_path="data/raw/delivery_logistics.csv"):
    np.random.seed(42)
    random.seed(42)
    
    partners = ["Partner 1", "Partner 2", "Partner 3"]
    weather = ["Clear", "Rain", "Fog", "Storm"]
    traffic = ["Low", "Medium", "High", "Very High"]
    vehicle = ["Motorcycle", "Van", "Truck"]
    
    data = {
        "delivery_id": [f"DEL_{i:06d}" for i in range(num_records)],
        "delivery_partner": np.random.choice(partners, num_records, p=[0.4, 0.35, 0.25]),
        "distance_km": np.random.lognormal(mean=2.0, sigma=0.8, size=num_records).round(1),
        "weather": np.random.choice(weather, num_records, p=[0.6, 0.2, 0.15, 0.05]),
        "traffic": np.random.choice(traffic, num_records, p=[0.3, 0.4, 0.2, 0.1]),
        "vehicle_type": np.random.choice(vehicle, num_records, p=[0.6, 0.3, 0.1]),
    }
    
    df = pd.DataFrame(data)
    
    # Cap distance
    df["distance_km"] = df["distance_km"].clip(1.0, 150.0)
    
    # Generate target 'is_late' based on features
    # Base probability
    prob_late = 0.1
    
    # Distance effect
    prob_late += (df["distance_km"] / 150.0) * 0.2
    
    # Weather effect
    weather_risk = {"Clear": 0.0, "Rain": 0.15, "Fog": 0.2, "Storm": 0.4}
    prob_late += df["weather"].map(weather_risk)
    
    # Traffic effect
    traffic_risk = {"Low": 0.0, "Medium": 0.05, "High": 0.2, "Very High": 0.35}
    prob_late += df["traffic"].map(traffic_risk)
    
    # Partner effect
    partner_risk = {"Partner 1": 0.0, "Partner 2": -0.05, "Partner 3": 0.1}
    prob_late += df["delivery_partner"].map(partner_risk)
    
    # Ensure prob is between 0 and 1
    prob_late = prob_late.clip(0.0, 1.0)
    
    # Generate binary outcome
    df["is_late"] = np.random.binomial(1, prob_late)
    
    # Save to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"Generated {num_records} mock records and saved to {output_path}")
    print(df.head())
    print(f"Delay rate: {df['is_late'].mean():.2%}")

if __name__ == "__main__":
    generate_mock_data()
