import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv('data/processed/combined.csv')
X = df.drop(columns=['delayed']).values

scaler = StandardScaler()
scaler.fit(X)

joblib.dump(scaler, 'checkpoints/scaler.pkl')
print("Global scaler saved successfully to checkpoints/scaler.pkl")
