import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# 1. Load the data and sort it by time
print("Loading cleaned dataset...")
df = pd.read_parquet("data/interim/02_cleaned.parquet")
df = df.sort_values("timestamp")

# 2. Create an engineered feature (Interaction term)
print("Creating 'temp_humid_interaction' feature...")
df["temp_humid_interaction"] = (df["temperature_c"] * df["humidity_pct"]) / 100

# Define our feature matrix (X) columns and target variable (y)
feature_cols = ["temperature_c", "humidity_pct", "co2_ppm", "temp_humid_interaction"]
X = df[feature_cols]
y = df["yield_kg"].values

# 3. Apply MinMaxScaler to features
print("Scaling features to fall between 0 and 1...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. Make sure directories exist to save our work
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# 5. Save the scaler tool so we can reuse it later
print("Saving scaler to 'models/minmax_scaler.joblib'...")
joblib.dump(scaler, "models/minmax_scaler.joblib")

# 6. Put scaled numbers back into a structured table and save it
processed = pd.DataFrame(X_scaled, columns=[c + "_scaled" for c in feature_cols])
processed["yield_kg"] = y
processed.to_parquet("data/processed/features.parquet", index=False)

print("Feature Engineering & Min-Max Scaling complete! File saved to 'data/processed/features.parquet'")
print("-" * 40)