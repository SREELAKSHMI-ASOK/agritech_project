import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# 1. Load the data and sort it by time
print("Loading cleaned dataset...")
df = pd.read_parquet("data/interim/02_cleaned.parquet")
df = df.sort_values("timestamp")

# 2. Add our Day 7 feature column
df["temp_humid_interaction"] = (df["temperature_c"] * df["humidity_pct"]) / 100
feature_cols = ["temperature_c", "humidity_pct", "co2_ppm", "temp_humid_interaction"]

# 3. Chronological split (80% Train, 20% Test)
print("Splitting data into 80% Train and 20% Test by time...")
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

# Split features and targets
X_train = train_df[feature_cols]
y_train = train_df["yield_kg"].values

X_test = test_df[feature_cols]
y_test = test_df["yield_kg"].values

# 4. Safe Scaling (Fit ONLY on Train data)
print("Scaling safely (Fitting scaler on Train data only)...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fits and transforms train
X_test_scaled = scaler.transform(X_test)       # ONLY transforms test using train rules

# 5. Save the safe scaler object
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/minmax_scaler_train.joblib")

# 6. Print out metrics you need for your project submission dashboard
print("\n" + "="*50)
print("METRICS FOR YOUR PROJECT REPORT / README:")
print("="*50)
print(f"Total Rows: {len(df)}")
print(f"Train Dataset Size: {len(train_df)} rows")
print(f"Test Dataset Size:  {len(test_df)} rows")
print(f"Train Date Range:   {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
print(f"Test Date Range:    {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
print("="*50 + "\n")