import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

def main():
    print("--- Running Time-Series Cross-Validation ---")
    
    # Load processed data
    features_path = os.path.join("data", "processed", "features.parquet")
    if not os.path.exists(features_path):
        print(f"Error: Dataset not found at {features_path}")
        return
        
    df = pd.read_parquet(features_path)
    
    target_col = 'yield_kg'
    X = df[["temperature_c_scaled", "humidity_pct_scaled", "co2_ppm_scaled"]]
    y = df[target_col]
    
    # Use the same training subset split for cross-validation consistency
    split_idx = int(len(df) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    
    # Cross-Validation configuration
    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    lin_reg = LinearRegression()
    
    # Compute Cross-Validation error (turning negative scores back into positive values)
    rf_scores = -cross_val_score(rf, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")
    lin_scores = -cross_val_score(lin_reg, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")
    
    print(f"\n[RESULTS] Random Forest CV MAE: {rf_scores.mean():.2f} +/- {rf_scores.std():.2f}")
    print(f"[RESULTS] Linear Regression CV MAE: {lin_scores.mean():.2f} +/- {lin_scores.std():.2f}")

if __name__ == "__main__":
    main()