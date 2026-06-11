import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def main():
    data_path = "data/processed/features.parquet"
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}.")
        return
        
    # 1. Load data
    df = pd.read_parquet(data_path)
    
    # Updated to match your exact columns
    feature_cols = ['temperature_c_scaled', 'humidity_pct_scaled', 'co2_ppm_scaled', 'temp_humid_interaction_scaled']
    target_col = "yield_kg"
    
    # 2. Split data (80% train, 20% test)
    train_size = int(len(df) * 0.8)
    X_train = df[feature_cols].iloc[:train_size]
    y_train = df[target_col].iloc[:train_size]
    X_test = df[feature_cols].iloc[train_size:]
    y_test = df[target_col].iloc[train_size:]

    # 3. Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Predict and evaluate
    pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    r2 = r2_score(y_test, pred_test)

    print(f"Test MAE: {mae:.2f} kg")
    print(f"Test RMSE: {rmse:.2f} kg")
    print(f"Test R²: {r2:.3f}")

    # 5. Save performance metrics to JSON
    metrics_data = {
        "test_mae": round(mae, 2),
        "test_rmse": round(rmse, 2),
        "test_r2": round(r2, 3)
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/linear_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)
    print("-> Metrics saved to reports/linear_metrics.json")

    # 6. Print coefficients for interpretation
    print("\nModel Coefficients:")
    for name, coef in zip(feature_cols, model.coef_):
        print(f"   {name}: {coef:.3f}")

    # 7. Save model artifact
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/linear_regression.joblib")
    print("-> Model saved to models/linear_regression.joblib")

if __name__ == "__main__":
    main()