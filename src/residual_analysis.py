import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

def main():
    data_path = "data/processed/features.parquet"
    model_path = "models/linear_regression.joblib"
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("Error: Missing features data or trained model file. Run Day 9 first!")
        return
        
    # 1. Load data and the trained model
    df = pd.read_parquet(data_path)
    model = joblib.load(model_path)
    
    # Updated to match your exact columns
    feature_cols = ['temperature_c_scaled', 'humidity_pct_scaled', 'co2_ppm_scaled', 'temp_humid_interaction_scaled']
    target_col = "yield_kg"
    
    # 2. Extract the test set (Must match Day 9 split exactly)
    train_size = int(len(df) * 0.8)
    X_test = df[feature_cols].iloc[train_size:]
    y_test = df[target_col].iloc[train_size:]

    # 3. Generate predictions and calculate residuals
    pred_test = model.predict(X_test)
    residuals = y_test - pred_test

    # 4. Create diagnostic plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Residuals vs Predicted
    axes[0].scatter(pred_test, residuals, alpha=0.6, color="#1f77b4")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted Yield (kg)")
    axes[0].set_ylabel("Residual (kg)")
    axes[0].set_title("Residuals vs. Predicted Yield")

    # Plot 2: Residuals vs Humidity (Using your exact scaled column name)
    axes[1].scatter(X_test["humidity_pct_scaled"], residuals, alpha=0.6, color="#2ca02c")
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Scaled Humidity")
    axes[1].set_ylabel("Residual (kg)")
    axes[1].set_title("Residuals vs. Humidity")

    plt.tight_layout()
    
    # Save the output figure
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig("reports/figures/residuals_linear.png", dpi=150)
    print("-> Residual diagnostic plots saved to reports/figures/residuals_linear.png")

if __name__ == "__main__":
    main()