import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    print("--- Running Random Forest Model Training ---")
    
    # Load processed data
    features_path = os.path.join("data", "processed", "features.parquet")
    if not os.path.exists(features_path):
        print(f"Error: Dataset not found at {features_path}")
        return
        
    df = pd.read_parquet(features_path)
    
    # Target and features setup
    X = df[["temperature_c_scaled", "humidity_pct_scaled", "co2_ppm_scaled"]]
    y = df['yield_kg']
    
    # Chronological 80/20 train/test split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train the model
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Evaluate
    pred = rf.predict(X_test)
    print(f"\n[RESULTS] Random Forest Test MAE: {mean_absolute_error(y_test, pred):.2f} kg")
    print(f"[RESULTS] Random Forest Test R²: {r2_score(y_test, pred):.3f}")
    
    # Save feature importance chart
    importances = rf.feature_importances_
    labels = ["Temperature (Scaled)", "Humidity (Scaled)", "CO2 (Scaled)"]
    
    plt.figure(figsize=(8, 5))
    plt.barh(labels, importances, color='skyblue')
    plt.xlabel("Importance Score")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    
    fig_path = os.path.join("reports", "figures", "rf_importance.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved chart to: {fig_path}")
    
    # Save the model file
    model_path = os.path.join("models", "random_forest.joblib")
    joblib.dump(rf, model_path)
    print(f"Saved model to: {model_path}")

if __name__ == "__main__":
    main()