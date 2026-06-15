import os
import json
import pandas as pd
from joblib import dump
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def main():
    print("--- Loading Processed Feature Data ---")
    features_path = os.path.join("data", "processed", "features.parquet")
    
    if not os.path.exists(features_path):
        print(f"Error: Could not find {features_path}.")
        return
        
    df = pd.read_parquet(features_path)
    target_col = "yield_kg" 
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split data chronologically (80% train, 20% validation/test)
    split_idx = int(len(df) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    
    print(f"Training features shape: {X_train.shape}")

    print("\n--- Initializing Grid Search Optimization ---")
    tscv = TimeSeriesSplit(n_splits=3)
    
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 8, 16],
        "min_samples_leaf": [1, 3, 5]
    }
    
    base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True
    )
    
    print("Searching for the best parameters... please wait.")
    search.fit(X_train, y_train)
    
    # Prints results directly to terminal window
    print("Best params:", search.best_params_)
    print("Best CV MAE:", -search.best_score_)
    
    best_model = search.best_estimator_

    # Save configuration to models/ directory
    os.makedirs("models", exist_ok=True)
    with open("models/rf_best_params.json", "w") as f:
        json.dump(search.best_params_, f, indent=2)
    print("Saved 'rf_best_params.json' successfully.")
        
    # Log cv_results_ head to CSV
    cv_results_df = pd.DataFrame(search.cv_results_)
    cv_results_df = cv_results_df.sort_values(by="rank_test_score")
    
    os.makedirs("reports", exist_ok=True)
    results_csv_path = os.path.join("reports", "grid_search_results.csv")
    cv_results_df.head(10).to_csv(results_csv_path, index=False)
    print(f"Saved hyperparameter search log sheet to {results_csv_path}")
    
    # Save the tuned model binary file as champion
    dump(best_model, os.path.join("models", "champion.joblib"))
    print("Saved 'champion.joblib' to models/ folder.")

if __name__ == "__main__":
    main()