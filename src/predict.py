import json
import joblib
import pandas as pd
from pathlib import Path

# Set up the paths relative to your project structure
MODEL_DIR = Path("models")

# Load your training artifacts 
_scaler = joblib.load(MODEL_DIR / "minmax_scaler.joblib")
_model = joblib.load(MODEL_DIR / "random_forest.joblib")

def make_prediction(temperature_c: float, humidity_pct: float, co2_ppm: float) -> float:
    """
    Accepts raw sensor values, calculates the interaction feature for the scaler,
    and passes the correct 3 scaled features to the Random Forest model.
    """
    # 1. Calculate the interaction term
    temp_humid_interaction = (temperature_c * humidity_pct) / 100
    
    # 2. Create a DataFrame for the scaler
    input_data = pd.DataFrame([{
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "co2_ppm": co2_ppm,
        "temp_humid_interaction": temp_humid_interaction
    }])
    
    # 3. Scale all 4 features using your loaded scaler
    scaled_features = _scaler.transform(input_data)
    
    # 4. Wrap the first 3 scaled columns into a DataFrame with names for the RandomForest
    # This matches the column names the model was trained on and removes UserWarnings
    scaled_data_for_model = pd.DataFrame(
        scaled_features[:, :3], 
        columns=["temperature_c_scaled", "humidity_pct_scaled", "co2_ppm_scaled"]
    )
    
    # 5. Generate the prediction and extract the float value
    prediction = float(_model.predict(scaled_data_for_model)[0])
    return prediction

if __name__ == "__main__":
    # Test the completed pipeline function with sample readings
    print("--- Running Pipeline Test ---")
    try:
        sample_yield = make_prediction(temperature_c=22.0, humidity_pct=88.0, co2_ppm=920.0)
        print("Pipeline Test Success!")
        print(f"Predicted Yield: {sample_yield:.2f} kg")
    except Exception as e:
        print(f"An error occurred during testing: {e}")