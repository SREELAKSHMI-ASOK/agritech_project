import json
import joblib
import pandas as pd
from pathlib import Path

# Set up the paths relative to your project root structure
MODEL_DIR = Path("models")

# Load your training artifacts 
_scaler = joblib.load(MODEL_DIR / "minmax_scaler.joblib")
_model = joblib.load(MODEL_DIR / "champion.joblib")

def make_prediction(temperature_c: float, humidity_pct: float, co2_ppm: float) -> float:
    """
    Accepts raw sensor values, calculates the interaction feature for the scaler,
    and passes all 4 scaled features to the champion Random Forest model.
    """
    # 1. Calculate the interaction term
    temp_humid_interaction = (temperature_c * humidity_pct) / 100
    
    # 2. Create a DataFrame for the scaler matching the 4 expected training features
    input_data = pd.DataFrame([{
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "co2_ppm": co2_ppm,
        "temp_humid_interaction": temp_humid_interaction
    }])
    
    # 3. Scale all 4 features
    scaled_features = _scaler.transform(input_data)
    
    # 4. Put all 4 scaled columns into a DataFrame with names matching the training stage
    scaled_data_for_model = pd.DataFrame(
        scaled_features, 
        columns=[
            "temperature_c_scaled", 
            "humidity_pct_scaled", 
            "co2_ppm_scaled", 
            "temp_humid_interaction_scaled"
        ]
    )
    
    # 5. Generate the prediction
    prediction = float(_model.predict(scaled_data_for_model)[0])
    return prediction

if __name__ == "__main__":
    print("--- Running Pipeline Test ---")
    try:
        sample_yield = make_prediction(temperature_c=22.0, humidity_pct=88.0, co2_ppm=900.0)
        print("Pipeline Test Success!")
        print(f"Predicted Yield: {sample_yield:.2f} kg")
    except Exception as e:
        print(f"An error occurred during testing: {e}")