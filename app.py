import streamlit as st
from pathlib import Path
import joblib

st.set_page_config(
    page_title="Mushroom Yield Forecast", 
    layout="centered"
)
st.title("Polyhouse Yield Predictor")
st.caption("Agritech environmental forecasting from sensor data")


@st.cache_resource
def load_agritech_artifacts():
    """
    Caches the model and scaler loading so it only happens ONCE.
    This keeps the UI lightning fast on the second interaction.
    """
    model_path = Path("models/champion.joblib")
    scaler_path = Path("models/minmax_scaler.joblib")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


try:
    model, scaler = load_agritech_artifacts()
except Exception as e:
    st.error(f"Could not load model files: {e}")


with st.sidebar:
    st.header("Sensor Readings")
    
    temp = st.slider(
        label="Temperature (°C)", 
        min_value=10.0, 
        max_value=35.0, 
        value=22.0, 
        step=0.1
    )
    
    humid = st.slider(
        label="Humidity (%)", 
        min_value=50.0, 
        max_value=100.0, 
        value=88.0, 
        step=0.5
    )
    
    co2 = st.slider(
        label="CO₂ (ppm)", 
        min_value=400, 
        max_value=2000, 
        value=900, 
        step=10
    )


if st.button("Predict Yield"):
    try:
        
        from src.predict import make_prediction
        kg = make_prediction(temp, humid, co2)
        
        st.metric(
            label="Estimated Daily Yield", 
            value=f"{float(kg):.2f} kg"
        )
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")