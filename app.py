import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Mushroom Yield Forecast", 
    layout="centered"
)

st.title("Polyhouse Yield Predictor")
st.caption("Zelbytes Agritech branding environmental forecasting from sensor data")

# Streamlit user-friendly error block using your required format
try:
    from src.predict import make_prediction as predict_yield
except FileNotFoundError:
    st.error("Model artifacts missing. Run the training pipeline in `src/` first.")
    st.stop()
except ImportError:
    st.error("Model artifacts missing. Run the training pipeline in `src/` first.")
    st.stop()

# Sidebar for interactive input controls
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

# Out-of-range warning alerts to users
if temp > 32.0 or temp < 12.0 or humid < 55.0:
    st.warning("⚠️ Warning: Selected settings fall outside standard safe polyhouse growth limits!")

# Main view prediction button section with a spinner
if st.button("Predict Yield"):
    with st.spinner("Calculating environmental prediction yields..."):
        try:
            kg = predict_yield(temp, humid, co2)
            st.success("Prediction generated successfully!")
            st.metric(
                label="Estimated Daily Yield", 
                value=f"{kg:.2f} kg"
            )
        except Exception as e:
            st.error(f"Could not calculate prediction. Details: {e}")

# Sensitivity chart section using your exact required snippet structure
st.subheader("What-if: humidity sweep")
temp_fixed, co2_fixed = temp, co2  # Uses your current slider choices as the stable baseline metrics
humid_range = np.linspace(70, 98, 29)

try:
    preds = [predict_yield(temp_fixed, h, co2_fixed) for h in humid_range]
    chart_df = pd.DataFrame({"Humidity (%)": humid_range, "Predicted yield (kg)": preds})
    st.line_chart(chart_df, x="Humidity (%)", y="Predicted yield (kg)")
except Exception as e:
    st.info("Adjust input sliders above to generate real-time sensitivity tables.")

# Metadata panel expander using your exact required format and real data values
with st.expander("Model information"):
    st.markdown(f"""
    - **Model:** Tuned Random Forest
    - **Test MAE:** 0.4434 kg/day
    - **Training data:** Polyhouse sensors Jan–Dec 2024
    """)