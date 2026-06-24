import streamlit as st
import numpy as np
import pandas as pd
import csv
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("logs/predictions.csv")

# Prediction logging function
def log_prediction(temp, humid, co2, predicted_kg):
    LOG_PATH.parent.mkdir(exist_ok=True)

    write_header = not LOG_PATH.exists()

    with LOG_PATH.open("a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "timestamp_utc",
                "temp_c",
                "humidity_pct",
                "co2_ppm",
                "predicted_kg"
            ])

        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            temp,
            humid,
            co2,
            round(predicted_kg, 3)
        ])

st.set_page_config(
    page_title="Mushroom Yield Forecast",
    layout="centered"
)

st.title("Polyhouse Yield Predictor")
st.caption(
    "Zelbytes Agritech branding environmental forecasting from sensor data"
)

# Streamlit user-friendly error block
try:
    from src.predict import make_prediction as predict_yield
except FileNotFoundError:
    st.error(
        "Model artifacts missing. Run the training pipeline in `src/` first."
    )
    st.stop()
except ImportError:
    st.error(
        "Model artifacts missing. Run the training pipeline in `src/` first."
    )
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

# Out-of-range warning alerts
if temp > 32.0 or temp < 12.0 or humid < 55.0:
    st.warning(
        "⚠️ Warning: Selected settings fall outside standard safe polyhouse growth limits!"
    )

# Main prediction section
if st.button("Predict Yield"):
    with st.spinner("Calculating environmental prediction yields..."):
        try:
            kg = predict_yield(temp, humid, co2)

            # Save prediction to CSV
            log_prediction(temp, humid, co2, kg)

            st.success("Prediction generated successfully!")

            st.metric(
                label="Estimated Daily Yield",
                value=f"{kg:.2f} kg"
            )

        except Exception as e:
            st.error(
                f"Could not calculate prediction. Details: {e}"
            )

# Sensitivity chart section
st.subheader("What-if: humidity sweep")

temp_fixed = temp
co2_fixed = co2

humid_range = np.linspace(70, 98, 29)

try:
    preds = [
        predict_yield(temp_fixed, h, co2_fixed)
        for h in humid_range
    ]

    chart_df = pd.DataFrame({
        "Humidity (%)": humid_range,
        "Predicted yield (kg)": preds
    })

    st.line_chart(
        chart_df,
        x="Humidity (%)",
        y="Predicted yield (kg)"
    )

except Exception:
    st.info(
        "Adjust input sliders above to generate real-time sensitivity tables."
    )

# Metadata panel
with st.expander("Model information"):
    st.markdown("""
    - **Model:** Tuned Random Forest
    - **Test MAE:** 0.4434 kg/day
    - **Training data:** Polyhouse sensors Jan–Dec 2024
    """)