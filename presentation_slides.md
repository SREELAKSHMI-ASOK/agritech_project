# Mushroom Yield Forecast
## Deployed Agritech ML Pipeline
**Presenter:** SREELAKSHMI ASOK  
**Project Role:** AI & Data Science Intern, Zelbytes

---

## 1. Problem & Agritech Context

* **Objective:** Predict daily oyster mushroom yield (kg) using controlled polyhouse sensor metrics.
* **Why it matters:** * **Underestimating Yield:** Causes worker shortages and distribution shipping bottlenecks.
  * **Overestimating Yield:** Risks failing buyer contracts, breaking trust, and damaging supply reliability.
* **Goal:** Create a real-time web application to give greenhouse managers accurate planning foresight.

---

## 2. Data Pipeline & Feature Engineering

* **Dataset Overview:** 365 days of continuous polyhouse sensor data records.
* **Monitored Features:** Temperature (°C), Relative Humidity (%), and Carbon Dioxide levels (ppm).
* **Feature Engineering (`temp_humid_interaction`):** $$\text{Interaction Score} = \frac{\text{temperature\_c} \times \text{humidity\_pct}}{100}$$
  Captures the combined stress/growth impact of ambient warmth and moisture on fungal development.

---

## 3. Strict Validation Strategy

* **Leakage Prevention:** Avoided standard random splits which leak future time trends into past predictions.
* **Chronological Train/Test Split (80% / 20%):**
  * **Total Size:** 365 rows
  * **Training Set:** 292 rows (2024-01-01 to 2024-10-18)
  * **Testing Set:** 73 rows (2024-10-19 to 2024-12-30)
* **Pre-processing:** Implemented input scaling using `minmax_scaler.joblib` fitted strictly on the training partition.

---

## 4. Modeling & Performance Metrics

| Model | Test MAE | RMSE | R² | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | 0.4170 | 0.5345 | 0.429 | Baseline |
| **Default Random Forest** | 0.4386 | 0.5521 | 0.390 | Alternate |
| **Tuned Random Forest** | 0.4434 | 0.5528 | 0.389 | **Champion** |

* **Selection Rationale:** While error indices are close, the Tuned Random Forest captures complex non-linear interactions better and balances out the operational risks of extreme prediction misses.

---

## 5. Live Application Demo

* **Live Tool URL:** [zelbytes-mushroom-yield-predict.streamlit.app](https://zelbytes-mushroom-yield-predict.streamlit.app/)
* **Core Application Capabilities:**
  * **Interactive Feature Sliders:** Quick inputs for real-time inference generation.
  * **What-if Humidity Sweep Chart:** Simulates yield trajectory across 70% to 98% relative moisture environments.
  * **Trust Metrics Panel:** Discloses model baseline margins and evaluation errors openly to operators.
  * **Defensive Error Handling:** Built-in sensor bounds check to prevent system crashes.

---

## 6. Live App Interface Backup

If network conditions drop during live validation, this UI blueprint shows the runtime dashboard environment:

![Streamlit Production UI Blueprint](dashboard_screenshot.png)

---

## 7. MLOps Monitoring & Production Safety

* **Lightweight Logging:** Every prediction cycle saves data instantly to a system text tracking log:
  `[Timestamp] Temp: 22.0, Humidity: 88.0, CO2: 900 -> Prediction: 14.14kg`
* **Automated Rebuilding Triggers:**
  1. Mean Absolute Error drops past 2.0 kg/day.
  2. Accumulation of 30 days of un-modeled farm logs.
  3. Total error variance spikes by 15% or more.

---

## 8. Lessons Learned & Next Steps

* **Top 3 Core Competencies Built:**
  1. End-to-end deployment engineering using Streamlit and Cloud hosting services.
  2. Robust temporal validation to defend against data leakage in time-ordered settings.
  3. Production error logging and drift monitoring strategy design.
* **Areas for Future Growth:**
  1. Adding multi-step automated model retraining workflows.
  2. Integrating additional feature streams like light intensity sensors.

---
## Thank you! Q&A Session