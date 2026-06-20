# App Monitoring Plan

## Sample Log Format
We log every user input and prediction in a lightweight text file:
`[Timestamp] Temp: 22.0, Humidity: 88.0, CO2: 920 -> Prediction: 13.83kg`

## Model Artifact Handling
The trained model weights (`champion.joblib`) and preprocessing configurations (`minmax_scaler.joblib`) are tracked and stored inside the `models/` directory within the version-controlled repository for direct access during deployment.

## Model Retraining Triggers
We will retrain the machine learning model if:
1. The prediction error (MAE) goes above 2.0 kg/day from our base performance of 0.4434 kg/day.
2. 30 days of new greenhouse data have been collected.