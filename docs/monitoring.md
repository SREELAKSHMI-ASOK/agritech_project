# App Monitoring Plan

## Sample Log Format
We log every user input and prediction in a lightweight text file:

```text
[Timestamp] Temp: 22.0, Humidity: 88.0, CO2: 900 -> Prediction: 14.14kg
```

## Model Artifact Handling
The trained model weights (`champion.joblib`) and preprocessing configurations (`minmax_scaler.joblib`) are tracked and stored inside the `models/` directory within the version-controlled repository for direct access during deployment.

## Model Retraining Triggers
We will retrain the machine learning model if:

1. The prediction error (MAE) goes above 2.0 kg/day from our base performance of 0.4434 kg/day.
2. 30 days of new greenhouse data have been collected.
3. Test MAE increases by 15% compared to the current deployed model.

## Monitoring Plan

### What to Watch
- Prediction values compared to the historical yield range.
- Sensor input anomalies (e.g., sudden CO₂ spikes, unusual temperature or humidity values).
- Frequency of predictions that fall outside expected operating conditions.

### Drift Triggers
- Retrain the model if test MAE increases by 15%.
- Alert administrators if a prediction exceeds the maximum historical yield observed in the training data.
- Investigate data drift when sensor distributions differ significantly from historical patterns.

## Iteration Roadmap
1. Add a weekly automated retraining pipeline.
2. Add light intensity as a new sensor feature to improve prediction accuracy.
3. Build a Streamlit admin panel to visualize prediction logs and monitoring history.
4. Enhance monitoring with automated drift detection and alert notifications.
```