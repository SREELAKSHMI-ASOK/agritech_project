# Agritech Project - Feature Engineering & Temporal Split

## Feature Engineering Definitions
* **temp_humid_interaction**: `(temperature_c * humidity_pct) / 100`
  * *Note*: Captures the combined behavior of atmospheric warmth and moisture levels on crop outputs.

---

## Temporal Train/Test Split Summary
* **Split Type**: Chronological (80% Train / 20% Test)

### Dataset Sizes & Ranges
* **Total Rows**: 365
* **Train Dataset Size**: 292 rows
* **Test Dataset Size**: 73 rows
* **Train Date Range**: 2024-01-01 to 2024-10-18
* **Test Date Range**: 2024-10-19 to 2024-12-30