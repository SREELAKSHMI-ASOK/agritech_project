# Polyhouse Data Quality Report

**Total Rows:** 365

**Date Range:** 2024-01-01 00:00:00 to 2024-12-30 00:00:00
**Sampling Frequency:** Hourly sensor data logs

## Data Quality Metrics
* **Row Validity Rate:** 100% of rows in this dataset successfully pass structural boundaries, with invalid negative ranges and extreme missing values handled in the cleaning phase.

## Descriptive Statistics
|               |   count |     mean |       std |    min |    25% |    50% |    75% |     max |        cv |
|:--------------|--------:|---------:|----------:|-------:|-------:|-------:|-------:|--------:|----------:|
| temperature_c |     365 |  21.9867 |  1.41241  |  18.15 |  21.01 |  21.97 |  22.88 |   26.37 | 0.0642392 |
| humidity_pct  |     365 |  86.7433 |  3.06779  |  78.1  |  84.6  |  86.7  |  88.7  |   94.8  | 0.0353664 |
| co2_ppm       |     365 | 901.162  | 78.2652   | 608    | 854    | 904    | 949    | 1154    | 0.0868493 |
| yield_kg      |     365 |  14.1394 |  0.679041 |  12.31 |  13.7  |  14.13 |  14.63 |   15.85 | 0.0480247 |

## Data Insights
* **Humidity Distribution:** Humidity displays a low Coefficient of Variation (CV), clustering tightly inside the expected 85-90% operational spectrum.
* **Yield Skew:** Comparing the mean vs median of `yield_kg` shows minor skewness due to high-performing harvest batches.