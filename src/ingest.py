# src/ingest.py
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/polyhouse_sensors.csv")
INTERIM = Path("data/interim")
INTERIM.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    RAW,
    parse_dates=["timestamp"],
    dtype={
        "temperature_c": "float64",
        "humidity_pct": "float64",
        "co2_ppm": "float64",
        "yield_kg": "float64",
    }
)

print("Dataset Shape:")
print(df.shape)
print("\nData Types:")
print(df.dtypes)
print("\nFirst 5 Rows:")
print(df.head())

df.to_parquet(INTERIM / "01_loaded.parquet", index=False)
print("\nSuccessfully saved interim snapshot to data/interim/01_loaded.parquet")