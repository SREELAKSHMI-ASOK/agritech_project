# src/clean.py
import pandas as pd
from pathlib import Path

df = pd.read_parquet("data/interim/01_loaded.parquet")

# Missing report
print("--- Missing Values Report ---")
print(df.isna().sum())

# Valid ranges for oyster polyhouse
valid = (
    df["humidity_pct"].between(50, 100)
    & df["temperature_c"].between(10, 35)
    & df["co2_ppm"].between(400, 2000)
    & df["yield_kg"].notna()
)
df = df[valid].copy()

# Short gap: forward-fill sensor columns only
cols = ["temperature_c", "humidity_pct", "co2_ppm"]
df[cols] = df[cols].ffill(limit=2)

# Drop remaining rows with null target
df = df.dropna(subset=["yield_kg"])

# Duplicates by timestamp
df = df.drop_duplicates(subset=["timestamp"], keep="last")

# Save the final cleaned data
df.to_parquet("data/interim/02_cleaned.parquet", index=False)
print(f"\nClean rows: {len(df)}")

# Extra helper line: let's save a 50+ row sample for your assignment upload
df.head(55).to_csv("data/processed/cleaned_sample_50_rows.csv", index=False)
print("Successfully generated verification sample at data/processed/")