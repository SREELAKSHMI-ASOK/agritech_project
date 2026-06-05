import pandas as pd
from pathlib import Path

# Load the cleaned data from your interim folder
df = pd.read_parquet("data/interim/02_cleaned.parquet")

# Calculate summary statistics and Coefficient of Variation
features = ["temperature_c", "humidity_pct", "co2_ppm", "yield_kg"]
summary = df[features].describe().T
summary["cv"] = summary["std"] / summary["mean"]

# Build the report markdown text
report = []
report.append("# Polyhouse Data Quality Report\n")
report.append(f"**Total Rows:** {len(df):,}\n")

min_date = df['timestamp'].min()
max_date = df['timestamp'].max()
report.append(f"**Date Range:** {min_date} to {max_date}\n")

report.append("## Descriptive Statistics")
report.append(summary.to_markdown())

report.append("\n## Data Insights")
report.append("* **Humidity Distribution:** Review if humidity clusters tightly around the expected 85-90% range.")
report.append("* **Yield Skew:** Compare the mean vs median of yield_kg to check for data skewness.")

# Save the report to your reports directory
Path("reports").mkdir(exist_ok=True)
Path("reports/data_quality.md").write_text("\n".join(report), encoding="utf-8")

print("Quality report generated successfully!")