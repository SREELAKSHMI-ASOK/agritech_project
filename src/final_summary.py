from pathlib import Path

# Create the final combined summary document required by the main task page
eda_summary = """# Task 3: EDA Summary & Visualizations

## Methodology
Exploratory Data Analysis was conducted on the polyhouse sensor datasets to evaluate environmental conditions vs mushroom crop yields.

## Key Visual Observations
1. **Correlation Structure:** The heatmap reveals linear strengths. CO2 and Humidity show critical patterns with output weight.
2. **Environmental Thresholds:** Scatter plots indicate that yield peaks sharply under controlled climate windows.

## File Verification
* Both background metrics and system plots have been compiled into the repository files.
"""

Path("reports/eda_summary.md").write_text(eda_summary, encoding="utf-8")
print("Final summary page generated successfully!")