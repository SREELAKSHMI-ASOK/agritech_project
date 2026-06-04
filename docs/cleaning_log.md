# Agritech Dataset Cleaning Log

## Column Meanings
* `timestamp`: The date and time the sensors logged the environmental data.
* `temperature`: Ambient air temperature inside the polyhouse. Essential for mushroom cultivation.
* `humidity`: Relative humidity levels. Must be under 100%.
* `CO2`: Carbon dioxide concentrations. Affects mushroom growth.
* `harvested_yield`: Total crop weight gathered. Our target variable.

## Cleaning Strategy
1. Exact duplicates were removed.
2. Invalid sensor readings (Humidity > 100%) were filtered out.
3. Empty rows missing key columns were dropped to keep data clean.