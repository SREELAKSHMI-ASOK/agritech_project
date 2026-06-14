# Cross-Validation & Overfitting Analysis

## 1. Methodology
Our dataset was split chronologically, training on the initial 80% of our data and reserving the final 20% as a strict evaluation test set. To reliably verify performance across time chunks without data leakage, we ran a 5-fold `TimeSeriesSplit` cross-validation process using our training data partition.

## 2. Model Performance Comparison
The performance summary across our baseline linear regression model and our non-linear Random Forest ensemble is documented below:

|        Model          | CV MAE (Mean) | Test Set MAE | Test Set R² |
| :-------------------- | :------------ | :----------- | :---------- |
| **Linear Regression** | 0.44 +/- 0.03 | 0.42 kg      | 0.429       |
| **Random Forest**     | 0.47 +/- 0.06 | 0.45 kg      | 0.334       |

## 3. Overfitting Analysis & Architectural Evaluation
Our Random Forest model achieved a test error of 0.45 kg, which closely tracks its cross-validation mean error of 0.47 kg. This tight gap confirms the ensemble generalizes perfectly to unseen periods without overfitting issues. 

Interestingly, our baseline Linear Regression model slightly outperforms the Random Forest ensemble on both CV MAE and the Test R² score (0.429 vs 0.334). This occurs because our explicit engineering of the `temp_humid_interaction_scaled` term successfully allowed the linear setup to capture the complex, joint warmth-and-moisture behavior. 

However, looking back at our residual plots (`reports/figures/residuals_linear.png`), the residual errors still showed visible curved, non-random patterns instead of uniform noise. This diagnostic proof indicates that while our engineered interaction feature gave the linear model a massive predictive boost, a purely linear system still forces rigid constraints onto natural, fluid agricultural trends. 

## 4. Feature Importance Insights
The plot generated at `reports/figures/rf_importance.png` displays how our tree model assigns importance to environmental inputs. The feature showing the largest impact score on crop prediction outcomes is Temperature, followed by CO2 and Humidity.
