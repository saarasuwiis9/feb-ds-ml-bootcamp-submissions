# Reflection Paper: House Price Prediction
## February 22, 2026

## 1. Implementation Process

I loaded the clean house dataset and split it into features (X) and target (y = Price). The features included Size_sqft, Bedrooms, Bathrooms, YearBuilt, Location, HouseAge, and engineered features. I used an 80/20 train-test split with random_state=42 for reproducibility.

I trained two models: Linear Regression and Random Forest (100 trees). Both models were evaluated using R², MAE, MSE, and RMSE metrics.

## 2. Model Comparison

Random Forest performed better than Linear Regression across all metrics. Random Forest achieved higher R² (better fit) and lower MAE/RMSE (smaller errors). This shows Random Forest captures non-linear patterns better than Linear Regression.

Linear Regression assumes linear relationships between features and price, which may not hold for real estate data. Random Forest handles complex interactions between features like location, size, and age more effectively.

## 3. Random Forest Explanation

Random Forest is an ensemble method that builds multiple decision trees during training. Each tree is trained on a random subset of data and features. For predictions, all trees vote and the average is returned.

This approach reduces overfitting compared to a single decision tree. By averaging many trees, Random Forest produces more stable and accurate predictions. It also handles non-linear relationships and feature interactions naturally.

## 4. Metrics Discussion

- R²: Measures how well the model explains variance in prices. Higher is better (max 1.0).
- MAE: Average absolute error in dollars. Lower is better. Easy to interpret.
- MSE: Squared errors, penalizes large mistakes more. Lower is better.
- RMSE: Square root of MSE, same units as price. Lower is better.

For house prices, MAE and RMSE are most useful because they show prediction error in dollars. R² shows overall model fit quality.

## 5. Findings

Random Forest outperformed Linear Regression, showing that house prices have non-linear relationships with features. The models achieved R² around 0.85-0.86, meaning they explain 85-86% of price variance.

The sanity check showed both models make reasonable predictions close to actual prices. Random Forest's lower MAE means it's more accurate on average. For real estate applications, Random Forest is the better choice due to its ability to capture complex patterns in the data.
