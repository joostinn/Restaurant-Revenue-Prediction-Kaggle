This project builds a complete machine learning pipeline to predict annual restaurant revenue using the Kaggle Restaurant Revenue Prediction dataset.
The workflow includes:
- Data cleaning and preprocessing
- Feature engineering
- Model comparison across linear and tree‑based algorithms
- Hyperparameter tuning
- Final model training using the best estimator
- CSV export of predictions
The goal is to identify the most accurate regression model and produce a clean, reproducible pipeline suitable for real‑world deployment or Kaggle submission.

📂 Dataset
The dataset contains:
- Restaurant metadata (city, type, opening date)
- Numerical and categorical features
- Target variable: revenue
Revenue is highly skewed, so a log‑transform is applied to stabilize variance and improve model performance.

🧹 Preprocessing
✔ Dropped unnecessary columns
- Id was removed at the start
- A new ID column is recreated later for submission
✔ Encoded categorical variables
- Label Encoding / One‑Hot Encoding depending on feature type
✔ Feature scaling
- Only applied to linear models (Ridge, Lasso, ElasticNet)
- Tree models (RandomForest, LightGBM, XGBoost) use raw features
✔ Target transformation
Revenue is transformed using:
y_{\mathrm{log}}=\log (1+y)
This improves model stability and reduces MAE.

🤖 Model Comparison
The following models were evaluated using 10‑fold cross‑validation and Mean Absolute Error (MAE):
🔹 Linear Models (scaled)
- Ridge Regression
- Lasso Regression
- ElasticNet
🔹 Tree‑Based Models (unscaled)
- RandomForestRegressor
- LightGBM
- XGBoost
Each model was trained on the log‑transformed target and evaluated fairly using consistent CV splits.

🏆 Best Model: RandomForestRegressor
After comparing all models, RandomForestRegressor achieved the lowest MAE and was selected as the final model.
Reasons it performed best:
- Handles nonlinear relationships
- Robust to outliers
- Works well with mixed feature types
- No scaling required
- Stable performance across folds
