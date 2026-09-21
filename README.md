# Restaurant Revenue Prediction

A Kaggle machine learning project that predicts restaurant revenue from restaurant metadata and other available features.

## Project objective

The goal is to build a reproducible regression workflow that cleans the Kaggle data, compares multiple algorithms, selects a strong estimator, and creates a submission file with revenue predictions.

## Workflow

The notebook covers:

- Loading and inspecting the training and test data
- Removing identifiers from the modeling features
- Encoding categorical variables
- Applying preprocessing appropriate for each model family
- Transforming the skewed revenue target with `log1p`
- Comparing linear and tree-based regressors
- Evaluating models with cross-validation and mean absolute error
- Training the selected model and exporting predictions to CSV

## Models compared

- Ridge Regression
- Lasso Regression
- Elastic Net
- Random Forest Regressor
- LightGBM
- XGBoost

The original analysis identifies `RandomForestRegressor` as the best-performing candidate among the evaluated models based on cross-validated MAE.

## Dataset

The project uses the Kaggle Restaurant Revenue Prediction dataset. The data includes restaurant-related information such as city, restaurant type, opening date, and other numerical or categorical attributes. The target is restaurant revenue.

The repository includes:

```text
.
├── Restaurant Price Prediction.ipynb  # Main analysis notebook
├── train.csv                          # Training data
├── test.csv                           # Test data
├── sampleSubmission.csv               # Kaggle submission template
├── submission.csv                     # Generated predictions
└── README.md
```

## Running the notebook

Install the main dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm jupyter
```

Then launch Jupyter and open `Restaurant Price Prediction.ipynb`:

```bash
jupyter notebook
```

Run the notebook from top to bottom. Depending on the versions of LightGBM and XGBoost installed, small API or parameter changes may be required.

## Evaluation notes

Revenue is modeled on a logarithmic scale to reduce skew and improve stability. When comparing results, ensure that the inverse transformation and evaluation metric are applied consistently. Cross-validation scores are estimates and may differ from the final Kaggle leaderboard score.

## License

No license has been specified for this repository. Contact the repository owner before redistributing or using the code commercially.
