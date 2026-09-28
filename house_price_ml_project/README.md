# House Price Prediction — End-to-End ML Project

A complete machine-learning regression project built around the uploaded `train.csv` dataset.

## Dataset

- Rows: **1,460**
- Columns: **81**
- Target: **SalePrice**
- Identifier removed from modeling: **Id**
- Numeric input features: **36**
- Categorical input features: **43**

## Project workflow

**CSV → EDA → preprocessing → train/test split → model training → evaluation → model saving → Streamlit web interface**

### Preprocessing

The training pipeline automatically:

1. Separates `SalePrice` as the target.
2. Removes `Id`.
3. Detects numeric and categorical columns.
4. Fills numeric missing values with the median.
5. Fills categorical missing values with the most frequent category.
6. One-hot encodes categorical columns.
7. Applies the same transformations to test data and web-app inputs.

### Models

The project compares:
- Random Forest Regressor
- Gradient Boosting Regressor
- Ridge Regression

On the included 80/20 random split (`random_state=42`), the current trained results are:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | $17,226 | $28,712 | 0.8925 |
| Gradient Boosting | $16,542 | $28,546 | 0.8938 |
| Ridge | $18,698 | $30,918 | 0.8754 |

Best current model: **Gradient Boosting**

> Metrics can change slightly if you change the split, preprocessing, or model hyperparameters.

## How to run

### 1. Open terminal in this folder

```bash
cd house_price_ml_project
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install packages

```bash
pip install -r requirements.txt
```

### 4. Retrain the model (optional)

```bash
python src/train.py
```

### 5. Run the web application

```bash
streamlit run app.py
```

Your browser should open the local Streamlit application.

## Important files

- `data/train.csv` — dataset
- `src/eda.py` — basic EDA report generator
- `src/train.py` — preprocessing, training, evaluation and model saving
- `models/best_model.joblib` — trained complete pipeline
- `models/metadata.json` — feature schema and metrics
- `reports/metrics.json` — model comparison
- `reports/feature_importance.csv` — top features for tree model
- `app.py` — web interface
- `requirements.txt` — Python dependencies

## Viva / interview explanation

**Problem:** Predict a house's sale price from property characteristics.

**Type:** Supervised Machine Learning → Regression.

**Why regression?** `SalePrice` is a continuous numerical value.

**Why remove Id?** It identifies rows but does not describe the property.

**Why imputation?** ML estimators generally cannot directly work with missing values in these pipelines.

**Why One-Hot Encoding?** Models need numerical input, while columns such as `Neighborhood`, `HouseStyle`, and `SaleType` are categorical.

**Why Pipeline?** It guarantees that training and prediction use the same preprocessing steps.

**Why compare models?** Different algorithms learn different patterns. We select using validation/test metrics rather than assuming one algorithm is always best.

**MAE:** Average absolute prediction error in dollars.

**RMSE:** Like MAE, but penalizes large errors more strongly.

**R²:** How much variance in house prices the model explains relative to a baseline; closer to 1 is generally better.

## Extensions you can add later

- Log-transform `SalePrice` and compare performance.
- Add cross-validation and hyperparameter tuning with `GridSearchCV`/`RandomizedSearchCV`.
- Add SHAP explanations.
- Add prediction-vs-actual plots.
- Deploy the Streamlit app to a cloud service.
- Add a FastAPI REST API alongside the Streamlit UI.
