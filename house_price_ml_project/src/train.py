"""
Train the House Price Prediction models.

Dataset: Kaggle House Prices / Ames Housing-style train.csv
Target: SalePrice
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
TARGET = "SalePrice"

X = df.drop(columns=[TARGET, "Id"], errors="ignore")
y = df[TARGET]

numeric_features = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            categorical_features,
        ),
    ],
    remainder="drop",
)

candidates = {
    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        max_features=0.7,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        loss="huber",
        random_state=42,
    ),
    "Ridge": Ridge(alpha=20),
}

results = {}
trained = {}

for name, model in candidates.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    prediction = pipe.predict(X_test)

    results[name] = {
        "MAE": float(mean_absolute_error(y_test, prediction)),
        "RMSE": float(mean_squared_error(y_test, prediction) ** 0.5),
        "R2": float(r2_score(y_test, prediction)),
    }
    trained[name] = pipe

best_name = min(results, key=lambda name: results[name]["RMSE"])
best_model = trained[best_name]

joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

metadata = {
    "target": TARGET,
    "best_model": best_name,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "all_features": X.columns.tolist(),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "metrics": results,
}
(MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
(REPORT_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

# Export the most important transformed features when supported.
fitted_preprocessor = best_model.named_steps["preprocessor"]
estimator = best_model.named_steps["model"]
if hasattr(estimator, "feature_importances_"):
    feature_names = fitted_preprocessor.get_feature_names_out()
    importance = estimator.feature_importances_
    fi = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
    )
    fi.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

print("\nModel comparison:")
for name, scores in results.items():
    print(
        f"{name:20s} | "
        f"MAE={scores['MAE']:,.0f} | "
        f"RMSE={scores['RMSE']:,.0f} | "
        f"R2={scores['R2']:.4f}"
    )
print(f"\nBest model: {best_name}")
