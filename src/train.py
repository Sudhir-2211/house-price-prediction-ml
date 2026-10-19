"""Model Training, Comparison, and Pipeline Serialization.

Author: Sudhir Kumar (GitHub: Sudhir-2211)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error


NUMERIC_FEATURES = [
    "sqft_living",
    "bedrooms",
    "bathrooms",
    "year_built",
    "lot_size_acres",
    "garage_spaces",
    "condition_rating",
    "has_pool",
]

CATEGORICAL_FEATURES = ["location_tier"]


def build_preprocessor() -> ColumnTransformer:
    """Builds scikit-learn ColumnTransformer for numerical and categorical attributes."""
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def train_and_evaluate(data_path: str, models_dir: str):
    os.makedirs(models_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    candidate_models = {
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42),
    }

    results = {}
    best_model_name = None
    best_pipeline = None
    best_r2 = -float("inf")

    print("==================================================================")
    print("           TRAINING & COMPARING HOUSE PRICE REGRESSION MODELS      ")
    print("==================================================================")

    for name, estimator in candidate_models.items():
        preprocessor = build_preprocessor()
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", estimator)])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100.0

        results[name] = {
            "R2_Score": round(float(r2), 4),
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "MAPE_pct": round(float(mape), 2),
        }

        print(f"\n{name}:")
        print(f"  • R² Score : {r2:.4f}")
        print(f"  • MAE      : ${mae:,.2f}")
        print(f"  • RMSE     : ${rmse:,.2f}")
        print(f"  • MAPE     : {mape:.2f}%")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline

    print("\n------------------------------------------------------------------")
    print(f"Best Performing Model: {best_model_name} (R² = {best_r2:.4f})")
    print("------------------------------------------------------------------")

    # Serialize best pipeline
    pipeline_path = os.path.join(models_dir, "house_price_pipeline.joblib")
    joblib.dump(best_pipeline, pipeline_path)
    print(f"Saved best model pipeline to: {pipeline_path}")

    # Save metrics JSON
    metrics_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"best_model": best_model_name, "comparison": results}, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_path}")

    return best_pipeline, results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d_path = os.path.join(base_dir, "data", "house_prices.csv")
    m_dir = os.path.join(base_dir, "models")
    train_and_evaluate(d_path, m_dir)
