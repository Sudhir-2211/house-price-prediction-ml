"""Automated unit tests for House Price Regression model."""

import pytest
import os
import joblib
import pandas as pd
from src.predict import predict_price
from src.train import NUMERIC_FEATURES, CATEGORICAL_FEATURES


@pytest.fixture(scope="module")
def model_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "house_price_pipeline.joblib")
    assert os.path.exists(model_path), "Model pipeline artifact must exist"
    return joblib.load(model_path)


@pytest.fixture(scope="module")
def dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "house_prices.csv")
    assert os.path.exists(data_path), "Dataset must exist"
    return pd.read_csv(data_path)


def test_dataset_integrity(dataset):
    assert len(dataset) >= 2000
    assert not dataset.isnull().any().any()
    assert (dataset["price"] > 50000).all()
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["price"]:
        assert col in dataset.columns


def test_model_inference(model_pipeline):
    sample = pd.DataFrame([{
        "sqft_living": 2500,
        "bedrooms": 4,
        "bathrooms": 2.5,
        "location_tier": "Suburban Central",
        "year_built": 2010,
        "lot_size_acres": 0.3,
        "garage_spaces": 2,
        "condition_rating": 4,
        "has_pool": 0,
    }])
    pred = model_pipeline.predict(sample)
    assert len(pred) == 1
    assert 200000 < pred[0] < 1200000


def test_predict_price_wrapper():
    res = predict_price(
        sqft_living=2000,
        bedrooms=3,
        bathrooms=2.0,
        location_tier="Urban Prime",
        year_built=2015,
        lot_size_acres=0.15,
        garage_spaces=1,
        condition_rating=5,
        has_pool=0,
    )
    assert "estimated_price" in res
    assert "confidence_lower" in res
    assert "confidence_upper" in res
    assert res["confidence_lower"] < res["estimated_price"] < res["confidence_upper"]


def test_scale_monotonicity(model_pipeline):
    # Larger house in same location must predict higher price than smaller house
    small_house = pd.DataFrame([{
        "sqft_living": 1000,
        "bedrooms": 2,
        "bathrooms": 1.0,
        "location_tier": "Suburban Central",
        "year_built": 1990,
        "lot_size_acres": 0.1,
        "garage_spaces": 1,
        "condition_rating": 3,
        "has_pool": 0,
    }])
    large_house = pd.DataFrame([{
        "sqft_living": 4000,
        "bedrooms": 5,
        "bathrooms": 3.5,
        "location_tier": "Suburban Central",
        "year_built": 1990,
        "lot_size_acres": 0.1,
        "garage_spaces": 1,
        "condition_rating": 3,
        "has_pool": 0,
    }])
    pred_small = model_pipeline.predict(small_house)[0]
    pred_large = model_pipeline.predict(large_house)[0]
    assert pred_large > pred_small
