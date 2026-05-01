import pandas as pd

from src.data.data_loader import generate_sample_data
from src.data.data_preprocessor import build_preprocessor, split_features_target
from src.models.model import create_model


def test_generate_sample_data_contains_target():
    dataframe = generate_sample_data(rows=20, random_state=7)
    assert "SalePrice" in dataframe.columns
    assert "Neighborhood" in dataframe.columns
    assert len(dataframe) == 20


def test_preprocessor_transforms_sample_data():
    dataframe = generate_sample_data(rows=50, random_state=7)
    split_data = split_features_target(dataframe, "SalePrice", 0.2, 7)
    preprocessor = build_preprocessor(dataframe, "SalePrice")
    transformed = preprocessor.fit_transform(split_data.X_train)
    assert transformed.shape[0] == len(split_data.X_train)


def test_model_factory_random_forest():
    model = create_model(
        "random_forest",
        {"n_estimators": 5, "max_depth": 2, "random_state": 7},
    )
    assert model.n_estimators == 5
