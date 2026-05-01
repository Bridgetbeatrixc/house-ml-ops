from __future__ import annotations

from typing import Any

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression


def create_model(model_type: str, params: dict[str, Any]):
    if model_type == "random_forest":
        return RandomForestRegressor(**params)
    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(**params)
    if model_type == "linear_regression":
        return LinearRegression(**params)
    raise ValueError(f"Unsupported model type: {model_type}")
