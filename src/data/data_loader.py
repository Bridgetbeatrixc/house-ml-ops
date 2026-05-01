from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


FEATURE_COLUMNS = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "YearRemodAdd",
    "LotArea",
    "Neighborhood",
    "HouseStyle",
    "ExterQual",
    "KitchenQual",
    "BsmtQual",
    "GarageFinish",
]


def generate_sample_data(rows: int = 1000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    data = pd.DataFrame(
        {
            "OverallQual": rng.integers(2, 10, rows),
            "GrLivArea": rng.normal(1500, 520, rows).clip(400, 4500),
            "GarageCars": rng.integers(0, 4, rows),
            "GarageArea": rng.normal(470, 210, rows).clip(0, 1200),
            "TotalBsmtSF": rng.normal(1050, 430, rows).clip(0, 3000),
            "FullBath": rng.integers(0, 4, rows),
            "YearBuilt": rng.integers(1900, 2011, rows),
            "YearRemodAdd": rng.integers(1950, 2011, rows),
            "LotArea": rng.normal(10500, 4200, rows).clip(1300, 50000),
            "Neighborhood": rng.choice(
                ["CollgCr", "Veenker", "Crawfor", "NoRidge", "NridgHt", "Somerst"],
                rows,
            ),
            "HouseStyle": rng.choice(["1Story", "2Story", "1.5Fin", "SLvl"], rows),
            "ExterQual": rng.choice(["TA", "Gd", "Ex", "Fa"], rows),
            "KitchenQual": rng.choice(["TA", "Gd", "Ex", "Fa"], rows),
            "BsmtQual": rng.choice(["TA", "Gd", "Ex", "Fa", None], rows),
            "GarageFinish": rng.choice(["Unf", "RFn", "Fin", None], rows),
        }
    )
    quality_bonus = data["OverallQual"] * 21000
    size_bonus = data["GrLivArea"] * 58 + data["TotalBsmtSF"] * 24
    garage_bonus = data["GarageCars"] * 9000 + data["GarageArea"] * 18
    age_penalty = (2011 - data["YearBuilt"]) * 450
    neighborhood_bonus = data["Neighborhood"].isin(["NoRidge", "NridgHt"]).astype(int) * 55000
    kitchen_bonus = data["KitchenQual"].map({"Ex": 45000, "Gd": 24000, "TA": 8000, "Fa": -5000})
    noise = rng.normal(0, 18000, rows)
    data["SalePrice"] = (
        25000
        + quality_bonus
        + size_bonus
        + garage_bonus
        + neighborhood_bonus
        + kitchen_bonus
        - age_penalty
        + noise
    ).clip(35000, 700000)
    return data


def load_data(
    data_path: str | Path,
    target_column: str,
    feature_columns: list[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    path = Path(data_path)
    logger.info(
        "Starting data loading",
        extra={"event": "data_loading_started", "data_path": str(path)},
    )

    if path.exists():
        dataframe = pd.read_csv(path)
        logger.info(
            "Loaded dataset from CSV",
            extra={
                "event": "data_loaded",
                "rows": len(dataframe),
                "columns": list(dataframe.columns),
            },
        )
    else:
        dataframe = generate_sample_data(random_state=random_state)
        logger.warning(
            "CSV not found; generated sample dataset",
            extra={
                "event": "sample_data_generated",
                "data_path": str(path),
                "rows": len(dataframe),
            },
        )

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    if feature_columns:
        missing_features = [column for column in feature_columns if column not in dataframe.columns]
        if missing_features:
            raise ValueError(f"Feature columns not found in dataset: {missing_features}")
        dataframe = dataframe[feature_columns + [target_column]]
        logger.info(
            "Selected configured feature columns",
            extra={
                "event": "feature_columns_selected",
                "feature_count": len(feature_columns),
            },
        )

    return dataframe
