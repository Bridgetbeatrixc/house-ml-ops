from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_features_target(
    dataframe: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> SplitData:
    logger.info(
        "Splitting features and target",
        extra={"event": "preprocessing_split_started", "target_column": target_column},
    )
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    logger.info(
        "Completed train-test split",
        extra={
            "event": "preprocessing_split_completed",
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        },
    )
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def build_preprocessor(dataframe: pd.DataFrame, target_column: str) -> ColumnTransformer:
    feature_frame = dataframe.drop(columns=[target_column])
    numeric_features = feature_frame.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = feature_frame.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    logger.info(
        "Built preprocessing pipeline",
        extra={
            "event": "preprocessing_pipeline_built",
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        },
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
