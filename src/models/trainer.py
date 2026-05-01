from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

from src.data.data_loader import load_data
from src.data.data_preprocessor import build_preprocessor, split_features_target
from src.models.model import create_model
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger, setup_logging


logger = get_logger(__name__)


def evaluate_model(model: Pipeline, X_test, y_test) -> dict[str, float]:
    predictions = model.predict(X_test)
    return {
        "rmse": root_mean_squared_error(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }


def train(config: dict[str, Any]) -> dict[str, Any]:
    project_config = config["project"]
    data_config = config["data"]
    model_config = config["model"]
    mlflow_config = config["mlflow"]
    artifact_config = config["artifacts"]

    artifact_dir = resolve_path(artifact_config["directory"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = artifact_dir / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)
    tempfile.tempdir = str(temp_dir)

    dataframe = load_data(
        resolve_path(data_config["path"]),
        target_column=data_config["target_column"],
        feature_columns=data_config.get("feature_columns"),
        random_state=project_config["random_state"],
    )
    split_data = split_features_target(
        dataframe,
        target_column=data_config["target_column"],
        test_size=data_config["test_size"],
        random_state=project_config["random_state"],
    )
    preprocessor = build_preprocessor(dataframe, data_config["target_column"])
    estimator = create_model(model_config["type"], model_config["params"])
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    mlflow_tracking_path = resolve_path(mlflow_config["tracking_uri"])
    mlflow_tracking_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlflow_tracking_path.as_uri())
    mlflow.set_experiment(mlflow_config["experiment_name"])

    logger.info(
        "Starting model training",
        extra={
            "event": "training_started",
            "model_type": model_config["type"],
            "params": model_config["params"],
        },
    )

    with mlflow.start_run(run_name=project_config["name"]) as run:
        mlflow.log_param("model_type", model_config["type"])
        mlflow.log_params(model_config["params"])
        mlflow.log_param("target_column", data_config["target_column"])
        mlflow.log_param("test_size", data_config["test_size"])
        mlflow.log_param("feature_count", len(dataframe.columns) - 1)

        pipeline.fit(split_data.X_train, split_data.y_train)
        metrics = evaluate_model(pipeline, split_data.X_test, split_data.y_test)
        mlflow.log_metrics(metrics)

        model_path = resolve_path(artifact_config["model_path"])
        preprocessor_path = resolve_path(artifact_config["preprocessor_path"])
        joblib.dump(pipeline, model_path)
        joblib.dump(preprocessor, preprocessor_path)

        if mlflow_config.get("log_model_artifact", True):
            try:
                mlflow.sklearn.log_model(pipeline, artifact_path="model")
            except PermissionError:
                logger.warning(
                    "MLflow native model logging failed; continuing with joblib artifact",
                    extra={"event": "mlflow_native_model_logging_skipped"},
                )
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(preprocessor_path))

        logger.info(
            "Completed model training",
            extra={
                "event": "training_completed",
                "run_id": run.info.run_id,
                "metrics": metrics,
                "model_path": str(model_path),
            },
        )

    return {"model": pipeline, "metrics": metrics, "model_path": Path(model_path)}


def main() -> None:
    config = load_config()
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    result = train(config)
    logger.info(
        "Training command finished",
        extra={"event": "training_command_finished", "metrics": result["metrics"]},
    )


if __name__ == "__main__":
    main()
