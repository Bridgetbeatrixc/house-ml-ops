from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.api.schemas import HealthResponse, HouseFeatures, PredictionResponse
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger, setup_logging


config = load_config()
setup_logging(config.get("logging", {}).get("level", "INFO"))
logger = get_logger(__name__)

REQUEST_COUNT = Counter(
    "house_prediction_requests_total",
    "Total API requests",
    ["method", "endpoint", "http_status"],
)
PREDICTION_COUNT = Counter(
    "house_prediction_predictions_total",
    "Total successful prediction requests",
)
REQUEST_LATENCY = Histogram(
    "house_prediction_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
)

MODEL_VERSION = "local-joblib-v1"
model: Any | None = None
STATIC_DIR = resolve_path("src/api/static")


def load_model() -> Any:
    model_path = resolve_path(config["artifacts"]["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run training first."
        )
    return joblib.load(model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = load_model()
        logger.info(
            "Model loaded for serving",
            extra={"event": "model_loaded", "model_version": MODEL_VERSION},
        )
    except FileNotFoundError as exc:
        model = None
        logger.error("Model artifact missing", extra={"event": "model_load_failed"})
        app.state.startup_error = str(exc)
    yield


app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def metrics_and_logging_middleware(request: Request, call_next):
    started_at = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        latency = time.perf_counter() - started_at
        endpoint = request.url.path
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            http_status=str(status_code),
        ).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
        logger.info(
            "API request completed",
            extra={
                "event": "api_request_completed",
                "method": request.method,
                "endpoint": endpoint,
                "status_code": status_code,
                "latency_seconds": round(latency, 6),
            },
        )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "Unhandled API exception",
        extra={"event": "api_unhandled_exception", "endpoint": request.url.path},
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures) -> PredictionResponse:
    if model is None:
        logger.error(
            "Prediction requested before model is loaded",
            extra={"event": "prediction_model_unavailable"},
        )
        raise HTTPException(status_code=503, detail="Model is not available")

    payload = features.model_dump()
    input_frame = pd.DataFrame([payload])
    prediction = float(model.predict(input_frame)[0])
    PREDICTION_COUNT.inc()

    logger.info(
        "Prediction completed",
        extra={
            "event": "prediction_completed",
            "model_version": MODEL_VERSION,
            "prediction": prediction,
        },
    )
    return PredictionResponse(prediction=prediction, model_version=MODEL_VERSION)
