# House Price Prediction MLOps

A modular house price prediction pipeline built on the Kaggle *House Prices: Advanced Regression Techniques* dataset, with MLflow experiment tracking, structured logging, FastAPI model serving, Docker deployment and Prometheus monitoring. A Streamlit app shows the data, model performance and batch predictions.

## Highlights

- **Reproducible training** with config-driven preprocessing and model code in `src/`
- **Experiment tracking** in MLflow (RMSE, MAE, R², plus model and preprocessor artifacts)
- **Model serving** through FastAPI with `/predict`, `/health` and a Prometheus `/metrics` endpoint
- **Containerised** with Docker Compose (API + Prometheus)
- **Structured JSON logging** and a pytest suite

## Project structure

```text
house_price_prediction/
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   ├── data/
│   │   ├── data_loader.py
│   │   └── data_preprocessor.py
│   ├── models/
│   │   ├── model.py
│   │   └── trainer.py
│   └── utils/
│       ├── config.py
│       └── logger.py
├── config/
│   └── config.yaml
├── monitoring/
│   └── prometheus.yml
├── tests/
│   └── test_pipeline.py
├── artifacts/
├── mlruns/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (use source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

## Dataset

Download the data from [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and put the training file at `data/train.csv`, or use the Kaggle API:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_kaggle_data.ps1
```

Before running the script, make sure `kaggle.json` is in `C:\Users\<username>\.kaggle\kaggle.json` and that you have accepted the competition rules on Kaggle.

The pipeline predicts `SalePrice` from this feature subset, used for both training and serving:

```text
OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF, FullBath,
YearBuilt, YearRemodAdd, LotArea, Neighborhood, HouseStyle, ExterQual,
KitchenQual, BsmtQual, GarageFinish
```

If `data/train.csv` is missing, the pipeline generates a synthetic dataset with the same schema so the demo still runs.

## Training

```bash
python -m src.models.trainer
```

Outputs:

- model saved to `artifacts/model.joblib`
- preprocessor saved to `artifacts/preprocessor.joblib`
- RMSE, MAE and R² logged to MLflow, along with both joblib artifacts
- structured JSON logs in the terminal

## Running the API

After training, start the API:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Endpoints: `GET /`, `GET /health`, `GET /metrics`, `POST /predict`. A simple frontend is served at `http://localhost:8000/`.

On Windows PowerShell you can also run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_api.ps1
```

Example request:

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"OverallQual\":7,\"GrLivArea\":1710,\"GarageCars\":2,\"GarageArea\":548,\"TotalBsmtSF\":856,\"FullBath\":2,\"YearBuilt\":2003,\"YearRemodAdd\":2003,\"LotArea\":8450,\"Neighborhood\":\"CollgCr\",\"HouseStyle\":\"2Story\",\"ExterQual\":\"Gd\",\"KitchenQual\":\"Gd\",\"BsmtQual\":\"Gd\",\"GarageFinish\":\"RFn\"}"
```

## Docker

```bash
docker compose up --build
```

- FastAPI: `http://localhost:8000`
- Prometheus: `http://localhost:9090`

## MLflow UI

```bash
mlflow ui --backend-store-uri mlruns --port 5000
```

Then open `http://localhost:5000`.

## Streamlit app

```bash
streamlit run streamlit_app.py
```

(or `powershell -ExecutionPolicy Bypass -File scripts/run_streamlit.ps1` on Windows)

The Streamlit app includes:

- an about section and project overview
- CSV upload for batch house price predictions
- feature distribution and correlation charts
- model performance (RMSE, MAE, R²)

For Streamlit Community Cloud, `runtime.txt` pins Python 3.11 so packages like `pandas` and `pyarrow` install from wheels instead of compiling from source.

## Tests

```bash
pytest
```
