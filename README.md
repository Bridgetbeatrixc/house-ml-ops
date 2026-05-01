# House Price Prediction MLOps

Proyek ini berisi pipeline prediksi harga rumah berbasis dataset Kaggle House Prices - Advanced Regression Techniques yang modular, dilengkapi MLflow experiment tracking, structured logging, FastAPI serving, dan Docker deployment dengan monitoring Prometheus.

## Struktur

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

## Setup Lokal

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Unduh dataset dari [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data), lalu letakkan file training di:

```text
data/train.csv
```

Atau gunakan Kaggle API:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_kaggle_data.ps1
```

Sebelum menjalankan script, pastikan `kaggle.json` sudah ada di `C:\Users\<username>\.kaggle\kaggle.json` dan rules competition sudah di-accept di Kaggle.

Pipeline memakai target `SalePrice` dan subset fitur berikut untuk training serta API serving:

```text
OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF, FullBath,
YearBuilt, YearRemodAdd, LotArea, Neighborhood, HouseStyle, ExterQual,
KitchenQual, BsmtQual, GarageFinish
```

Jika `data/train.csv` belum tersedia, pipeline akan membuat dataset sintetis dengan schema yang sama agar demo tetap bisa dijalankan.

## Training Model

Jalankan training dengan:

```bash
python -m src.models.trainer
```

Output utama:

- model tersimpan di `artifacts/model.joblib`
- preprocessor tersimpan di `artifacts/preprocessor.joblib`
- metrik training dilog ke MLflow: RMSE, MAE, dan R2
- artifact joblib model dan preprocessor dilog ke MLflow
- structured log tampil sebagai JSON di terminal

## Menjalankan API

Pastikan model sudah dilatih, lalu jalankan:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Endpoint:

- `GET /`
- `GET /health`
- `GET /metrics`
- `POST /predict`

Frontend sederhana tersedia di `http://localhost:8000/`.

Di Windows PowerShell, frontend/API juga bisa dijalankan dengan:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_api.ps1
```

Contoh request:

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"OverallQual\":7,\"GrLivArea\":1710,\"GarageCars\":2,\"GarageArea\":548,\"TotalBsmtSF\":856,\"FullBath\":2,\"YearBuilt\":2003,\"YearRemodAdd\":2003,\"LotArea\":8450,\"Neighborhood\":\"CollgCr\",\"HouseStyle\":\"2Story\",\"ExterQual\":\"Gd\",\"KitchenQual\":\"Gd\",\"BsmtQual\":\"Gd\",\"GarageFinish\":\"RFn\"}"
```

## Docker

```bash
docker compose up --build
```

Service:

- FastAPI: `http://localhost:8000`
- Prometheus: `http://localhost:9090`

## MLflow UI

```bash
mlflow ui --backend-store-uri mlruns --port 5000
```

Lalu buka `http://localhost:5000`.

## Testing

```bash
pytest
```
