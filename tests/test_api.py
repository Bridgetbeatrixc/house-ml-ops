from fastapi.testclient import TestClient

from src.api.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_frontend_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "House Price Prediction" in response.text


def test_predict_endpoint():
    payload = {
        "OverallQual": 7,
        "GrLivArea": 1710,
        "GarageCars": 2,
        "GarageArea": 548,
        "TotalBsmtSF": 856,
        "FullBath": 2,
        "YearBuilt": 2003,
        "YearRemodAdd": 2003,
        "LotArea": 8450,
        "Neighborhood": "CollgCr",
        "HouseStyle": "2Story",
        "ExterQual": "Gd",
        "KitchenQual": "Gd",
        "BsmtQual": "Gd",
        "GarageFinish": "RFn",
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_metrics_endpoint():
    with TestClient(app) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    assert "house_prediction_requests_total" in response.text
