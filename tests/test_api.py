import pytest
from fastapi.testclient import TestClient
from src.api import app  # Importer votre fichier api.py

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_returns_expected_fields():
    payload = {f: 0.5 for f in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert all(k in json_data for k in ["prediction", "probability_default", "risk_label"])

def test_predict_all_none_fields():
    # Teste la robustesse face à un objet vide
    response = client.post("/predict", json={})
    assert response.status_code == 200 

def test_batch_nominal():
    payload = [{ "EXT_SOURCE_1": 0.1 }, { "EXT_SOURCE_1": 0.9 }]
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    assert len(response.json()) == 2

def test_batch_limit_exceeded():
    payload = [{}] * 101
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 400