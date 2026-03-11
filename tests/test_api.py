"""
tests/test_api.py
─────────────────────────────────────────────────────────────────────────────
Tests de l'API Credit Scoring (FastAPI + LightGBM).

Stratégie :
  - Les fixtures `client` et `valid_payload` viennent de conftest.py
  - Le vrai modèle n'est JAMAIS chargé (tout est mocké dans conftest.py)
  - Chaque test est indépendant et couvre un cas précis

Couverture :
  ✅ Endpoints de base  : GET /  |  GET /health
  ✅ Prédiction normale : POST /predict  (cas nominaux)
  ✅ Labels de risque   : 4 zones (très faible / modéré / élevé / très élevé)
  ✅ Batch              : POST /predict/batch  (nominal + limite 101)
  ✅ Cas limites        : payload vide, type incorrect, valeur hors plage
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
import src.api as api_module

# Seuil métier défini dans api.py — on l'importe pour ne pas le dupliquer
THRESHOLD = api_module.OPTIMAL_THRESHOLD


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def set_proba(proba: float):
    """
    Configure le mock du modèle pour renvoyer une probabilité précise.
    Appelé dans chaque test qui a besoin de contrôler la sortie du modèle.
    """
    api_module.model.predict_proba = MagicMock(
        return_value=np.array([[1 - proba, proba]])
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ENDPOINTS DE BASE
# ═══════════════════════════════════════════════════════════════════════════════

def test_root_endpoint(client):
    """GET / doit retourner 200 avec une clé 'message'."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint(client):
    """
    GET /health doit retourner :
      - status = "ok"
      - model_loaded : bool
      - features_count : int
      - threshold : float
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["features_count"], int)
    assert isinstance(data["threshold"], float)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PRÉDICTION NORMALE
# ═══════════════════════════════════════════════════════════════════════════════

def test_predict_returns_expected_fields(client, valid_payload):
    """POST /predict avec payload valide → tous les champs de réponse présents."""
    set_proba(0.5)
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    # Vérifie que tous les champs attendus sont présents
    for field in ["prediction", "probability_default", "risk_label",
                  "threshold_used", "model_available"]:
        assert field in body, f"Champ manquant : {field}"


def test_predict_low_risk(client, valid_payload):
    """proba = 0.15 → prediction = 0 (sous le seuil)."""
    set_proba(0.15)
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 0
    assert body["probability_default"] < THRESHOLD


def test_predict_high_risk(client, valid_payload):
    """proba = 0.80 → prediction = 1 (au-dessus du seuil)."""
    set_proba(0.80)
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert body["probability_default"] > THRESHOLD


def test_predict_all_none_fields(client):
    """
    Payload avec tous les champs à None (tous Optional dans api.py).
    L'API doit répondre 200 — le preprocessor gère les valeurs manquantes.
    """
    set_proba(0.5)
    payload = {
        "EXT_SOURCE_1": None, "EXT_SOURCE_2": None, "EXT_SOURCE_3": None,
        "AMT_CREDIT": None, "AMT_ANNUITY": None, "DAYS_EMPLOYED": None,
        "AMT_GOODS_PRICE": None, "DAYS_BIRTH": None,
        "DAYS_LAST_PHONE_CHANGE": None, "AMT_INCOME_TOTAL": None
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LABELS DE RISQUE (4 zones)
# ═══════════════════════════════════════════════════════════════════════════════

def test_risk_label_tres_faible(client, valid_payload):
    """proba = 0.10 → 'Très faible risque'  (règle : proba < 0.2)."""
    set_proba(0.10)
    response = client.post("/predict", json=valid_payload)
    assert response.json()["risk_label"] == "Très faible risque"


def test_risk_label_modere(client, valid_payload):
    """proba = 0.30 → 'Risque modéré'  (règle : 0.2 <= proba < THRESHOLD)."""
    set_proba(0.30)
    response = client.post("/predict", json=valid_payload)
    assert response.json()["risk_label"] == "Risque modéré"


def test_risk_label_eleve(client, valid_payload):
    """proba = 0.60 → 'Risque élevé'  (règle : THRESHOLD <= proba < 0.7)."""
    set_proba(0.60)
    response = client.post("/predict", json=valid_payload)
    assert response.json()["risk_label"] == "Risque élevé"


def test_risk_label_tres_eleve(client, valid_payload):
    """proba = 0.80 → 'Risque très élevé'  (règle : proba >= 0.7)."""
    set_proba(0.80)
    response = client.post("/predict", json=valid_payload)
    assert response.json()["risk_label"] == "Risque très élevé"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BATCH
# ═══════════════════════════════════════════════════════════════════════════════

def test_batch_nominal(client, valid_payload):
    """POST /predict/batch avec 2 requêtes → liste de 2 réponses."""
    set_proba(0.5)
    response = client.post("/predict/batch", json=[valid_payload, valid_payload])
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_batch_limit_exceeded(client, valid_payload):
    """POST /predict/batch avec 101 requêtes → 400 (limite = 100)."""
    response = client.post("/predict/batch", json=[valid_payload] * 101)
    assert response.status_code == 400


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CAS LIMITES — robustesse de la validation Pydantic
# ═══════════════════════════════════════════════════════════════════════════════

def test_invalid_ext_source_above_range(client, valid_payload):
    """
    EXT_SOURCE_1 = 1.5 → hors plage [0, 1] définie dans le schéma Pydantic.
    Pydantic doit rejeter avec 422 (Unprocessable Entity).
    """
    payload = valid_payload.copy()
    payload["EXT_SOURCE_1"] = 1.5          # ge=0, le=1 dans CreditRequest
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_ext_source_below_range(client, valid_payload):
    """
    EXT_SOURCE_2 = -0.1 → hors plage [0, 1].
    Pydantic doit rejeter avec 422.
    """
    payload = valid_payload.copy()
    payload["EXT_SOURCE_2"] = -0.1
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_type_string_instead_of_float(client, valid_payload):
    """
    EXT_SOURCE_1 = "abc" → type incorrect (float attendu).
    Pydantic doit rejeter avec 422.
    """
    payload = valid_payload.copy()
    payload["EXT_SOURCE_1"] = "abc"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_empty_payload(client):
    """
    Payload vide {} → tous les champs sont Optional donc Pydantic accepte.
    L'API doit répondre 200 (le preprocessor gère les None).
    """
    set_proba(0.5)
    response = client.post("/predict", json={})
    assert response.status_code == 200
