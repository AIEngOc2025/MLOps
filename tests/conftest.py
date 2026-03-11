# tests/conftest.py

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# ── 1. Définition des mocks partagés ──────────────────────────────────────────
# Ces objets remplacent le vrai modèle et preprocessor dans TOUS les tests
# → évite de charger les vrais fichiers .pkl/.joblib à chaque test

MOCK_FEATURES = [
    'EXT_SOURCE_1', 'EXT_SOURCE_3', 'EXT_SOURCE_2', 'AMT_CREDIT',
    'AMT_ANNUITY', 'DAYS_EMPLOYED', 'AMT_GOODS_PRICE',
    'DAYS_BIRTH', 'DAYS_LAST_PHONE_CHANGE', 'AMT_INCOME_TOTAL'
]

# ── 2. Fixture client : créée UNE FOIS, partagée entre tous les fichiers ───────
@pytest.fixture(scope="session")
def client():
    """
    Crée le TestClient FastAPI avec les vrais artefacts remplacés par des mocks.
    scope="session" = créé une seule fois pour toute la session de tests.
    """
    mock_preprocessor = MagicMock()
    mock_preprocessor.transform = MagicMock(return_value=np.zeros((1, 10)))
    mock_model = MagicMock()

    with patch("joblib.load") as mock_load:
        mock_load.side_effect = [mock_preprocessor, MOCK_FEATURES, mock_model]
        from src.api import app
        import src.api as api_module
        api_module.preprocessor      = mock_preprocessor
        api_module.model             = mock_model
        api_module.SELECTED_FEATURES = MOCK_FEATURES

    return TestClient(app)


# ── 3. Fixture payload valide : données de test réutilisables ─────────────────
@pytest.fixture
def valid_payload():
    """Payload standard valide pour les tests de /predict."""
    return {
        "EXT_SOURCE_1": 0.6, "EXT_SOURCE_2": 0.7, "EXT_SOURCE_3": 0.5,
        "AMT_CREDIT": 120000.0, "AMT_ANNUITY": 6000.0,
        "DAYS_EMPLOYED": -1200.0, "AMT_GOODS_PRICE": 100000.0,
        "DAYS_BIRTH": -16000.0, "DAYS_LAST_PHONE_CHANGE": -50.0,
        "AMT_INCOME_TOTAL": 60000.0
    }