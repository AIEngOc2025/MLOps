"""
Credit Scoring API — FastAPI
Modèle : LightGBM | 10 features | Seuil métier : 0.48

Logging structuré :
  - Chaque appel /predict est loggé en JSON dans logs/predictions.jsonl
  - Contenu : timestamp, latence, inputs, outputs
  - Format JSONL (une ligne JSON par appel) → facile à parser pour Evidently
"""

import time
import json
import traceback
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="Credit Scoring API",
    description="API de scoring crédit basée sur LightGBM (Home Credit dataset)",
    version="1.0.0",
)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR        = os.path.join(BASE_DIR, "models/")
OPTIMAL_THRESHOLD = float(os.getenv("OPTIMAL_THRESHOLD", 0.48))

# ─── LOGGING ──────────────────────────────────────────────────────────────────
# Dossier de logs créé automatiquement s'il n'existe pas
LOGS_DIR      = Path(BASE_DIR) / "logs"
LOGS_DIR.mkdir(exist_ok=True)
PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"   # une ligne JSON par appel

# Logger Python standard pour les erreurs système
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_prediction(entry: dict) -> None:
    """
    Écrit une entrée de log en format JSONL (JSON Lines).
    Chaque appel /predict génère une ligne JSON dans predictions.jsonl.
    Ce fichier sera lu par Evidently pour l'analyse de drift.
    """
    try:
        with open(PREDICTIONS_LOG, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        # On ne bloque jamais l'API si le logging échoue
        logger.warning(f"⚠️ Erreur logging : {e}")


# ─── CHARGEMENT DES ARTEFACTS ─────────────────────────────────────────────────
# Chargement UNE SEULE FOIS au démarrage — jamais dans les endpoints
try:
    preprocessor      = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
    SELECTED_FEATURES = joblib.load(os.path.join(MODELS_DIR, "selected_features.pkl"))
    model_path        = os.path.join(MODELS_DIR, "model.joblib")
    model             = joblib.load(model_path) if os.path.exists(model_path) else None
    logger.info("✅ Artefacts chargés avec succès")
except Exception as e:
    logger.error(f"⚠️ Erreur chargement artefacts : {e}")
    model, preprocessor, SELECTED_FEATURES = None, None, []


# ─── SCHÉMAS ──────────────────────────────────────────────────────────────────
class CreditRequest(BaseModel):
    # Ordre EXACT du preprocessor.pkl (10 features)
    EXT_SOURCE_1:           Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_3:           Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_2:           Optional[float] = Field(None, ge=0, le=1)
    AMT_CREDIT:             Optional[float] = Field(None)
    AMT_ANNUITY:            Optional[float] = Field(None)
    DAYS_EMPLOYED:          Optional[float] = Field(None)
    AMT_GOODS_PRICE:        Optional[float] = Field(None)
    DAYS_BIRTH:             Optional[float] = Field(None)
    DAYS_LAST_PHONE_CHANGE: Optional[float] = Field(None)
    AMT_INCOME_TOTAL:       Optional[float] = Field(None)


class CreditResponse(BaseModel):
    prediction:          int
    probability_default: float
    risk_label:          str
    threshold_used:      float
    model_available:     bool


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Credit Scoring API — voir /docs pour la documentation."}


@app.get("/health")
def health():
    return {
        "status":         "ok",
        "model_loaded":   model is not None,
        "features_count": len(SELECTED_FEATURES),
        "threshold":      OPTIMAL_THRESHOLD,
    }


@app.post("/predict", response_model=CreditResponse)
def predict(data: CreditRequest):
    # ── Mesure du temps de début ───────────────────────────────────────────────
    start_time = time.time()

    try:
        # ── Prédiction ────────────────────────────────────────────────────────
        input_dict = {f: getattr(data, f, None) for f in SELECTED_FEATURES}
        input_df   = pd.DataFrame([input_dict], columns=SELECTED_FEATURES)

        if model is not None and preprocessor is not None:
            X     = preprocessor.transform(input_df)
            proba = float(model.predict_proba(X)[0][1])
        else:
            proba = 0.5

        prediction = int(proba >= OPTIMAL_THRESHOLD)

        if proba < 0.2:
            risk_label = "Très faible risque"
        elif proba < OPTIMAL_THRESHOLD:
            risk_label = "Risque modéré"
        elif proba < 0.7:
            risk_label = "Risque élevé"
        else:
            risk_label = "Risque très élevé"

        # ── Construction de la réponse ────────────────────────────────────────
        response = {
            "prediction":          prediction,
            "probability_default": round(proba, 4),
            "risk_label":          risk_label,
            "threshold_used":      OPTIMAL_THRESHOLD,
            "model_available":     model is not None,
        }

        # ── Logging APRÈS la prédiction ───────────────────────────────────────
        # On loggue tout ce qui est nécessaire pour l'analyse de drift
        latency_ms = round((time.time() - start_time) * 1000, 2)

        log_prediction({
            # ── Métadonnées ───────────────────────────────────────────────────
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "latency_ms": latency_ms,
            # ── Inputs (features envoyées par le client) ──────────────────────
            "EXT_SOURCE_1":           data.EXT_SOURCE_1,
            "EXT_SOURCE_2":           data.EXT_SOURCE_2,
            "EXT_SOURCE_3":           data.EXT_SOURCE_3,
            "AMT_CREDIT":             data.AMT_CREDIT,
            "AMT_ANNUITY":            data.AMT_ANNUITY,
            "DAYS_EMPLOYED":          data.DAYS_EMPLOYED,
            "AMT_GOODS_PRICE":        data.AMT_GOODS_PRICE,
            "DAYS_BIRTH":             data.DAYS_BIRTH,
            "DAYS_LAST_PHONE_CHANGE": data.DAYS_LAST_PHONE_CHANGE,
            "AMT_INCOME_TOTAL":       data.AMT_INCOME_TOTAL,
            # ── Outputs (résultats du modèle) ─────────────────────────────────
            "prediction":             prediction,
            "probability_default":    round(proba, 4),
            "risk_label":             risk_label,
        })

        return response

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(data: List[CreditRequest]):
    if len(data) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 requêtes.")
    return [predict(item) for item in data]


# ─── ENDPOINT DE MONITORING ───────────────────────────────────────────────────
@app.get("/logs/stats")
def logs_stats():
    """
    Retourne des statistiques basiques sur les logs de production.
    Utile pour un premier diagnostic sans ouvrir le fichier JSONL.
    """
    if not PREDICTIONS_LOG.exists():
        return {"total_predictions": 0, "log_file": str(PREDICTIONS_LOG)}

    lines = PREDICTIONS_LOG.read_text().strip().splitlines()
    if not lines:
        return {"total_predictions": 0}

    entries    = [json.loads(l) for l in lines]
    probas     = [e["probability_default"] for e in entries]
    latencies  = [e["latency_ms"] for e in entries]
    high_risk  = sum(1 for e in entries if e["prediction"] == 1)

    return {
        "total_predictions":    len(entries),
        "high_risk_count":      high_risk,
        "high_risk_rate":       round(high_risk / len(entries), 4),
        "avg_probability":      round(sum(probas) / len(probas), 4),
        "avg_latency_ms":       round(sum(latencies) / len(latencies), 2),
        "max_latency_ms":       max(latencies),
        "first_log":            entries[0]["timestamp"],
        "last_log":             entries[-1]["timestamp"],
    }