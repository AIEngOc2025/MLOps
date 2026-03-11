"""
Credit Scoring API — FastAPI
Modèle : LightGBM | 10 features | Seuil métier : 0.48
"""
import traceback

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

# ─── CHARGEMENT DES ARTEFACTS ─────────────────────────────────────────────────
try:
    preprocessor      = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
    SELECTED_FEATURES = joblib.load(os.path.join(MODELS_DIR, "selected_features.pkl"))
    # Ordre exact : EXT_SOURCE_1, EXT_SOURCE_3, EXT_SOURCE_2, AMT_CREDIT,
    #               AMT_ANNUITY, DAYS_EMPLOYED, AMT_GOODS_PRICE,
    #               DAYS_BIRTH, DAYS_LAST_PHONE_CHANGE, AMT_INCOME_TOTAL
    model_path = os.path.join(MODELS_DIR, "model.joblib")
    model      = joblib.load(model_path) if os.path.exists(model_path) else None
except Exception as e:
    print(f"⚠️  Erreur chargement artefacts : {e}")
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
    try:
        # Construire le DataFrame dans l'ordre EXACT du preprocessor
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

        return {
            "prediction":          prediction,
            "probability_default": round(proba, 4),
            "risk_label":          risk_label,
            "threshold_used":      OPTIMAL_THRESHOLD,
            "model_available":     model is not None,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(data: List[CreditRequest]):
    if len(data) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 requêtes.")
    return [predict(item) for item in data]


