import os
import time
import logging
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajustement du chemin pour pointer vers /models à la racine depuis /src
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

OPTIMAL_THRESHOLD = float(os.getenv("OPTIMAL_THRESHOLD", 0.48))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# --- MODULE UTILITAIRES : Moteur de Scoring ---
class ScoringEngine:
    def __init__(self, models_path: Path, threshold: float):
        self.threshold = threshold
        self.ready = False
        try:
            self.preprocessor = joblib.load(models_path / "preprocessor.pkl")
            self.features = joblib.load(models_path / "selected_features.pkl")
            self.model = joblib.load(models_path / "model.joblib")
            self.ready = True
            logger.info("✅ Modèles chargés avec succès.")
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèles : {e}")

    def predict(self, data_dict: dict):
        if not self.ready:
            raise RuntimeError("Le modèle n'est pas chargé.")
        
        # 1. Création du DataFrame initial
        df = pd.DataFrame([data_dict])
        
        # 2. Complétion des colonnes manquantes (pour test_predict_all_none_fields)
        for col in self.features:
            if col not in df.columns:
                df[col] = 0.0
        
        # 3. Alignement strict sur l'ordre des colonnes d'entraînement
        df_ordered = df[self.features]
        
        # 4. Transformation
        X_transformed = self.preprocessor.transform(df_ordered)
        
        # 5. RECONSTRUCTION DU DATAFRAME AVEC NOMS (Fix UserWarning)
        # On passe un DataFrame nommé au modèle au lieu d'un array NumPy
        X_final = pd.DataFrame(X_transformed, columns=self.features)
        
        # 6. Inférence
        proba = float(self.model.predict_proba(X_final)[0][1])
        return proba, int(proba >= self.threshold)

    def get_risk_label(self, proba: float):
        if proba < 0.2: return "Très faible risque"
        if proba < self.threshold: return "Risque modéré"
        if proba < 0.7: return "Risque élevé"
        return "Risque très élevé"

engine = ScoringEngine(MODELS_DIR, OPTIMAL_THRESHOLD)

# --- SCHÉMAS DE DONNÉES ---
class CreditRequest(BaseModel):
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0, le=1)
    AMT_CREDIT: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None
    DAYS_EMPLOYED: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None
    DAYS_BIRTH: Optional[float] = None
    DAYS_LAST_PHONE_CHANGE: Optional[float] = None
    AMT_INCOME_TOTAL: Optional[float] = None

# --- API FASTAPI ---
app = FastAPI(title="Credit Scoring API")

@app.get("/")
def read_root():
    """Répond au test_root_endpoint"""
    return {"message": "Bienvenue sur l'API de Scoring", "status": "online"}

@app.get("/health")
def health():
    """Répond au test_health_endpoint"""
    return {
        "status": "ok" if engine.ready else "error",
        "model_loaded": engine.ready
    }

@app.post("/predict")
async def predict(data: CreditRequest):
    """Répond aux tests unitaires de prédiction"""
    if not engine.ready:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        raw_data = data.model_dump()
        # Gestion des None pour la robustesse
        clean_data = {k: (v if v is not None else 0.0) for k, v in raw_data.items()}
        
        proba, pred = engine.predict(clean_data)
        
        return {
            "prediction": pred,
            "probability_default": round(proba, 4),
            "risk_label": engine.get_risk_label(proba)
        }
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne de prédiction")

@app.post("/predict/batch")
async def predict_batch(data: List[CreditRequest]):
    """Répond aux tests batch"""
    if len(data) > 100:
        raise HTTPException(status_code=400, detail="Batch trop volumineux (max 100)")
    
    results = []
    for item in data:
        results.append(await predict(item))
    return results

# --- INTERFACE GRADIO ---
def predict_ui(*args):
    if not engine.ready: 
        return "Erreur : Modèle non chargé"
    keys = list(CreditRequest.model_fields.keys())
    data_dict = {k: v for k, v in zip(keys, args)}
    proba, _ = engine.predict(data_dict)
    return engine.get_risk_label(proba)

# Utilisation d'une compréhension pour éviter DuplicateBlockError
ui = gr.Interface(
    fn=predict_ui, 
    inputs=[gr.Number(label=k) for k in CreditRequest.model_fields.keys()], 
    outputs="text"
)
app = gr.mount_gradio_app(app, ui, path="/ui")