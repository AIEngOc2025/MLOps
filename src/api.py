"""
Credit Scoring API — FastAPI
Modèle : LightGBM | 10 features | Seuil métier : 0.48

Logging structuré :
  - Chaque appel /predict est loggé en mémoire
  - Toutes les BATCH_SIZE requêtes → push vers Hugging Face Dataset
  - Fallback local : logs/predictions.jsonl si HF est indisponible
"""

import time
import json
import traceback
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

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

# ─── CONFIGURATION LOGGING ────────────────────────────────────────────────────
# Nombre de requêtes avant un push vers HF Dataset
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 30))

# Identifiants HF — injectés via variables d'environnement (secrets)
HF_TOKEN      = os.getenv("HF_TOKEN")
HF_USERNAME   = os.getenv("HF_USERNAME")
HF_DATASET_ID = f"{HF_USERNAME}/credit-score-logs" if HF_USERNAME else None

# Fallback local si HF est indisponible
LOGS_DIR        = Path(BASE_DIR) / "logs"
LOGS_DIR.mkdir(exist_ok=True)
PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"

# Logger Python standard
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── BUFFER DE LOGS EN MÉMOIRE ────────────────────────────────────────────────
# Liste partagée entre les requêtes — protégée par un Lock thread-safe
_log_buffer: list = []
_buffer_lock = Lock()


def flush_logs_to_hf(entries: list) -> bool:
    """
    Pousse le buffer de logs vers Hugging Face Dataset.
    Retourne True si succès, False si échec (fallback local).

    Stratégie :
      1. Télécharger le fichier JSONL existant sur HF (si présent)
      2. Ajouter les nouvelles entrées
      3. Repousser le fichier mis à jour
    """
    if not HF_TOKEN or not HF_DATASET_ID:
        logger.warning("⚠️  HF_TOKEN ou HF_USERNAME manquant — fallback local")
        return False

    try:
        from huggingface_hub import HfApi
        import tempfile

        api       = HfApi(token=HF_TOKEN)
        hf_file   = "predictions.jsonl"

        # ── Télécharger les logs existants sur HF ──────────────────────────
        existing_lines = []
        try:
            local_path = api.hf_hub_download(
                repo_id=HF_DATASET_ID,
                filename=hf_file,
                repo_type="dataset",
            )
            with open(local_path, "r") as f:
                existing_lines = f.readlines()
        except Exception:
            # Fichier inexistant sur HF — première fois
            logger.info("📄 Création du fichier de logs sur HF Dataset")

        # ── Ajouter les nouvelles entrées ──────────────────────────────────
        new_lines = [json.dumps(e, ensure_ascii=False) + "\n" for e in entries]
        all_lines = existing_lines + new_lines

        # ── Réécrire le fichier complet et pousser ─────────────────────────
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp.writelines(all_lines)
            tmp_path = tmp.name

        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=hf_file,
            repo_id=HF_DATASET_ID,
            repo_type="dataset",
            commit_message=f"logs: +{len(entries)} prédictions",
        )

        logger.info(f"✅ {len(entries)} logs pushés vers {HF_DATASET_ID}")
        return True

    except Exception as e:
        logger.error(f"❌ Erreur push HF : {e}")
        return False


def save_logs_locally(entries: list) -> None:
    """Fallback — sauvegarde les logs localement si HF est indisponible."""
    try:
        with open(PREDICTIONS_LOG, "a") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"💾 {len(entries)} logs sauvegardés localement")
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde locale : {e}")


def log_prediction(entry: dict) -> None:
    """
    Ajoute une entrée au buffer en mémoire.
    Si le buffer atteint BATCH_SIZE → flush vers HF Dataset.
    Thread-safe via Lock.
    """
    global _log_buffer

    with _buffer_lock:
        _log_buffer.append(entry)

        # ── Push si batch complet ──────────────────────────────────────────
        if len(_log_buffer) >= BATCH_SIZE:
            batch = _log_buffer.copy()
            _log_buffer = []   # vider le buffer avant le push

            # Push HF — fallback local si échec
            success = flush_logs_to_hf(batch)
            if not success:
                save_logs_locally(batch)


# ─── CHARGEMENT DES ARTEFACTS ─────────────────────────────────────────────────
# Chargement UNE SEULE FOIS au démarrage
try:
    preprocessor      = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
    SELECTED_FEATURES = joblib.load(os.path.join(MODELS_DIR, "selected_features.pkl"))
    model_path        = os.path.join(MODELS_DIR, "model.joblib")
    model             = joblib.load(model_path) if os.path.exists(model_path) else None
    logger.info("✅ Artefacts chargés avec succès")
except Exception as e:
    logger.error(f"⚠️  Erreur chargement artefacts : {e}")
    model, preprocessor, SELECTED_FEATURES = None, None, []


# ─── SCHÉMAS ──────────────────────────────────────────────────────────────────
class CreditRequest(BaseModel):
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
        "status":           "ok",
        "model_loaded":     model is not None,
        "features_count":   len(SELECTED_FEATURES),
        "threshold":        OPTIMAL_THRESHOLD,
        "buffer_size":      len(_log_buffer),   # logs en attente de push
        "batch_size":       BATCH_SIZE,
        "hf_dataset":       HF_DATASET_ID,
    }


@app.post("/predict", response_model=CreditResponse)
def predict(data: CreditRequest):
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

        response = {
            "prediction":          prediction,
            "probability_default": round(proba, 4),
            "risk_label":          risk_label,
            "threshold_used":      OPTIMAL_THRESHOLD,
            "model_available":     model is not None,
        }

        # ── Logging APRÈS la prédiction ───────────────────────────────────────
        latency_ms = round((time.time() - start_time) * 1000, 2)

        log_prediction({
            "timestamp":              datetime.now(timezone.utc).isoformat(),
            "latency_ms":             latency_ms,
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
    """Statistiques basiques sur les logs — buffer en mémoire + logs locaux."""
    stats = {
        "buffer_pending":  len(_log_buffer),
        "batch_size":      BATCH_SIZE,
        "hf_dataset":      HF_DATASET_ID,
        "local_log_file":  str(PREDICTIONS_LOG),
    }

    # Stats sur les logs locaux si présents
    if PREDICTIONS_LOG.exists():
        lines = PREDICTIONS_LOG.read_text().strip().splitlines()
        if lines:
            entries   = [json.loads(l) for l in lines]
            probas    = [e["probability_default"] for e in entries]
            latencies = [e["latency_ms"] for e in entries]
            high_risk = sum(1 for e in entries if e["prediction"] == 1)
            stats.update({
                "local_total":       len(entries),
                "local_high_risk":   round(high_risk / len(entries), 4),
                "local_avg_proba":   round(sum(probas) / len(probas), 4),
                "local_avg_latency": round(sum(latencies) / len(latencies), 2),
            })

    return stats


@app.post("/logs/flush")
def flush_logs():
    """
    Force le push immédiat du buffer vers HF Dataset.
    Utile pour vider le buffer avant un redémarrage du conteneur.
    """
    with _buffer_lock:
        if not _log_buffer:
            return {"message": "Buffer vide — rien à pusher"}
        batch = _log_buffer.copy()
        _log_buffer.clear()

    success = flush_logs_to_hf(batch)
    if not success:
        save_logs_locally(batch)
        return {"message": f"{len(batch)} logs sauvegardés localement (HF indisponible)"}

    return {"message": f"✅ {len(batch)} logs pushés vers {HF_DATASET_ID}"}
