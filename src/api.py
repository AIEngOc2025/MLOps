"""
Credit Scoring API — FastAPI + Gradio
Modèle : LightGBM | 10 features | Seuil métier : 0.48

Optimisations ETAPE4 :
  1. Pipeline numpy pur (V3)    : +56.9% sur preprocessing
  2. ONNX Runtime               : gain supplémentaire sur l'inférence
     → Chargé si models/model.onnx existe
     → Fallback automatique vers LightGBM si ONNX indisponible

Logique de chargement :
  - Si model.onnx existe  → ONNX Runtime (optimal)
  - Sinon                 → LightGBM sklearn (fallback)
  - /health indique quel moteur est actif

Routes :
  GET  /          → message de bienvenue
  GET  /health    → statut + moteur d'inférence actif
  POST /predict   → prédiction unitaire
  POST /predict/batch → prédiction batch (max 100)
  GET  /logs/stats    → statistiques des logs
  POST /logs/flush    → force le push des logs vers HF Dataset
  GET  /ui        → interface Gradio
"""

import time
import json
import traceback
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os
import requests

app = FastAPI(
    title="Credit Scoring API",
    description="API de scoring crédit basée sur LightGBM (Home Credit dataset)",
    version="2.1.0",
)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR        = os.path.join(BASE_DIR, "models/")
OPTIMAL_THRESHOLD = float(os.getenv("OPTIMAL_THRESHOLD", 0.48))

# ─── CONFIGURATION LOGGING ────────────────────────────────────────────────────
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", 30))
HF_TOKEN      = os.getenv("HF_TOKEN")
HF_USERNAME   = os.getenv("HF_USERNAME")
HF_DATASET_ID = f"{HF_USERNAME}/credit-score-logs" if HF_USERNAME else None

LOGS_DIR        = Path(BASE_DIR) / "logs"
LOGS_DIR.mkdir(exist_ok=True)
PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── BUFFER DE LOGS EN MÉMOIRE ────────────────────────────────────────────────
_log_buffer: list = []
_buffer_lock = Lock()


def flush_logs_to_hf(entries: list) -> bool:
    """Push le buffer de logs vers Hugging Face Dataset."""
    if not HF_TOKEN or not HF_DATASET_ID:
        logger.warning("⚠️  HF_TOKEN ou HF_USERNAME manquant — fallback local")
        return False
    try:
        from huggingface_hub import HfApi, hf_hub_download
        import tempfile

        api = HfApi(token=HF_TOKEN)

        existing_lines = []
        try:
            local_path = hf_hub_download(
                repo_id=HF_DATASET_ID,
                filename="predictions.jsonl",
                repo_type="dataset",
                token=HF_TOKEN,
            )
            with open(local_path, "r") as f:
                existing_lines = f.readlines()
        except Exception:
            logger.info("📄 Création du fichier de logs sur HF Dataset")

        new_lines = [json.dumps(e, ensure_ascii=False) + "\n" for e in entries]
        all_lines = existing_lines + new_lines

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp.writelines(all_lines)
            tmp_path = tmp.name

        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="predictions.jsonl",
            repo_id=HF_DATASET_ID,
            repo_type="dataset",
            commit_message=f"logs: +{len(entries)} prédictions",
        )

        os.unlink(tmp_path)
        logger.info(f"✅ {len(entries)} logs pushés vers {HF_DATASET_ID}")
        return True

    except Exception as e:
        logger.error(f"❌ Erreur push HF : {e}")
        return False


def save_logs_locally(entries: list) -> None:
    """Fallback — sauvegarde les logs localement."""
    try:
        with open(PREDICTIONS_LOG, "a") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"💾 {len(entries)} logs sauvegardés localement")
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde locale : {e}")


def log_prediction(entry: dict) -> None:
    """Ajoute une entrée au buffer. Flush en thread daemon si BATCH_SIZE atteint."""
    global _log_buffer
    with _buffer_lock:
        _log_buffer.append(entry)
        if len(_log_buffer) >= BATCH_SIZE:
            batch       = _log_buffer.copy()
            _log_buffer = []
            def _push():
                success = flush_logs_to_hf(batch)
                if not success:
                    save_logs_locally(batch)
            Thread(target=_push, daemon=True).start()


# ─── CHARGEMENT DES ARTEFACTS ─────────────────────────────────────────────────
# Chargement UNE SEULE FOIS au démarrage
# Extraction des paramètres numpy pour le pipeline optimisé (V3)
try:
    preprocessor      = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
    SELECTED_FEATURES = joblib.load(os.path.join(MODELS_DIR, "selected_features.pkl"))
    model_path        = os.path.join(MODELS_DIR, "model.joblib")
    model             = joblib.load(model_path) if os.path.exists(model_path) else None

    # ── Paramètres numpy pour preprocessing optimisé ──────────────────────────
    IMPUTE_VALUES = preprocessor.named_steps["imputer"].statistics_
    SCALE_MEAN    = preprocessor.named_steps["scaler"].mean_
    SCALE_STD     = preprocessor.named_steps["scaler"].scale_

    logger.info("✅ Artefacts LightGBM chargés")
except Exception as e:
    logger.error(f"⚠️  Erreur chargement artefacts : {e}")
    model, preprocessor, SELECTED_FEATURES = None, None, []
    IMPUTE_VALUES = SCALE_MEAN = SCALE_STD = None


# ─── CHARGEMENT ONNX RUNTIME (optionnel) ──────────────────────────────────────
# Si models/model.onnx existe → ONNX Runtime est utilisé pour l'inférence
# Sinon → fallback automatique vers LightGBM sklearn
onnx_session  = None
INFERENCE_ENGINE = "lightgbm"  # sera mis à jour si ONNX chargé

onnx_model_path = os.path.join(MODELS_DIR, "model.onnx")
if os.path.exists(onnx_model_path):
    try:
        import onnxruntime as rt
        onnx_session     = rt.InferenceSession(onnx_model_path)
        INFERENCE_ENGINE = "onnx_runtime"
        logger.info(f"✅ ONNX Runtime chargé — moteur actif : {INFERENCE_ENGINE}")
    except Exception as e:
        logger.warning(f"⚠️  ONNX Runtime indisponible ({e}) — fallback LightGBM")
        onnx_session     = None
        INFERENCE_ENGINE = "lightgbm_fallback"
else:
    logger.info("ℹ️  models/model.onnx absent — LightGBM actif")


# ─── PIPELINE NUMPY OPTIMISÉ ──────────────────────────────────────────────────
def numpy_preprocess(data: "CreditRequest") -> np.ndarray:
    """
    Reproduit Pipeline(SimpleImputer → StandardScaler) en numpy pur.
    Gain mesuré : +56.9% vs sklearn au runtime.
    """
    values = np.array(
        [[getattr(data, f) if getattr(data, f) is not None else np.nan
          for f in SELECTED_FEATURES]],
        dtype=np.float64
    )
    X = np.where(np.isnan(values), IMPUTE_VALUES, values)  # imputation
    X = (X - SCALE_MEAN) / SCALE_STD                       # scaling
    return X


def run_inference(X: np.ndarray) -> float:
    """
    Lance l'inférence selon le moteur disponible.
    Priorité : ONNX Runtime > LightGBM sklearn
    """
    if onnx_session is not None:
        # ── ONNX Runtime — float32 requis ────────────────────────────────────
        input_name  = onnx_session.get_inputs()[0].name
        onnx_output = onnx_session.run(None, {input_name: X.astype(np.float32)})
        return float(onnx_output[1][0][1])   # proba classe 1

    elif model is not None:
        # ── LightGBM sklearn — DataFrame avec noms pour éviter le warning ────
        X_df = pd.DataFrame(X, columns=SELECTED_FEATURES)
        return float(model.predict_proba(X_df)[0][1])

    else:
        return 0.5  # fallback si aucun modèle disponible


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


# ─── ENDPOINTS FASTAPI ────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Credit Scoring API v2.1 — /docs pour la documentation, /ui pour l'interface.",
        "inference_engine": INFERENCE_ENGINE,
    }


@app.get("/health")
def health():
    return {
        "status":            "ok",
        "model_loaded":      model is not None or onnx_session is not None,
        "features_count":    len(SELECTED_FEATURES),
        "threshold":         OPTIMAL_THRESHOLD,
        "buffer_size":       len(_log_buffer),
        "batch_size":        BATCH_SIZE,
        "hf_dataset":        HF_DATASET_ID,
        "pipeline":          "numpy_optimized_v3",
        "inference_engine":  INFERENCE_ENGINE,  # ← moteur actif visible ici
    }


@app.post("/predict", response_model=CreditResponse)
def predict(data: CreditRequest):
    start_time = time.time()
    try:
        if IMPUTE_VALUES is not None:
            # ── Pipeline optimisé V3 + inférence selon moteur disponible ──────
            X     = numpy_preprocess(data)
            proba = run_inference(X)
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
            "model_available":     model is not None or onnx_session is not None,
        }

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


@app.get("/logs/stats")
def logs_stats():
    stats = {
        "buffer_pending":   len(_log_buffer),
        "batch_size":       BATCH_SIZE,
        "hf_dataset":       HF_DATASET_ID,
        "inference_engine": INFERENCE_ENGINE,
    }
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
    with _buffer_lock:
        if not _log_buffer:
            return {"message": "Buffer vide — rien à pusher"}
        batch = _log_buffer.copy()
        _log_buffer.clear()
    success = flush_logs_to_hf(batch)
    if not success:
        save_logs_locally(batch)
        return {"message": f"{len(batch)} logs sauvegardés localement"}
    return {"message": f"✅ {len(batch)} logs pushés vers {HF_DATASET_ID}"}


# ─── INTERFACE GRADIO ─────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://127.0.0.1:7860/predict")


def predict_gradio(ext1, ext2, ext3, credit, annuity, employed, goods, birth, phone, income):
    """Fonction appelée par Gradio — appelle /predict en interne."""
    payload = {
        "EXT_SOURCE_1": ext1,  "EXT_SOURCE_2": ext2,  "EXT_SOURCE_3": ext3,
        "AMT_CREDIT": credit,  "AMT_ANNUITY": annuity, "DAYS_EMPLOYED": employed,
        "AMT_GOODS_PRICE": goods, "DAYS_BIRTH": birth,
        "DAYS_LAST_PHONE_CHANGE": phone, "AMT_INCOME_TOTAL": income,
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        if response.status_code != 200:
            return f"⚠️ Erreur API ({response.status_code}) : {response.json().get('detail', 'Erreur inconnue')}"
        res       = response.json()
        status    = "✅ ACCORDÉ" if res["prediction"] == 0 else "❌ REFUSÉ"
        proba     = f"Probabilité de défaut : {res['probability_default']:.2%}"
        risk      = f"Niveau de risque : {res['risk_label']}"
        threshold = f"Seuil de décision : {res['threshold_used']}"
        return f"{status}\n\n{proba}\n{risk}\n{threshold}"
    except requests.exceptions.ConnectionError:
        return "❌ Impossible de joindre l'API."
    except requests.exceptions.Timeout:
        return "⏱️ Timeout (5s)."
    except Exception as e:
        return f"Erreur inattendue : {e}"


gradio_app = gr.Interface(
    fn=predict_gradio,
    inputs=[
        gr.Slider(0, 1, step=0.01, value=0.5, label="EXT_SOURCE_1"),
        gr.Slider(0, 1, step=0.01, value=0.5, label="EXT_SOURCE_2"),
        gr.Slider(0, 1, step=0.01, value=0.5, label="EXT_SOURCE_3"),
        gr.Number(label="Montant Crédit (AMT_CREDIT)",          value=100000, minimum=0),
        gr.Number(label="Annuité (AMT_ANNUITY)",                value=5000,   minimum=0),
        gr.Number(label="Jours employés (négatif)",             value=-1000,  maximum=0),
        gr.Number(label="Prix des biens (AMT_GOODS_PRICE)",     value=80000,  minimum=0),
        gr.Number(label="Âge en jours (négatif)",               value=-15000, maximum=0),
        gr.Number(label="Dernier changement tél. (jours nég.)", value=-100,   maximum=0),
        gr.Number(label="Revenu Total (AMT_INCOME_TOTAL)",      value=50000,  minimum=0),
    ],
    outputs=gr.Textbox(label="Résultat de la prédiction", lines=5),
    title="🏦 Simulateur de Crédit — Scoring Client",
    description=(
        "Entrez les caractéristiques du client pour obtenir une prédiction de risque de défaut. "
        "Les champs en jours doivent être **négatifs** (ex: -15000 jours = ~41 ans)."
    ),
    flagging_mode="never",
)

# ─── MONTAGE GRADIO SUR FASTAPI ───────────────────────────────────────────────
app = gr.mount_gradio_app(app, gradio_app, path="/ui")
