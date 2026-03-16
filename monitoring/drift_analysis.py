"""
drift_analysis.py — Analyse de drift avec Evidently AI
═══════════════════════════════════════════════════════════════════════════════
Télécharge les logs de production depuis Hugging Face Dataset et analyse :
  - Data drift    : les features d'entrée ont-elles changé de distribution ?
  - Model drift   : les prédictions/probabilités ont-elles dérivé ?
  - Anomalies ops : latence anormale, taux de risque élevé

Sources de logs (par ordre de priorité) :
  1. Hugging Face Dataset (dataChaser/credit-score-logs) — production réelle
  2. Fichier local logs/predictions.jsonl                — fallback / PoC

Usage :
  python drift_analysis.py                             # logs HF + split temporel
  python drift_analysis.py --source local              # forcer logs locaux
  python drift_analysis.py --reference data/train.csv  # avec données réelles
  python drift_analysis.py --output reports/           # dossier de sortie custom

Prérequis :
  uv run python drift_analysis.py
  (huggingface_hub et evidently installés via uv)
"""

import argparse
import json
import logging
import os
from pathlib import Path
from datetime import datetime                    # ✅ Fix 1 : import tempfile supprimé

import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR     = Path(__file__).parent.parent
LOGS_DIR     = BASE_DIR / "logs"
REPORTS_DIR  = BASE_DIR / "reports"

# Fichier local de fallback
PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"

# HF Dataset
HF_TOKEN      = os.getenv("HF_TOKEN")
HF_USERNAME   = os.getenv("HF_USERNAME", "dataChaser")
HF_DATASET_ID = f"{HF_USERNAME}/credit-score-logs"
HF_FILE       = "predictions.jsonl"

# Features du modèle
FEATURES = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "AMT_CREDIT", "AMT_ANNUITY", "DAYS_EMPLOYED",
    "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_LAST_PHONE_CHANGE",
    "AMT_INCOME_TOTAL",
]
PREDICTION_COLS = ["prediction", "probability_default"]


# ═══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

def load_logs_from_hf() -> pd.DataFrame:
    """
    Télécharge predictions.jsonl depuis HF Dataset et retourne un DataFrame.

    ✅ Fix 2 : vérification explicite du token si dataset privé.
    Le token est optionnel pour les datasets publics.
    """
    from huggingface_hub import hf_hub_download

    # Avertissement si token absent — peut bloquer pour les datasets privés
    if not HF_TOKEN:
        logger.warning(
            "⚠️  HF_TOKEN absent — fonctionnel uniquement si le dataset est public."
        )

    logger.info(f"📥 Téléchargement des logs depuis {HF_DATASET_ID}...")

    local_path = hf_hub_download(
        repo_id=HF_DATASET_ID,
        filename=HF_FILE,
        repo_type="dataset",
        token=HF_TOKEN or None,    # None explicite si token absent
    )

    return _parse_jsonl(Path(local_path))


def load_logs_from_local(log_path: Path) -> pd.DataFrame:
    """Charge le fichier JSONL local."""
    if not log_path.exists():
        raise FileNotFoundError(f"Fichier de logs introuvable : {log_path}")
    return _parse_jsonl(log_path)


def _parse_jsonl(path: Path) -> pd.DataFrame:
    """Parse un fichier JSONL en DataFrame."""
    lines = path.read_text().strip().splitlines()
    if not lines:
        raise ValueError(f"Le fichier de logs est vide : {path}")
    df = pd.DataFrame([json.loads(l) for l in lines])
    logger.info(f"✅ {len(df)} logs chargés depuis {path}")
    return df


def load_production_logs(source: str) -> pd.DataFrame:
    """
    Charge les logs selon la source choisie.
    source = "hf"    → télécharge depuis HF Dataset
    source = "local" → lit le fichier local
    source = "auto"  → essaie HF, fallback local si échec
    """
    if source == "hf":
        return load_logs_from_hf()

    if source == "local":
        return load_logs_from_local(PREDICTIONS_LOG)

    # Mode auto — HF en priorité, fallback local
    try:
        return load_logs_from_hf()
    except Exception as e:
        logger.warning(f"⚠️  HF indisponible ({e}) — fallback local")
        return load_logs_from_local(PREDICTIONS_LOG)


def load_reference_data(
    reference_path: Path | None,
    production_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:         # ✅ Fix 3 : type de retour précis
    """
    Retourne (reference_df, current_df).

    Si données de référence fournies → reference = fichier CSV, current = tous les logs
    Sinon → split temporel : première moitié = référence, deuxième = courante

    ✅ Fix 5 : vérification que les deux moitiés ont assez de lignes.
    """
    cols = FEATURES + PREDICTION_COLS

    if reference_path and reference_path.exists():
        ref_df = pd.read_csv(reference_path)[cols]
        cur_df = production_df[cols]
        logger.info(f"✅ Référence : {len(ref_df)} lignes depuis {reference_path}")
        return ref_df, cur_df

    # Fallback : split temporel
    logger.warning(
        "⚠️  Pas de données de référence. "
        "Split temporel : première moitié = référence, deuxième = courante."
    )

    total = len(production_df)
    mid   = total // 2

    # ✅ Fix 5 : vérification des deux moitiés
    if mid < 10:
        raise ValueError(
            f"Pas assez de logs pour la référence : {mid} lignes (minimum 10 requis). "
            f"Total disponible : {total} logs."
        )
    if (total - mid) < 10:
        raise ValueError(
            f"Pas assez de logs pour la période courante : {total - mid} lignes "
            f"(minimum 10 requis). Total disponible : {total} logs."
        )

    ref_df = production_df.iloc[:mid][cols]
    cur_df = production_df.iloc[mid:][cols]

    logger.info(f"✅ Référence : {len(ref_df)} lignes | Courante : {len(cur_df)} lignes")
    return ref_df, cur_df


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSE OPÉRATIONNELLE
# ═══════════════════════════════════════════════════════════════════════════════

def run_operational_analysis(df: pd.DataFrame) -> dict:
    """Détecte les anomalies opérationnelles : latence, taux de risque, valeurs manquantes."""
    total          = len(df)
    high_risk      = int(df["prediction"].sum())
    avg_latency    = df["latency_ms"].mean()
    p95_latency    = df["latency_ms"].quantile(0.95)
    high_risk_rate = high_risk / total

    alerts = []
    if avg_latency > 500:
        alerts.append(f"⚠️  Latence moyenne élevée : {avg_latency:.1f}ms (seuil 500ms)")
    if p95_latency > 1000:
        alerts.append(f"⚠️  Latence P95 critique : {p95_latency:.1f}ms (seuil 1000ms)")
    if high_risk_rate > 0.5:
        alerts.append(f"⚠️  Taux risque élevé anormal : {high_risk_rate:.1%} (seuil 50%)")

    missing = {
        col: int(df[col].isna().sum())
        for col in FEATURES
        if df[col].isna().sum() > 0
    }
    if missing:
        alerts.append(f"⚠️  Valeurs manquantes : {missing}")

    print("\n" + "═" * 60)
    print("📊 RÉSUMÉ OPÉRATIONNEL")
    print("═" * 60)
    print(f"  Total prédictions   : {total}")
    print(f"  Taux risque élevé   : {high_risk_rate:.1%}")
    print(f"  Probabilité moyenne : {df['probability_default'].mean():.4f}")
    print(f"  Latence moyenne     : {avg_latency:.1f}ms")
    print(f"  Latence P95         : {p95_latency:.1f}ms")
    if alerts:
        print("\n🚨 ALERTES :")
        for a in alerts:
            print(f"  {a}")
    else:
        print("\n✅ Aucune anomalie détectée")
    print("═" * 60 + "\n")

    return {"total": total, "high_risk_rate": high_risk_rate, "alerts": alerts}


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSE DE DRIFT
# ═══════════════════════════════════════════════════════════════════════════════

def run_drift_analysis(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Génère le rapport Evidently HTML de détection de drift."""
    logger.info("🔍 Calcul du rapport de drift...")

    drift_report = Report(metrics=[                 # ✅ Fix 1 : variable renommée
        DatasetDriftMetric(),                       #    drift_report au lieu de report
        DataDriftPreset(),                          #    évite le conflit avec la classe Report
        DataQualityPreset(),
        ColumnDriftMetric(column_name="probability_default"),
        ColumnDriftMetric(column_name="prediction"),
    ])

    drift_report.run(reference_data=reference_df, current_data=current_df)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ✅ Fix 4 : format timestamp lisible et standard
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"drift_report_{timestamp}.html"

    drift_report.save_html(str(report_path))
    logger.info(f"✅ Rapport sauvegardé : {report_path}")

    return report_path


# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyse de drift — Credit Scoring API")
    parser.add_argument(
        "--source", choices=["hf", "local", "auto"], default="auto",
        help="Source des logs : hf | local | auto (défaut)"
    )
    parser.add_argument(
        "--reference", type=Path, default=None,
        help="Données de référence CSV (optionnel)"
    )
    parser.add_argument(
        "--output", type=Path, default=REPORTS_DIR,
        help="Dossier de sortie pour les rapports HTML"
    )
    args = parser.parse_args()

    # 1. Charger les logs
    production_df = load_production_logs(args.source)

    # 2. Analyse opérationnelle
    run_operational_analysis(production_df)

    # 3. Préparer référence et données courantes
    reference_df, current_df = load_reference_data(args.reference, production_df)

    # 4. Rapport Evidently
    report_path = run_drift_analysis(reference_df, current_df, args.output)
    print(f"📄 Rapport HTML : {report_path}")
    print(f"   → open {report_path}")


if __name__ == "__main__":
    main()