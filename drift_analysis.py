"""
drift_analysis.py — Analyse de drift avec Evidently AI
═══════════════════════════════════════════════════════════════════════════════
Lit les logs de production (predictions.jsonl) et les compare aux données
d'entraînement pour détecter :
  - Data drift    : les features d'entrée ont-elles changé de distribution ?
  - Model drift   : les prédictions/probabilités ont-elles dérivé ?
  - Anomalies ops : latence anormale, taux d'erreur élevé

Usage :
  python drift_analysis.py                         # analyse complète
  python drift_analysis.py --reference data/train.csv  # avec données réelles
  python drift_analysis.py --output reports/        # dossier de sortie custom

Prérequis :
  pip install evidently pandas
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
from evidently.report import Report


from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Chemins par défaut
BASE_DIR     = Path(__file__).parent
LOGS_DIR     = BASE_DIR / "logs"
REPORTS_DIR  = BASE_DIR / "reports"
PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"

# Features du modèle — ordre exact
FEATURES = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "AMT_CREDIT", "AMT_ANNUITY", "DAYS_EMPLOYED",
    "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_LAST_PHONE_CHANGE",
    "AMT_INCOME_TOTAL",
]

# Colonnes de prédiction à surveiller pour le model drift
PREDICTION_COLS = ["prediction", "probability_default"]


# ═══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

def load_production_logs(log_path: Path) -> pd.DataFrame:
    """
    Charge le fichier JSONL de logs de production en DataFrame.
    Chaque ligne du fichier = un appel /predict.
    """
    if not log_path.exists():
        raise FileNotFoundError(f"Fichier de logs introuvable : {log_path}")

    lines = log_path.read_text().strip().splitlines()
    if not lines:
        raise ValueError("Le fichier de logs est vide.")

    df = pd.DataFrame([json.loads(l) for l in lines])
    logger.info(f"✅ {len(df)} logs chargés depuis {log_path}")
    return df


def load_reference_data(reference_path: Path | None, production_df: pd.DataFrame) -> pd.DataFrame:
    """
    Charge les données de référence (données d'entraînement).

    Si aucun fichier de référence n'est fourni :
    → On utilise la première moitié des logs comme référence
    → La deuxième moitié comme données courantes (simulation de drift)

    C'est l'approche recommandée pour un PoC sans données d'entraînement.
    """
    if reference_path and reference_path.exists():
        df = pd.read_csv(reference_path)
        logger.info(f"✅ Données de référence chargées : {len(df)} lignes")
        return df[FEATURES + PREDICTION_COLS]

    # Fallback : split temporel des logs de production
    logger.warning(
        "⚠️  Pas de données de référence fournies. "
        "Utilisation de la première moitié des logs comme référence."
    )
    mid = len(production_df) // 2
    if mid < 10:
        raise ValueError(
            "Pas assez de logs pour une analyse de drift (minimum 20 appels requis)."
        )
    return production_df.iloc[:mid][FEATURES + PREDICTION_COLS]


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSE DE DRIFT
# ═══════════════════════════════════════════════════════════════════════════════

def run_drift_analysis(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Lance le rapport Evidently de détection de drift.

    Contient :
      - DataDriftPreset     : drift sur toutes les features + prédictions
      - DataQualityPreset   : valeurs manquantes, distributions
      - ColumnDriftMetric   : drift par feature individuelle
      - DatasetDriftMetric  : drift global du dataset
    """
    logger.info("🔍 Calcul du rapport de drift...")

    report = Report(metrics=[
        # ── Drift global ──────────────────────────────────────────────────────
        DatasetDriftMetric(),

        # ── Drift par feature ─────────────────────────────────────────────────
        DataDriftPreset(),

        # ── Qualité des données ───────────────────────────────────────────────
        DataQualityPreset(),
       

        # ── Drift sur les prédictions (model drift) ───────────────────────────
        ColumnDriftMetric(column_name="probability_default"),
        ColumnDriftMetric(column_name="prediction"),
    ])

    report.run(
        reference_data=reference_df[FEATURES + PREDICTION_COLS],
        current_data=current_df[FEATURES + PREDICTION_COLS],
    )

    # ── Sauvegarde du rapport HTML ────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"drift_report_{timestamp}.html"
    report.save_html(str(report_path))
    logger.info(f"✅ Rapport sauvegardé : {report_path}")

    return report_path


def run_operational_analysis(df: pd.DataFrame) -> dict:
    """
    Analyse des métriques opérationnelles depuis les logs.
    Détecte : latence anormale, taux de risque élevé, valeurs manquantes.
    """
    logger.info("📊 Analyse opérationnelle...")

    total       = len(df)
    high_risk   = df["prediction"].sum()
    avg_latency = df["latency_ms"].mean()
    max_latency = df["latency_ms"].max()
    p95_latency = df["latency_ms"].quantile(0.95)

    # ── Seuils d'alerte ───────────────────────────────────────────────────────
    alerts = []

    if avg_latency > 500:
        alerts.append(f"⚠️  Latence moyenne élevée : {avg_latency:.1f}ms (seuil : 500ms)")

    if p95_latency > 1000:
        alerts.append(f"⚠️  Latence P95 critique : {p95_latency:.1f}ms (seuil : 1000ms)")

    high_risk_rate = high_risk / total
    if high_risk_rate > 0.5:
        alerts.append(
            f"⚠️  Taux de risque élevé anormal : {high_risk_rate:.1%} "
            f"(seuil : 50%)"
        )

    # ── Valeurs manquantes par feature ────────────────────────────────────────
    missing = {col: int(df[col].isna().sum()) for col in FEATURES}
    missing_alerts = {k: v for k, v in missing.items() if v > 0}
    if missing_alerts:
        alerts.append(f"⚠️  Valeurs manquantes détectées : {missing_alerts}")

    stats = {
        "total_predictions": total,
        "high_risk_count":   int(high_risk),
        "high_risk_rate":    round(high_risk_rate, 4),
        "avg_probability":   round(df["probability_default"].mean(), 4),
        "avg_latency_ms":    round(avg_latency, 2),
        "p95_latency_ms":    round(p95_latency, 2),
        "max_latency_ms":    round(max_latency, 2),
        "alerts":            alerts,
    }

    # ── Affichage du résumé ───────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("📊 RÉSUMÉ OPÉRATIONNEL")
    print("═" * 60)
    print(f"  Total prédictions   : {total}")
    print(f"  Taux risque élevé   : {high_risk_rate:.1%}")
    print(f"  Probabilité moyenne : {stats['avg_probability']:.4f}")
    print(f"  Latence moyenne     : {avg_latency:.1f}ms")
    print(f"  Latence P95         : {p95_latency:.1f}ms")

    if alerts:
        print("\n🚨 ALERTES :")
        for alert in alerts:
            print(f"  {alert}")
    else:
        print("\n✅ Aucune anomalie détectée")

    print("═" * 60 + "\n")

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyse de drift — Credit Scoring API")
    parser.add_argument("--logs",      type=Path, default=PREDICTIONS_LOG,
                        help="Chemin vers le fichier predictions.jsonl")
    parser.add_argument("--reference", type=Path, default=None,
                        help="Chemin vers les données de référence (CSV)")
    parser.add_argument("--output",    type=Path, default=REPORTS_DIR,
                        help="Dossier de sortie pour les rapports HTML")
    args = parser.parse_args()

    # ── 1. Charger les logs de production ────────────────────────────────────
    production_df = load_production_logs(args.logs)

    # ── 2. Analyse opérationnelle ─────────────────────────────────────────────
    run_operational_analysis(production_df)

    # ── 3. Préparer les données de référence et courantes ─────────────────────
    reference_df = load_reference_data(args.reference, production_df)
    mid          = len(production_df) // 2
    current_df   = production_df.iloc[mid:][FEATURES + PREDICTION_COLS] \
                   if args.reference is None \
                   else production_df[FEATURES + PREDICTION_COLS]

    # ── 4. Rapport de drift Evidently ─────────────────────────────────────────
    report_path = run_drift_analysis(reference_df, current_df, args.output)
    print(f"📄 Rapport HTML disponible : {report_path}")
    print(f"   → Ouvrir dans le navigateur : open {report_path}")


if __name__ == "__main__":
    main()
