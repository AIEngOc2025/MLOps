"""
generate_fake_logs.py — Génération de logs de production simulés
═══════════════════════════════════════════════════════════════════════════════
Génère un fichier predictions.jsonl simulant des appels réels à l'API.

Deux populations simulées :
  - Période 1 (référence) : distribution normale des features
  - Période 2 (courante)  : drift introduit sur EXT_SOURCE_1, AMT_CREDIT
                            → simule une dérive réelle en production

Usage :
  python generate_fake_logs.py
  python generate_fake_logs.py --n 500 --output logs/predictions.jsonl
"""

import json
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
LOGS_DIR    = BASE_DIR / "logs"
OUTPUT_FILE = LOGS_DIR / "predictions.jsonl"

THRESHOLD = 0.48


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_proba(features: dict) -> float:
    """
    Simule une probabilité de défaut basée sur les features.
    Logique simplifiée — pas le vrai modèle LightGBM.
    """
    # EXT_SOURCE élevé → risque faible
    ext_score = (
        (features.get("EXT_SOURCE_1") or 0.5) +
        (features.get("EXT_SOURCE_2") or 0.5) +
        (features.get("EXT_SOURCE_3") or 0.5)
    ) / 3

    # AMT_CREDIT élevé par rapport au revenu → risque élevé
    credit      = features.get("AMT_CREDIT") or 100000
    income      = features.get("AMT_INCOME_TOTAL") or 50000
    credit_ratio = min(credit / max(income, 1), 5) / 5

    # DAYS_EMPLOYED négatif → employé (risque faible)
    days_employed = features.get("DAYS_EMPLOYED") or -500
    employed_score = min(abs(days_employed) / 3000, 1) if days_employed < 0 else 0

    # Combinaison simple
    proba = 0.8 * (1 - ext_score) + 0.1 * credit_ratio + 0.1 * (1 - employed_score)
    proba = float(np.clip(proba + np.random.normal(0, 0.05), 0.01, 0.99))
    return round(proba, 4)


def generate_features(drift: bool = False) -> dict:
    """
    Génère un jeu de features aléatoires.

    drift=False : distribution normale (période de référence)
    drift=True  : drift introduit sur EXT_SOURCE_1 et AMT_CREDIT
                  → simule une dérive en production
    """
    if drift:
        # ── Période avec drift ────────────────────────────────────────────────
        # EXT_SOURCE_1 tire vers le bas (clients plus risqués)
        ext1 = float(np.clip(np.random.normal(0.35, 0.15), 0, 1))
        # AMT_CREDIT plus élevé (clients empruntent plus)
        amt_credit = float(np.random.normal(220000, 60000))
    else:
        # ── Période normale ───────────────────────────────────────────────────
        ext1       = float(np.clip(np.random.normal(0.55, 0.15), 0, 1))
        amt_credit = float(np.random.normal(150000, 50000))

    return {
        "EXT_SOURCE_1":           round(ext1, 4),
        "EXT_SOURCE_2":           round(float(np.clip(np.random.normal(0.55, 0.2), 0, 1)), 4),
        "EXT_SOURCE_3":           round(float(np.clip(np.random.normal(0.50, 0.2), 0, 1)), 4),
        "AMT_CREDIT":             round(max(amt_credit, 10000), 2),
        "AMT_ANNUITY":            round(float(np.random.normal(25000, 8000)), 2),
        "DAYS_EMPLOYED":          round(float(np.random.normal(-1500, 800)), 0),
        "AMT_GOODS_PRICE":        round(float(np.random.normal(130000, 45000)), 2),
        "DAYS_BIRTH":             round(float(np.random.normal(-14000, 3000)), 0),
        "DAYS_LAST_PHONE_CHANGE": round(float(np.random.normal(-500, 300)), 0),
        "AMT_INCOME_TOTAL":       round(float(np.random.normal(180000, 80000)), 2),
    }


def generate_log_entry(timestamp: datetime, drift: bool = False) -> dict:
    """Génère une entrée de log complète simulant un appel /predict."""
    features = generate_features(drift=drift)
    proba    = simulate_proba(features)
    pred     = int(proba >= THRESHOLD)

    if proba < 0.2:
        risk_label = "Très faible risque"
    elif proba < THRESHOLD:
        risk_label = "Risque modéré"
    elif proba < 0.7:
        risk_label = "Risque élevé"
    else:
        risk_label = "Risque très élevé"

    return {
        # ── Métadonnées ───────────────────────────────────────────────────────
        "timestamp":           timestamp.isoformat(),
        "latency_ms":          round(random.uniform(15, 120), 2),
        # ── Inputs ────────────────────────────────────────────────────────────
        **features,
        # ── Outputs ───────────────────────────────────────────────────────────
        "prediction":          pred,
        "probability_default": proba,
        "risk_label":          risk_label,
    }


def generate_logs(n: int, output_path: Path) -> None:
    """
    Génère n logs répartis sur 60 jours.
      - Première moitié  : sans drift (période de référence)
      - Deuxième moitié  : avec drift (période courante)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mid       = n // 2
    now       = datetime.now(timezone.utc)
    start     = now - timedelta(days=60)

    entries = []

    # ── Période 1 : référence (sans drift) ───────────────────────────────────
    for i in range(mid):
        ts = start + timedelta(seconds=i * (60 * 60 * 24 * 30 / mid))
        entries.append(generate_log_entry(ts, drift=False))

    # ── Période 2 : courante (avec drift) ────────────────────────────────────
    mid_date = start + timedelta(days=30)
    for i in range(n - mid):
        ts = mid_date + timedelta(seconds=i * (60 * 60 * 24 * 30 / (n - mid)))
        entries.append(generate_log_entry(ts, drift=True))

    # ── Écriture du fichier JSONL ─────────────────────────────────────────────
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"✅ {n} logs générés dans {output_path}")
    logger.info(f"   → {mid} logs sans drift  (période de référence)")
    logger.info(f"   → {n - mid} logs avec drift (période courante)")
    logger.info(f"   → Drift simulé sur : EXT_SOURCE_1, AMT_CREDIT")


# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génération de logs simulés")
    parser.add_argument("--n",      type=int,  default=200,
                        help="Nombre de logs à générer (défaut : 200)")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE,
                        help="Chemin du fichier de sortie")
    args = parser.parse_args()

    np.random.seed(42)
    random.seed(42)

    generate_logs(args.n, args.output)
    print(f"\n📄 Fichier généré : {args.output}")
    print(f"   → Lancer l'analyse : python drift_analysis.py")
