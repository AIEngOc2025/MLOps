"""
simulate_traffic.py — Simulation de trafic réel vers l'API Credit Scoring
═══════════════════════════════════════════════════════════════════════════════
Envoie des requêtes variées vers l'API déployée sur Hugging Face.

Trois populations simulées :
  - Période 1 : clients normaux (distribution stable)
  - Période 2 : drift modéré  (EXT_SOURCE_1 plus bas, AMT_CREDIT plus élevé)
  - Période 3 : drift fort    (valeurs extrêmes → risque élevé garanti)

Usage :
  python simulate_traffic.py                         # 120 requêtes, API HF
  python simulate_traffic.py --n 200                 # 200 requêtes
  python simulate_traffic.py --url http://localhost:7860  # API locale
  python simulate_traffic.py --drift high            # drift agressif uniquement
"""

import argparse
import random
import time
import logging

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
DEFAULT_URL = "https://datachaser-credit-score.hf.space/predict"


# ═══════════════════════════════════════════════════════════════════════════════
# GÉNÉRATION DES PAYLOADS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_payload(drift: str = "none") -> dict:
    """
    Génère un payload aléatoire pour /predict.

    drift="none"     → distribution normale (clients peu risqués)
    drift="moderate" → drift modéré sur EXT_SOURCE_1 et AMT_CREDIT
    drift="high"     → drift agressif → probabilités élevées garanties
    """
    if drift == "high":
        # ── Drift fort — valeurs extrêmes garantissant un risque élevé ────────
        ext1       = float(np.clip(np.random.normal(0.10, 0.05), 0, 0.25))  # très bas
        ext2       = float(np.clip(np.random.normal(0.10, 0.05), 0, 0.25))  # très bas
        ext3       = float(np.clip(np.random.normal(0.10, 0.05), 0, 0.25))  # très bas
        amt_credit = float(np.random.normal(500000, 80000))                 # très élevé
        amt_income = float(np.random.normal(25000, 5000))                   # très bas
        days_emp   = float(np.random.normal(-100, 50))                      # peu d'ancienneté

    elif drift == "moderate":
        # ── Drift modéré ───────────────────────────────────────────────────────
        ext1       = float(np.clip(np.random.normal(0.30, 0.10), 0, 1))
        ext2       = float(np.clip(np.random.normal(0.35, 0.10), 0, 1))
        ext3       = float(np.clip(np.random.normal(0.30, 0.10), 0, 1))
        amt_credit = float(np.random.normal(300000, 60000))
        amt_income = float(np.random.normal(60000, 20000))
        days_emp   = float(np.random.normal(-500, 300))

    else:
        # ── Distribution normale (référence) ──────────────────────────────────
        ext1       = float(np.clip(np.random.normal(0.55, 0.15), 0, 1))
        ext2       = float(np.clip(np.random.normal(0.55, 0.15), 0, 1))
        ext3       = float(np.clip(np.random.normal(0.50, 0.15), 0, 1))
        amt_credit = float(np.random.normal(150000, 50000))
        amt_income = float(np.random.normal(180000, 80000))
        days_emp   = float(np.random.normal(-1500, 800))

    return {
        "EXT_SOURCE_1":           round(ext1, 4),
        "EXT_SOURCE_2":           round(ext2, 4),
        "EXT_SOURCE_3":           round(ext3, 4),
        "AMT_CREDIT":             round(max(amt_credit, 10000), 2),
        "AMT_ANNUITY":            round(float(np.random.normal(25000, 8000)), 2),
        "DAYS_EMPLOYED":          round(days_emp, 0),
        "AMT_GOODS_PRICE":        round(float(np.random.normal(130000, 45000)), 2),
        "DAYS_BIRTH":             round(float(np.random.normal(-14000, 3000)), 0),
        "DAYS_LAST_PHONE_CHANGE": round(float(np.random.normal(-500, 300)), 0),
        "AMT_INCOME_TOTAL":       round(max(amt_income, 1000), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION DU TRAFIC
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_traffic(api_url: str, n: int, delay: float, drift_mode: str) -> None:
    """
    Envoie n requêtes vers l'API en 3 périodes :
      - 1/3 : sans drift (référence)
      - 1/3 : drift modéré
      - 1/3 : drift fort
    """
    third   = n // 3
    success = 0
    errors  = 0
    start   = time.time()

    logger.info(f"🚀 Début simulation — {n} requêtes vers {api_url}")

    if drift_mode == "high":
        # Mode drift agressif uniquement
        logger.info(f"   → {n} requêtes en drift fort")
        periods = [("high", n)]
    else:
        # Mode normal : 3 périodes
        logger.info(f"   → {third} requêtes sans drift    (référence)")
        logger.info(f"   → {third} requêtes drift modéré")
        logger.info(f"   → {n - 2*third} requêtes drift fort")
        periods = [
            ("none",     third),
            ("moderate", third),
            ("high",     n - 2 * third),
        ]

    req_num = 0
    for period_drift, period_n in periods:
        for _ in range(period_n):
            req_num += 1
            payload = generate_payload(drift=period_drift)

            try:
                response = requests.post(api_url, json=payload, timeout=15)

                if response.status_code == 200:
                    success += 1
                    res = response.json()
                    if req_num % 10 == 0:
                        logger.info(
                            f"  [{req_num}/{n}] ✅ "
                            f"proba={res['probability_default']:.3f} "
                            f"| {res['risk_label']:<20} "
                            f"| drift={period_drift}"
                        )
                else:
                    errors += 1
                    logger.warning(f"  [{req_num}/{n}] ⚠️  Status {response.status_code}")

            except requests.exceptions.Timeout:
                errors += 1
                logger.warning(f"  [{req_num}/{n}] ⏱️  Timeout")
            except requests.exceptions.ConnectionError:
                errors += 1
                logger.error(f"  [{req_num}/{n}] ❌ Connexion impossible — {api_url}")
                break
            except Exception as e:
                errors += 1
                logger.error(f"  [{req_num}/{n}] ❌ Erreur : {e}")

            if delay > 0:
                time.sleep(delay)

    # ── Résumé ────────────────────────────────────────────────────────────────
    duration = time.time() - start
    print("\n" + "═" * 55)
    print("📊 RÉSUMÉ SIMULATION")
    print("═" * 55)
    print(f"  Total envoyées  : {n}")
    print(f"  Succès          : {success}")
    print(f"  Erreurs         : {errors}")
    print(f"  Durée totale    : {duration:.1f}s")
    print(f"  Débit moyen     : {success / max(duration, 1):.1f} req/s")
    print("═" * 55)
    print(f"\n✅ Rafraîchis le dashboard pour voir le drift détecté.")
    print(f"   → Les logs sont pushés vers HF toutes les 30 requêtes.")


# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation de trafic — Credit Scoring API")
    parser.add_argument("--url",   type=str,   default=DEFAULT_URL,
                        help=f"URL de l'API (défaut : {DEFAULT_URL})")
    parser.add_argument("--n",     type=int,   default=120,
                        help="Nombre de requêtes (défaut : 120)")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Délai entre requêtes en secondes (défaut : 0.1s)")
    parser.add_argument("--drift", choices=["auto", "high"], default="auto",
                        help="Mode drift : auto (3 périodes) | high (drift fort uniquement)")
    args = parser.parse_args()

    np.random.seed(42)
    random.seed(42)

    simulate_traffic(args.url, args.n, args.delay, args.drift)