
"""
profiling/profile_api.py — Profiling de l'API Credit Scoring
═══════════════════════════════════════════════════════════════════════════════
Mesure précise du temps passé dans chaque étape du pipeline /predict :
  1. Validation Pydantic
  2. preprocessor.transform()
  3. model.predict_proba()
  4. log_prediction() (buffer + flush HF)

Deux modes :
  - cProfile  : profiling détaillé fonction par fonction
  - Manuel    : chronomètre par étape (plus lisible)

Usage :
  uv run python profiling/profile_api.py             # profiling manuel
  uv run python profiling/profile_api.py --mode cprofile  # cProfile détaillé
  uv run python profiling/profile_api.py --n 100     # 100 itérations

Prérequis :
  Modèles chargés dans models/
"""

import argparse
import cProfile
import io
import logging
import os
import pstats
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── CHEMINS ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"


# ─── FEATURES ─────────────────────────────────────────────────────────────────
FEATURES = [
    "EXT_SOURCE_1", "EXT_SOURCE_3", "EXT_SOURCE_2",
    "AMT_CREDIT", "AMT_ANNUITY", "DAYS_EMPLOYED",
    "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_LAST_PHONE_CHANGE",
    "AMT_INCOME_TOTAL",
]

THRESHOLD = 0.48

# ─── PAYLOAD DE TEST ──────────────────────────────────────────────────────────
SAMPLE_PAYLOAD = {
    "EXT_SOURCE_1": 0.6, "EXT_SOURCE_2": 0.7, "EXT_SOURCE_3": 0.5,
    "AMT_CREDIT": 120000.0, "AMT_ANNUITY": 6000.0,
    "DAYS_EMPLOYED": -1200.0, "AMT_GOODS_PRICE": 100000.0,
    "DAYS_BIRTH": -16000.0, "DAYS_LAST_PHONE_CHANGE": -50.0,
    "AMT_INCOME_TOTAL": 60000.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES ARTEFACTS
# ═══════════════════════════════════════════════════════════════════════════════

def load_artifacts():
    """Charge les artefacts du modèle."""
    logger.info("📦 Chargement des artefacts...")

    t0 = time.perf_counter()
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    t1 = time.perf_counter()

    selected_features = joblib.load(MODELS_DIR / "selected_features.pkl")
    t2 = time.perf_counter()

    model = joblib.load(MODELS_DIR / "model.joblib")
    t3 = time.perf_counter()

    logger.info(f"  preprocessor.pkl     : {(t1-t0)*1000:.2f}ms")
    logger.info(f"  selected_features.pkl: {(t2-t1)*1000:.2f}ms")
    logger.info(f"  model.joblib         : {(t3-t2)*1000:.2f}ms")
    logger.info(f"  Total chargement     : {(t3-t0)*1000:.2f}ms")

    return preprocessor, selected_features, model


# ═══════════════════════════════════════════════════════════════════════════════
# PROFILING MANUEL — chronomètre par étape
# ═══════════════════════════════════════════════════════════════════════════════

def profile_manual(preprocessor, features, model, n: int) -> dict:
    """
    Mesure le temps de chaque étape sur n itérations.
    Retourne les statistiques (moyenne, min, max, p95) par étape.
    """
    logger.info(f"\n🔍 Profiling manuel — {n} itérations")

    times = {
        "dataframe_build":    [],
        "preprocessor":       [],
        "model_predict":      [],
        "postprocess":        [],
        "total":              [],
    }

    for i in range(n):
        t_start = time.perf_counter()

        # ── Étape 1 : construction du DataFrame ───────────────────────────────
        t0 = time.perf_counter()
        input_dict = {f: SAMPLE_PAYLOAD.get(f) for f in features}
        input_df   = pd.DataFrame([input_dict], columns=features)
        t1 = time.perf_counter()
        times["dataframe_build"].append((t1 - t0) * 1000)

        # ── Étape 2 : preprocessor.transform() ───────────────────────────────
        t0 = time.perf_counter()
        X = preprocessor.transform(input_df)
        t1 = time.perf_counter()
        times["preprocessor"].append((t1 - t0) * 1000)

        # ── Étape 3 : model.predict_proba() ──────────────────────────────────
        t0 = time.perf_counter()
        proba = float(model.predict_proba(X)[0][1])
        t1 = time.perf_counter()
        times["model_predict"].append((t1 - t0) * 1000)

        # ── Étape 4 : post-traitement (labels, arrondi) ───────────────────────
        t0 = time.perf_counter()
        prediction = int(proba >= THRESHOLD)
        risk_label = (
            "Très faible risque" if proba < 0.2 else
            "Risque modéré"      if proba < THRESHOLD else
            "Risque élevé"       if proba < 0.7 else
            "Risque très élevé"
        )
        _ = round(proba, 4)
        t1 = time.perf_counter()
        times["postprocess"].append((t1 - t0) * 1000)

        times["total"].append((time.perf_counter() - t_start) * 1000)

    return times


def print_profiling_report(times: dict) -> None:
    """Affiche le rapport de profiling."""
    print("\n" + "═" * 65)
    print("📊 RAPPORT DE PROFILING — Pipeline /predict")
    print("═" * 65)
    print(f"{'Étape':<25} {'Moy':>8} {'Min':>8} {'Max':>8} {'P95':>8} {'%':>6}")
    print("─" * 65)

    total_avg = sum(
        sum(v) / len(v)
        for k, v in times.items()
        if k != "total"
    )

    for step, values in times.items():
        if step == "total":
            continue
        avg = sum(values) / len(values)
        mn  = min(values)
        mx  = max(values)
        p95 = sorted(values)[int(len(values) * 0.95)]
        pct = (avg / total_avg * 100) if total_avg > 0 else 0

        print(f"  {step:<23} {avg:>7.3f}ms {mn:>7.3f}ms {mx:>7.3f}ms {p95:>7.3f}ms {pct:>5.1f}%")

    print("─" * 65)
    total_vals = times["total"]
    avg_total  = sum(total_vals) / len(total_vals)
    p95_total  = sorted(total_vals)[int(len(total_vals) * 0.95)]
    print(f"  {'TOTAL':<23} {avg_total:>7.3f}ms {'':>8} {'':>8} {p95_total:>7.3f}ms")
    print("═" * 65)

    # ── Identification du goulot d'étranglement ────────────────────────────
    bottleneck = max(
        {k: sum(v)/len(v) for k, v in times.items() if k != "total"}.items(),
        key=lambda x: x[1]
    )
    print(f"\n🚨 Goulot principal : {bottleneck[0]} ({bottleneck[1]:.3f}ms)")
    print(f"   → Cible d'optimisation prioritaire\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PROFILING cPROFILE — détail fonction par fonction
# ═══════════════════════════════════════════════════════════════════════════════

def profile_cprofile(preprocessor, features, model, n: int) -> None:
    """Lance cProfile sur le pipeline complet."""
    logger.info(f"\n🔍 Profiling cProfile — {n} itérations")

    def pipeline():
        for _ in range(n):
            input_dict = {f: SAMPLE_PAYLOAD.get(f) for f in features}
            input_df   = pd.DataFrame([input_dict], columns=features)
            X          = preprocessor.transform(input_df)
            proba      = float(model.predict_proba(X)[0][1])
            _          = int(proba >= THRESHOLD)

    # ── Lancer cProfile ───────────────────────────────────────────────────────
    profiler = cProfile.Profile()
    profiler.enable()
    pipeline()
    profiler.disable()

    # ── Afficher les résultats triés par temps cumulé ─────────────────────────
    stream = io.StringIO()
    stats  = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(20)  # top 20 fonctions

    print("\n" + "═" * 65)
    print("📊 RAPPORT cPROFILE — Top 20 fonctions par temps cumulé")
    print("═" * 65)
    print(stream.getvalue())

    # ── Sauvegarder le rapport ────────────────────────────────────────────────
    report_path = Path(__file__).parent.parent /"reports"/ "profiling" / "cprofile_report.txt"
    with open(report_path, "w") as f:
        stats2 = pstats.Stats(profiler, stream=f)
        stats2.sort_stats("cumulative")
        stats2.print_stats()
    logger.info(f"✅ Rapport cProfile sauvegardé : {report_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Profiling — Credit Scoring API")
    parser.add_argument("--mode", choices=["manual", "cprofile", "both"], default="both",
                        help="Mode de profiling (défaut : both)")
    parser.add_argument("--n",   type=int, default=200,
                        help="Nombre d'itérations (défaut : 200)")
    args = parser.parse_args()

    # 1. Charger les artefacts
    preprocessor, features, model = load_artifacts()

    # 2. Profiling
    if args.mode in ("manual", "both"):
        times = profile_manual(preprocessor, features, model, args.n)
        print_profiling_report(times)

    if args.mode in ("cprofile", "both"):
        profile_cprofile(preprocessor, features, model, args.n)


if __name__ == "__main__":
    main()
