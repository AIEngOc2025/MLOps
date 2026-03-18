"""
profiling/optimize_pipeline.py — Optimisation du pipeline /predict
═══════════════════════════════════════════════════════════════════════════════
Compare 4 versions du pipeline de prédiction :

  V1 — Original     : pd.DataFrame + sklearn Pipeline (imputer + scaler) + LGBM
  V2 — Fix warning  : même chose mais DataFrame avec noms → élimine le warning
  V3 — Numpy optim  : imputation + scaling numpy pur, bypass sklearn au runtime
  V4 — Full optim   : numpy array + imputation + scaling + DataFrame pour LGBM

Preprocessor détecté : Pipeline(SimpleImputer(median) → StandardScaler)
Paramètres extraits au chargement, réutilisés à chaque inférence.

Usage :
  uv run --active python profiling/optimize_pipeline.py
  uv run --active python profiling/optimize_pipeline.py --n 500
"""

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── CHEMINS ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
MODELS_DIR  = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "profiling" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── FEATURES ─────────────────────────────────────────────────────────────────
FEATURES = [
    "EXT_SOURCE_1", "EXT_SOURCE_3", "EXT_SOURCE_2",
    "AMT_CREDIT", "AMT_ANNUITY", "DAYS_EMPLOYED",
    "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_LAST_PHONE_CHANGE",
    "AMT_INCOME_TOTAL",
]

SAMPLE_PAYLOAD = {
    "EXT_SOURCE_1": 0.6,  "EXT_SOURCE_2": 0.7,  "EXT_SOURCE_3": 0.5,
    "AMT_CREDIT": 120000.0, "AMT_ANNUITY": 6000.0, "DAYS_EMPLOYED": -1200.0,
    "AMT_GOODS_PRICE": 100000.0, "DAYS_BIRTH": -16000.0,
    "DAYS_LAST_PHONE_CHANGE": -50.0, "AMT_INCOME_TOTAL": 60000.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES ARTEFACTS
# ═══════════════════════════════════════════════════════════════════════════════

def load_artifacts():
    """
    Charge les artefacts et extrait les paramètres d'optimisation.
    Supporte : Pipeline(SimpleImputer → StandardScaler).
    """
    logger.info("📦 Chargement des artefacts...")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    features     = joblib.load(MODELS_DIR / "selected_features.pkl")
    model        = joblib.load(MODELS_DIR / "model.joblib")

    # ── Extraction des paramètres du Pipeline ─────────────────────────────────
    # SimpleImputer — valeurs de remplacement des NaN (médianes)
    fill_values = preprocessor.named_steps["imputer"].statistics_

    # StandardScaler — moyenne et écart-type pour la normalisation
    scale_mean  = preprocessor.named_steps["scaler"].mean_
    scale_std   = preprocessor.named_steps["scaler"].scale_

    logger.info(f"✅ fill_values  : {fill_values}")
    logger.info(f"✅ scale_mean   : {scale_mean}")
    logger.info(f"✅ scale_std    : {scale_std}")

    return preprocessor, features, model, fill_values, scale_mean, scale_std


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER — imputation + scaling numpy
# ═══════════════════════════════════════════════════════════════════════════════

def numpy_preprocess(payload: dict, features: list,
                     fill_values: np.ndarray,
                     scale_mean: np.ndarray,
                     scale_std: np.ndarray) -> np.ndarray:
    """
    Reproduit Pipeline(SimpleImputer → StandardScaler) en numpy pur.
    Bypass toute la validation sklearn → gain sur check_array et _validate_input.

    Étapes :
      1. Construire un array numpy depuis le payload
      2. Remplacer les NaN par les médianes (fill_values)
      3. Normaliser : (x - mean) / std
    """
    # Construction numpy — une seule allocation mémoire
    values = np.array(
        [[payload.get(f) if payload.get(f) is not None else np.nan for f in features]],
        dtype=np.float64
    )
    # Imputation — remplace NaN par la médiane
    X = np.where(np.isnan(values), fill_values, values)
    # Scaling — normalisation StandardScaler
    X = (X - scale_mean) / scale_std
    return X


# ═══════════════════════════════════════════════════════════════════════════════
# VERSIONS DU PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def pipeline_v1_original(payload, preprocessor, features, model):
    """V1 — Pipeline original (identique à api.py actuel)."""
    input_df = pd.DataFrame([{f: payload.get(f) for f in features}], columns=features)
    X        = preprocessor.transform(input_df)       # numpy array sans noms
    proba    = float(model.predict_proba(X)[0][1])    # warning sklearn
    return proba


def pipeline_v2_fix_warning(payload, preprocessor, features, model):
    """
    V2 — Fix du warning sklearn.
    Repasse X en DataFrame avec noms de colonnes avant predict_proba.
    → élimine le warning + réduit overhead validation LightGBM.
    """
    input_df = pd.DataFrame([{f: payload.get(f) for f in features}], columns=features)
    X_np     = preprocessor.transform(input_df)
    X_df     = pd.DataFrame(X_np, columns=features)   # noms de colonnes restaurés
    proba    = float(model.predict_proba(X_df)[0][1])
    return proba


def pipeline_v3_numpy_preprocess(payload, features, model,
                                  fill_values, scale_mean, scale_std):
    """
    V3 — Preprocessing numpy pur.
    Bypass SimpleImputer + StandardScaler sklearn → pas de check_array.
    Passe un DataFrame avec noms à LightGBM.
    """
    X     = numpy_preprocess(payload, features, fill_values, scale_mean, scale_std)
    X_df  = pd.DataFrame(X, columns=features)
    proba = float(model.predict_proba(X_df)[0][1])
    return proba


def pipeline_v4_full_optim(payload, features, model,
                            fill_values, scale_mean, scale_std):
    """
    V4 — Optimisation maximale.
    Identique à V3 — séparé pour mesurer l'impact de futures optimisations LGBM.
    C'est la version candidate pour intégration dans api.py.
    """
    X     = numpy_preprocess(payload, features, fill_values, scale_mean, scale_std)
    X_df  = pd.DataFrame(X, columns=features)
    proba = float(model.predict_proba(X_df)[0][1])
    return proba


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION DE COHÉRENCE
# ═══════════════════════════════════════════════════════════════════════════════

def validate_consistency(preprocessor, features, model,
                          fill_values, scale_mean, scale_std) -> bool:
    """
    Vérifie que toutes les versions donnent la même probabilité.
    CRITIQUE : l'optimisation ne doit jamais changer les prédictions.
    Tolérance : 1e-5 (différences numériques float acceptables).
    """
    logger.info("🔍 Validation de la cohérence des prédictions...")

    p1 = pipeline_v1_original(SAMPLE_PAYLOAD, preprocessor, features, model)
    p2 = pipeline_v2_fix_warning(SAMPLE_PAYLOAD, preprocessor, features, model)
    p3 = pipeline_v3_numpy_preprocess(SAMPLE_PAYLOAD, features, model,
                                       fill_values, scale_mean, scale_std)
    p4 = pipeline_v4_full_optim(SAMPLE_PAYLOAD, features, model,
                                 fill_values, scale_mean, scale_std)

    logger.info(f"  V1 proba : {p1:.8f}")
    logger.info(f"  V2 proba : {p2:.8f}")
    logger.info(f"  V3 proba : {p3:.8f}")
    logger.info(f"  V4 proba : {p4:.8f}")

    tolerance = 1e-5
    ok = all(abs(p - p1) < tolerance for p in [p2, p3, p4])

    if ok:
        logger.info("✅ Toutes les versions sont cohérentes")
    else:
        logger.error("❌ INCOHÉRENCE — optimisation invalide")
        for label, p in [("V2", p2), ("V3", p3), ("V4", p4)]:
            diff = abs(p - p1)
            if diff >= tolerance:
                logger.error(f"  {label} : delta = {diff:.2e} (seuil = {tolerance:.0e})")

    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark(fn, n: int, label: str) -> dict:
    """Mesure le temps d'exécution de fn sur n itérations."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    return {
        "label":   label,
        "mean_ms": round(float(arr.mean()), 3),
        "min_ms":  round(float(arr.min()),  3),
        "max_ms":  round(float(arr.max()),  3),
        "p95_ms":  round(float(np.percentile(arr, 95)), 3),
        "p99_ms":  round(float(np.percentile(arr, 99)), 3),
    }


def print_comparison(results: list, baseline_ms: float) -> None:
    """Affiche le tableau comparatif avec gains relatifs."""
    print("\n" + "═" * 75)
    print("🚀 COMPARAISON DES OPTIMISATIONS")
    print("═" * 75)
    print(f"{'Version':<38} {'Moy':>8} {'P95':>8} {'Gain':>8} {'Statut'}")
    print("─" * 75)

    for r in results:
        gain     = round((baseline_ms - r["mean_ms"]) / baseline_ms * 100, 1)
        gain_str = f"+{gain}%" if gain > 0 else f"{gain}%"
        status   = "✅ mieux" if gain > 5 else ("➡️  neutre" if gain > -5 else "❌ pire")
        print(
            f"{r['label']:<38} "
            f"{r['mean_ms']:>7.3f}ms "
            f"{r['p95_ms']:>7.3f}ms "
            f"{gain_str:>8}  "
            f"{status}"
        )

    print("═" * 75)
    best      = min(results, key=lambda x: x["mean_ms"])
    gain_best = round((baseline_ms - best["mean_ms"]) / baseline_ms * 100, 1)
    print(f"\n🏆 Meilleure version : {best['label']}")
    print(f"   Gain : {gain_best}%  ({baseline_ms:.3f}ms → {best['mean_ms']:.3f}ms)")
    print(f"   → Intégrer cette version dans api.py\n")


# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Optimisation pipeline — Credit Scoring")
    parser.add_argument("--n", type=int, default=200,
                        help="Nombre d'itérations par version (défaut : 200)")
    args = parser.parse_args()

    preprocessor, features, model, fill_values, scale_mean, scale_std = load_artifacts()

    # ── Validation cohérence ──────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = validate_consistency(preprocessor, features, model,
                                  fill_values, scale_mean, scale_std)
    if not ok:
        logger.error("❌ Arrêt — incohérence entre les versions")
        return

    # ── Warm-up ───────────────────────────────────────────────────────────────
    logger.info("🔥 Warm-up...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(10):
            pipeline_v1_original(SAMPLE_PAYLOAD, preprocessor, features, model)

    # ── Benchmark ─────────────────────────────────────────────────────────────
    logger.info(f"⏱️  Benchmark {args.n} itérations par version...")

    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        results.append(benchmark(
            lambda: pipeline_v1_original(SAMPLE_PAYLOAD, preprocessor, features, model),
            args.n, "V1 — Original (baseline)"
        ))
        results.append(benchmark(
            lambda: pipeline_v2_fix_warning(SAMPLE_PAYLOAD, preprocessor, features, model),
            args.n, "V2 — Fix warning DataFrame"
        ))
        results.append(benchmark(
            lambda: pipeline_v3_numpy_preprocess(
                SAMPLE_PAYLOAD, features, model, fill_values, scale_mean, scale_std),
            args.n, "V3 — Numpy impute + scale"
        ))
        results.append(benchmark(
            lambda: pipeline_v4_full_optim(
                SAMPLE_PAYLOAD, features, model, fill_values, scale_mean, scale_std),
            args.n, "V4 — Full optim (no sklearn runtime)"
        ))

    # ── Résultats ─────────────────────────────────────────────────────────────
    baseline_ms = results[0]["mean_ms"]
    print_comparison(results, baseline_ms)

    # ── Sauvegarde JSON ───────────────────────────────────────────────────────
    report = {
        "baseline_ms": baseline_ms,
        "iterations":  args.n,
        "results":     results,
    }
    report_path = REPORTS_DIR / "reports" / "profiling" / "optimization_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"✅ Rapport : {report_path}")


if __name__ == "__main__":
    main()
