"""
profiling/convert_to_onnx.py — Conversion du modèle LightGBM vers ONNX
════════════════════════════════════════════════════════════════════════════
Convertit le pipeline sklearn (SimpleImputer + StandardScaler + LightGBM)
en modèle ONNX pour utilisation avec ONNX Runtime.

Stratégie :
  - On ne convertit PAS le pipeline sklearn complet
  - On convertit UNIQUEMENT le modèle LightGBM (déjà optimisé en numpy V3)
  - Le preprocessing reste en numpy pur (déjà optimisé, gain de 56.9%)
  - ONNX Runtime remplace uniquement model.predict_proba()

Pourquoi pas tout le pipeline ?
  - Le preprocessing numpy V3 est déjà optimal
  - Convertir SimpleImputer + StandardScaler en ONNX n'apporte pas de gain
  - Convertir LightGBM → ONNX élimine l'overhead du wrapper sklearn

Usage :
  python profiling/convert_to_onnx.py          # conversion + validation
  python profiling/convert_to_onnx.py --no-validate  # conversion seule

Prérequis (Linux/Docker uniquement) :
  pip install onnxmltools skl2onnx onnxruntime lightgbm
"""

import argparse
import logging
import os
from pathlib import Path

import joblib
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── CHEMINS ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
MODELS_DIR  = BASE_DIR / "models"
ONNX_PATH   = MODELS_DIR / "model.onnx"

# ─── FEATURES ─────────────────────────────────────────────────────────────────
FEATURES = [
    "EXT_SOURCE_1", "EXT_SOURCE_3", "EXT_SOURCE_2",
    "AMT_CREDIT", "AMT_ANNUITY", "DAYS_EMPLOYED",
    "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_LAST_PHONE_CHANGE",
    "AMT_INCOME_TOTAL",
]

SAMPLE_INPUT = np.array([[
    0.6, 0.5, 0.7, 120000.0, 6000.0,
    -1200.0, 100000.0, -16000.0, -50.0, 60000.0
]], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════

def convert_lightgbm_to_onnx(model, n_features: int) -> bytes:
    """
    Convertit LGBMClassifier en ONNX via onnxmltools.
    onnxmltools a son propre convertisseur LightGBM natif.
    """
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    logger.info("🔄 Conversion LightGBM → ONNX via onnxmltools...")

    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    onnx_model = convert_lightgbm(
        model,
        initial_types=initial_type,
        target_opset=12,
    )

    logger.info("✅ Conversion réussie")
    return onnx_model.SerializeToString()

def save_onnx_model(onnx_bytes: bytes, path: Path) -> None:
    """Sauvegarde le modèle ONNX sur disque."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onnx_bytes)
    size_kb = path.stat().st_size / 1024
    logger.info(f"✅ Modèle ONNX sauvegardé : {path} ({size_kb:.1f} KB)")


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_onnx(model, onnx_path: Path, preprocessor) -> bool:
    """
    Vérifie que le modèle ONNX donne les mêmes prédictions que LightGBM.
    CRITIQUE : la conversion ne doit pas changer les probabilités.
    Tolérance : 1e-4 (légères différences float32 acceptables).
    """
    import onnxruntime as rt
    import pandas as pd

    logger.info("🔍 Validation du modèle ONNX...")

    # ── Prédiction LightGBM original ─────────────────────────────────────────
    fill_values = preprocessor.named_steps["imputer"].statistics_
    scale_mean  = preprocessor.named_steps["scaler"].mean_
    scale_std   = preprocessor.named_steps["scaler"].scale_

    X_np   = np.where(np.isnan(SAMPLE_INPUT), fill_values, SAMPLE_INPUT)
    X_np   = (X_np - scale_mean) / scale_std
    X_df   = pd.DataFrame(X_np, columns=FEATURES)
    proba_lgbm = float(model.predict_proba(X_df)[0][1])

    # ── Prédiction ONNX Runtime ───────────────────────────────────────────────
    sess        = rt.InferenceSession(str(onnx_path))
    input_name  = sess.get_inputs()[0].name
    X_float32   = X_np.astype(np.float32)
    onnx_output = sess.run(None, {input_name: X_float32})

    # onnx_output[1] = probabilités shape (N, 2) — colonne 1 = proba classe 1
    proba_onnx = float(onnx_output[1][0][1])

    logger.info(f"  LightGBM proba : {proba_lgbm:.8f}")
    logger.info(f"  ONNX Runtime   : {proba_onnx:.8f}")
    logger.info(f"  Delta          : {abs(proba_lgbm - proba_onnx):.2e}")

    tolerance = 1e-4
    ok = abs(proba_lgbm - proba_onnx) < tolerance

    if ok:
        logger.info("✅ Validation réussie — prédictions cohérentes")
    else:
        logger.error(f"❌ Incohérence : delta = {abs(proba_lgbm - proba_onnx):.2e}")

    return ok


def benchmark_onnx_vs_lgbm(model, onnx_path: Path, preprocessor, n: int = 200) -> None:
    """
    Benchmark ONNX Runtime vs LightGBM sklearn sur N itérations.
    Mesure uniquement l'étape d'inférence (preprocessing numpy déjà optimisé).
    """
    import time
    import onnxruntime as rt
    import pandas as pd

    logger.info(f"⏱️  Benchmark inférence — {n} itérations...")

    fill_values = preprocessor.named_steps["imputer"].statistics_
    scale_mean  = preprocessor.named_steps["scaler"].mean_
    scale_std   = preprocessor.named_steps["scaler"].scale_

    X_np      = np.where(np.isnan(SAMPLE_INPUT), fill_values, SAMPLE_INPUT)
    X_np      = (X_np - scale_mean) / scale_std
    X_df      = pd.DataFrame(X_np, columns=FEATURES)
    X_float32 = X_np.astype(np.float32)

    sess       = rt.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name

    # ── Benchmark LightGBM ────────────────────────────────────────────────────
    import warnings
    times_lgbm = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n):
            t0 = time.perf_counter()
            model.predict_proba(X_df)
            times_lgbm.append((time.perf_counter() - t0) * 1000)

    # ── Benchmark ONNX Runtime ────────────────────────────────────────────────
    times_onnx = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {input_name: X_float32})
        times_onnx.append((time.perf_counter() - t0) * 1000)

    arr_lgbm = np.array(times_lgbm)
    arr_onnx = np.array(times_onnx)

    gain = round((arr_lgbm.mean() - arr_onnx.mean()) / arr_lgbm.mean() * 100, 1)

    print("\n" + "═" * 60)
    print("⚡ BENCHMARK INFÉRENCE — LightGBM vs ONNX Runtime")
    print("═" * 60)
    print(f"{'':20} {'Moy':>8} {'P95':>8} {'P99':>8}")
    print("─" * 60)
    print(
        f"{'LightGBM sklearn':<20} "
        f"{arr_lgbm.mean():>7.3f}ms "
        f"{np.percentile(arr_lgbm, 95):>7.3f}ms "
        f"{np.percentile(arr_lgbm, 99):>7.3f}ms"
    )
    print(
        f"{'ONNX Runtime':<20} "
        f"{arr_onnx.mean():>7.3f}ms "
        f"{np.percentile(arr_onnx, 95):>7.3f}ms "
        f"{np.percentile(arr_onnx, 99):>7.3f}ms"
    )
    print("═" * 60)
    status = "✅ mieux" if gain > 5 else ("➡️  neutre" if gain > -5 else "❌ pire")
    print(f"\n🏆 Gain ONNX Runtime : {gain}%  {status}")
    print(f"   {arr_lgbm.mean():.3f}ms → {arr_onnx.mean():.3f}ms\n")


# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Conversion LightGBM → ONNX")
    parser.add_argument("--no-validate", action="store_true",
                        help="Sauter la validation et le benchmark")
    parser.add_argument("--n", type=int, default=200,
                        help="Itérations pour le benchmark (défaut : 200)")
    args = parser.parse_args()

    # Chargement
    logger.info("📦 Chargement des artefacts...")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    features     = joblib.load(MODELS_DIR / "selected_features.pkl")
    model        = joblib.load(MODELS_DIR / "model.joblib")
    logger.info(f"✅ Modèle : {type(model).__name__}")

    # Conversion
    onnx_bytes = convert_lightgbm_to_onnx(model, len(features))
    save_onnx_model(onnx_bytes, ONNX_PATH)

    if not args.no_validate:
        # Validation cohérence
        ok = validate_onnx(model, ONNX_PATH, preprocessor)
        if not ok:
            logger.error("❌ Modèle ONNX invalide — ne pas déployer")
            return

        # Benchmark inférence
        benchmark_onnx_vs_lgbm(model, ONNX_PATH, preprocessor, args.n)

    logger.info(f"\n📄 Modèle ONNX prêt : {ONNX_PATH}")
    logger.info("   → Ajouter models/model.onnx au repo pour le déploiement")


if __name__ == "__main__":
    main()
