"""
dashboard.py — Dashboard de monitoring Credit Scoring
═══════════════════════════════════════════════════════════════════════════════
Visualise les logs de production et les résultats d'analyse de drift.

Sources de données :
  - Local  : logs/predictions.jsonl
  - HF     : huggingface.co/datasets/dataChaser/credit-score-logs

Usage :
  streamlit run dashboard.py                    # source locale
  streamlit run dashboard.py -- --source hf     # source HF Dataset

Prérequis :
  uv add streamlit plotly evidently
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
PREDICTIONS_LOG = BASE_DIR / "logs" / "predictions.jsonl"

HF_TOKEN      = os.getenv("HF_TOKEN")
HF_USERNAME   = os.getenv("HF_USERNAME", "dataChaser")
HF_DATASET_ID = f"{HF_USERNAME}/credit-score-logs"

FEATURES = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "AMT_CREDIT", "AMT_ANNUITY", "DAYS_EMPLOYED",
    "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_LAST_PHONE_CHANGE",
    "AMT_INCOME_TOTAL",
]

THRESHOLD = 0.48

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60)  # cache 60s — rafraîchit les données toutes les minutes
def load_logs(source: str) -> pd.DataFrame:
    """
    Charge les logs depuis la source choisie.
    @st.cache_data évite de recharger à chaque interaction utilisateur.
    """
    if source == "hf":
        return _load_from_hf()
    return _load_from_local()


def _load_from_local() -> pd.DataFrame:
    """Charge le fichier JSONL local."""
    if not PREDICTIONS_LOG.exists():
        st.error(f"Fichier introuvable : {PREDICTIONS_LOG}")
        st.stop()
    lines = PREDICTIONS_LOG.read_text().strip().splitlines()
    if not lines:
        st.warning("Fichier de logs vide.")
        st.stop()
    return pd.DataFrame([json.loads(l) for l in lines])


def _load_from_hf() -> pd.DataFrame:
    """Télécharge le fichier JSONL depuis HF Dataset."""
    try:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(
            repo_id=HF_DATASET_ID,
            filename="predictions.jsonl",
            repo_type="dataset",
            token=HF_TOKEN or None,
        )
        lines = Path(local_path).read_text().strip().splitlines()
        return pd.DataFrame([json.loads(l) for l in lines])
    except Exception as e:
        st.error(f"Erreur chargement HF Dataset : {e}")
        st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# DÉTECTION DE DRIFT (Evidently)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)  # cache 5 minutes — calcul plus long
def compute_drift(df: pd.DataFrame) -> dict:
    """
    Calcule le drift via Evidently.
    Retourne un dict avec les résultats par feature.
    """
    try:
        from evidently.report import Report
        from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

        cols = FEATURES + ["prediction", "probability_default"]
        mid  = len(df) // 2

        if mid < 10:
            return {"error": "Pas assez de données pour calculer le drift (minimum 20 logs)."}

        ref_df = df.iloc[:mid][cols]
        cur_df = df.iloc[mid:][cols]

        # Drift global + par feature
        metrics = [DatasetDriftMetric()] + [
            ColumnDriftMetric(column_name=col) for col in cols
        ]
        report = Report(metrics=metrics)
        report.run(reference_data=ref_df, current_data=cur_df)

        result     = report.as_dict()
        drift_data = {}

        for metric in result.get("metrics", []):
            # Drift global du dataset
            if metric["metric"] == "DatasetDriftMetric":
                drift_data["dataset_drift"] = metric["result"]["dataset_drift"]
                drift_data["drift_share"]   = metric["result"]["share_of_drifted_columns"]

            # Drift par colonne
            if metric["metric"] == "ColumnDriftMetric":
                col  = metric["result"]["column_name"]
                drift_data[col] = {
                    "drifted":  metric["result"]["drift_detected"],
                    "p_value":  round(metric["result"].get("p_value", 0), 4),
                    "score":    round(metric["result"].get("stattest_threshold", 0), 4),
                }

        return drift_data

    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSANTS DU DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def render_kpis(df: pd.DataFrame) -> None:
    """Affiche les KPIs opérationnels en haut du dashboard."""
    total          = len(df)
    high_risk      = int(df["prediction"].sum())
    high_risk_rate = high_risk / total
    avg_proba      = df["probability_default"].mean()
    avg_latency    = df["latency_ms"].mean()
    p95_latency    = df["latency_ms"].quantile(0.95)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📊 Total prédictions", total)
    col2.metric("🚨 Taux risque élevé", f"{high_risk_rate:.1%}",
                delta=f"{high_risk_rate - 0.3:.1%}" if high_risk_rate > 0.3 else None,
                delta_color="inverse")
    col3.metric("🎯 Probabilité moyenne", f"{avg_proba:.3f}")
    col4.metric("⚡ Latence moyenne", f"{avg_latency:.0f}ms",
                delta=f"{avg_latency - 200:.0f}ms" if avg_latency > 200 else None,
                delta_color="inverse")
    col5.metric("📈 Latence P95", f"{p95_latency:.0f}ms")


def render_alerts(df: pd.DataFrame) -> None:
    """Affiche les alertes opérationnelles."""
    alerts = []
    if df["latency_ms"].mean() > 500:
        alerts.append("⚠️ Latence moyenne > 500ms")
    if df["latency_ms"].quantile(0.95) > 1000:
        alerts.append("⚠️ Latence P95 > 1000ms")
    if df["prediction"].mean() > 0.5:
        alerts.append("⚠️ Taux de risque élevé > 50%")

    missing = {col: int(df[col].isna().sum()) for col in FEATURES if df[col].isna().sum() > 0}
    if missing:
        alerts.append(f"⚠️ Valeurs manquantes : {missing}")

    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("✅ Aucune anomalie opérationnelle détectée")


def render_predictions_over_time(df: pd.DataFrame) -> None:
    """Graphique de l'évolution des probabilités dans le temps."""
    df["timestamp"] = pd.to_datetime(df["timestamp"],utc=True)
    df_sorted = df.sort_values("timestamp")

    fig = px.scatter(
        df_sorted,
        x="timestamp",
        y="probability_default",
        color="risk_label",
        title="📈 Évolution des probabilités de défaut",
        labels={"probability_default": "Probabilité", "timestamp": "Date"},
        color_discrete_map={
            "Très faible risque": "#2ecc71",
            "Risque modéré":      "#f39c12",
            "Risque élevé":       "#e74c3c",
            "Risque très élevé":  "#8e44ad",
        },
    )
    fig.add_hline(y=THRESHOLD, line_dash="dash", line_color="red",
                  annotation_text=f"Seuil ({THRESHOLD})")
    st.plotly_chart(fig, use_container_width=True)


def render_feature_distributions(df: pd.DataFrame) -> None:
    """Histogrammes des distributions des features."""
    st.subheader("📊 Distribution des features")

    # Split temporel pour comparer référence vs courante
    mid    = len(df) // 2
    ref_df = df.iloc[:mid].copy()
    cur_df = df.iloc[mid:].copy()
    ref_df["période"] = "Référence"
    cur_df["période"] = "Courante"
    combined = pd.concat([ref_df, cur_df])

    # Affichage en grille 2 colonnes
    cols = st.columns(2)
    for i, feature in enumerate(FEATURES):
        with cols[i % 2]:
            fig = px.histogram(
                combined,
                x=feature,
                color="période",
                barmode="overlay",
                title=feature,
                opacity=0.7,
                color_discrete_map={"Référence": "#3498db", "Courante": "#e74c3c"},
            )
            fig.update_layout(height=300, showlegend=(i == 0))
            st.plotly_chart(fig, use_container_width=True)


def render_drift_results(drift_data: dict) -> None:
    """Affiche les résultats de détection de drift."""
    if "error" in drift_data:
        st.warning(f"⚠️ Drift non calculable : {drift_data['error']}")
        return

    # ── Drift global ──────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    dataset_drift = drift_data.get("dataset_drift", False)
    drift_share   = drift_data.get("drift_share", 0)

    with col1:
        if dataset_drift:
            st.error(f"🚨 Dataset Drift DÉTECTÉ — {drift_share:.0%} des features ont drifté")
        else:
            st.success(f"✅ Pas de Dataset Drift — {drift_share:.0%} des features ont drifté")

    # ── Drift par feature ─────────────────────────────────────────────────────
    feature_drift = {
        k: v for k, v in drift_data.items()
        if k not in ["dataset_drift", "drift_share", "error"]
        and isinstance(v, dict)
    }

    if feature_drift:
        drift_df = pd.DataFrame([
            {
                "Feature":  feat,
                "Drifté":   "🔴 Oui" if info["drifted"] else "🟢 Non",
                "P-value":  info["p_value"],
            }
            for feat, info in feature_drift.items()
        ])
        st.dataframe(drift_df, use_container_width=True, hide_index=True)


def render_latency_distribution(df: pd.DataFrame) -> None:
    """Distribution de la latence."""
    fig = px.histogram(
        df,
        x="latency_ms",
        title="⚡ Distribution de la latence (ms)",
        labels={"latency_ms": "Latence (ms)"},
        nbins=30,
        color_discrete_sequence=["#3498db"],
    )
    fig.add_vline(x=df["latency_ms"].mean(), line_dash="dash",
                  annotation_text="Moyenne", line_color="orange")
    fig.add_vline(x=df["latency_ms"].quantile(0.95), line_dash="dash",
                  annotation_text="P95", line_color="red")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Credit Scoring — Monitoring",
        page_icon="🏦",
        layout="wide",
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Configuration")
        source = st.radio("Source des données", ["local", "hf"], index=0)
        st.caption("local → logs/predictions.jsonl")
        st.caption("hf    → HF Dataset")

        if st.button("🔄 Rafraîchir les données"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.caption(f"HF Dataset : {HF_DATASET_ID}")

    # ── Titre ─────────────────────────────────────────────────────────────────
    st.title("🏦 Credit Scoring — Dashboard Monitoring")
    st.caption(f"Source : {'HF Dataset' if source == 'hf' else 'Local'} | "
               f"Dernière mise à jour : {datetime.now().strftime('%H:%M:%S')}")

    # ── Chargement des données ────────────────────────────────────────────────
    with st.spinner("Chargement des données..."):
        df = load_logs(source)

    st.success(f"✅ {len(df)} logs chargés")

    # ── Section 1 : KPIs ──────────────────────────────────────────────────────
    st.header("📊 KPIs Opérationnels")
    render_kpis(df)

    st.divider()

    # ── Section 2 : Alertes ───────────────────────────────────────────────────
    st.header("🚨 Alertes")
    render_alerts(df)

    st.divider()

    # ── Section 3 : Évolution temporelle ─────────────────────────────────────
    st.header("📈 Évolution temporelle")
    render_predictions_over_time(df)

    col1, col2 = st.columns(2)
    with col1:
        render_latency_distribution(df)

    st.divider()

    # ── Section 4 : Distributions des features ───────────────────────────────
    st.header("📊 Distributions des features")
    st.caption("Comparaison référence (première moitié) vs courante (deuxième moitié)")
    render_feature_distributions(df)

    st.divider()

    # ── Section 5 : Drift ─────────────────────────────────────────────────────
    st.header("🔍 Détection de Drift (Evidently)")
    with st.spinner("Calcul du drift..."):
        drift_data = compute_drift(df)
    render_drift_results(drift_data)

    st.divider()

    # ── Section 6 : Données brutes ────────────────────────────────────────────
    with st.expander("🗃️ Données brutes"):
        st.dataframe(df.tail(50), use_container_width=True)


if __name__ == "__main__":
    main()
