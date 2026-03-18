import joblib
import pandas as pd
from pathlib import Path

class ScoringEngine:
    def __init__(self, models_dir: Path, threshold: float):
        self.threshold = threshold
        self.models_dir = models_dir
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            self.preprocessor = joblib.load(self.models_dir / "preprocessor.pkl")
            self.features = joblib.load(self.models_dir / "selected_features.pkl")
            self.model = joblib.load(self.models_dir / "model.joblib")
            self.ready = True
        except Exception as e:
            print(f"❌ Erreur Engine : {e}")
            self.ready = False

    def run_inference(self, data_dict: dict):
        df = pd.DataFrame([data_dict])[self.features]
        X = self.preprocessor.transform(df)
        proba = float(self.model.predict_proba(X)[0][1])
        return proba, int(proba >= self.threshold)

    def get_risk_label(self, proba: float):
        if proba < 0.2: return "Très faible"
        if proba < self.threshold: return "Modéré"
        if proba < 0.7: return "Élevé"
        return "Très élevé"