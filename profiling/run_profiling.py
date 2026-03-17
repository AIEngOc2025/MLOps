import cProfile
import pstats
import io
import pandas as pd
from pathlib import Path
from src.api import ScoringEngine

def profile_inference():
    # 1. Setup
    BASE_DIR = Path(__file__).resolve().parent.parent
    engine = ScoringEngine(BASE_DIR / "models", 0.48)
    
    # Données de test typiques
    sample_data = {
        "EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.3, "EXT_SOURCE_3": 0.1,
        "AMT_CREDIT": 50000, "AMT_ANNUITY": 2500, "DAYS_EMPLOYED": -1000,
        "AMT_GOODS_PRICE": 45000, "DAYS_BIRTH": -15000, 
        "DAYS_LAST_PHONE_CHANGE": -1, "AMT_INCOME_TOTAL": 20000
    }

    print("🚀 Démarrage du profiling (1000 itérations)...")
    
    # 2. Profiling
    pr = cProfile.Profile()
    pr.enable()

    for _ in range(1000):
        engine.predict(sample_data)

    pr.disable()

    # 3. Formatage des résultats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(15)  # Affiche le top 15 des fonctions les plus lourdes
    
    print(s.getvalue())

    # Sauvegarde pour le rapport d'étape 4
    with open(Path(__file__).parent / "profile_results.txt", "w") as f:
        f.write(s.getvalue())

if __name__ == "__main__":
    profile_inference()