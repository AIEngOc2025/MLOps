# 🏦 Credit Scoring API — MLOps

Système de scoring crédit basé sur **LightGBM**, exposé via une **API FastAPI**, testé avec **pytest**, conteneurisé avec **Docker** et déployé via un pipeline **CI/CD GitHub Actions**.

---

## Structure du projet

La structure a été organisée pour séparer clairement les différentes responsabilités du projet MLOps.

```
MLOps/
├── src/
│   └── api.py               # API FastAPI (endpoints /health, /predict, /predict/batch)
├── models/
│   ├── model.joblib         # Modèle LightGBM entraîné
│   ├── preprocessor.pkl     # Pipeline sklearn (imputation + normalisation)
│   └── selected_features.pkl# Liste ordonnée des features
├── tests/
│   └── test_api.py          # Tests unitaires et d'intégration (pytest)
├── monitoring/
│   ├── drift_analysis.py    # Scripts pour l'analyse de la dérive des données
│   └── reports/             # Rapports générés (HTML, images)
├── profiling/
│   ├── optimize_pipeline.py # Scripts pour le profilage et l'optimisation
│   └── reports/             # Rapports de profilage
├── utilitaires/
│   └── engine.py            # Scripts utilitaires divers
├── data/
│   └── ...                  # Données brutes et traitées
├── .github/
│   └── workflows/
│       └── ci-cd.yml        # Pipeline CI/CD GitHub Actions
├── Dockerfile               # Build multi-stage (dependencies → test → app)
├── compose.yml              # Orchestration de l'API avec Docker Compose
├── pyproject.toml           # Gestion des dépendances et configuration du projet
├── requirements.txt         # Dépendances Python (peut être redondant avec pyproject.toml)
├── .gitignore
└── README.md
```

---

## Modèle

| Paramètre | Valeur |
|---|---|
| Algorithme | LightGBM |
| Dataset | Home Credit Default Risk |
| Features | 10 (sélection par importance) |
| Seuil de décision | 0.48 (optimisé sur coût métier) |
| Métrique principale | AUC-ROC + coût métier (10×FN + 1×FP) |

**Features utilisées (ordre crucial) :**

`EXT_SOURCE_1`, `EXT_SOURCE_3`, `EXT_SOURCE_2`, `AMT_CREDIT`, `AMT_ANNUITY`, `DAYS_EMPLOYED`, `AMT_GOODS_PRICE`, `DAYS_BIRTH`, `DAYS_LAST_PHONE_CHANGE`, `AMT_INCOME_TOTAL`

**Labels de risque :**

| Probabilité de défaut | Label |
|---|---|
| < 0.20 | Très faible risque |
| 0.20 – 0.48 | Risque modéré |
| 0.48 – 0.70 | Risque élevé |
| ≥ 0.70 | Risque très élevé |

---

## Lancement rapide

### Prérequis

- Python 3.12+
- Docker & Docker Compose
- Les fichiers modèles dans `models/`

### En local

```bash
# 1. Installer les dépendances (de préférence dans un environnement virtuel)
pip install -r requirements.txt

# 2. Lancer l'API
uvicorn src.api:app --reload
```

### Avec Docker Compose

```bash
# Build et démarrage de l'API
docker compose up --build

# En arrière-plan
docker compose up -d --build
```

| Service | URL |
|---|---|
| API FastAPI | http://localhost:8000 |
| Documentation Swagger | http://localhost:8000/docs |

---

##  API — Endpoints

### `GET /health`

Vérifie l'état du serveur et le chargement des modèles.

```json
{
  "status": "ok",
  "model_loaded": true,
  "features_count": 10,
  "threshold": 0.48
}
```

### `POST /predict`

Prédit le risque de défaut pour un client.

**Exemple de requête :**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "EXT_SOURCE_1": 0.6, "EXT_SOURCE_2": 0.7, "EXT_SOURCE_3": 0.5,
    "AMT_CREDIT": 120000, "AMT_ANNUITY": 6000, "DAYS_EMPLOYED": -1200,
    "AMT_GOODS_PRICE": 100000, "DAYS_BIRTH": -16000,
    "DAYS_LAST_PHONE_CHANGE": -50, "AMT_INCOME_TOTAL": 60000
  }'
```

**Réponse :**
```json
{
  "prediction": 0,
  "probability_default": 0.4056,
  "risk_label": "Risque modéré",
  "threshold_used": 0.48,
  "model_available": true
}
```

### `POST /predict/batch`

Prédit pour une liste de clients (max 100).

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[{...}, {...}]'
```

---

##  Tests

```bash
# Lancer tous les tests
pytest tests/ -v

# Avec rapport de couverture
pytest tests/ -v --cov=src --cov-report=term-missing
```

Les tests couvrent : health check, prédiction unitaire, batch, labels de risque (4 niveaux), limite batch à 100, et les cas clients à risque faible/élevé. Tous les tests utilisent des **mocks** — aucun fichier `.pkl` requis pour les faire tourner.

---

##  Docker

Le `Dockerfile` utilise un **build multi-stage** :

| Stage | Rôle |
|---|---|
| `dependencies` | Installation des packages Python |
| `test` | Exécution des tests pytest (bloque le build si échec) |
| `app` | Image finale légère, sans les tests |

```bash
# Build uniquement l'image de production
docker build --target app -t credit-scoring-api:latest .

# Vérifier que le conteneur fonctionne
docker run -p 8000:8000 -v $(pwd)/models:/app/models:ro credit-scoring-api:latest
```

---

##  CI/CD — GitHub Actions

Le pipeline (`.github/workflows/ci-cd.yml`) s'exécute automatiquement à chaque push.

```
push / PR
    │
  ▼
 Tests pytest ──── ÉCHEC ──► ❌ Pipeline bloqué
    │
   OK
    │
    ▼
🐳 Build Docker ──► Push sur ghcr.io (GitHub Container Registry)
    │
    ▼ (main uniquement)
Déploiement ──► Smoke tests /health + /predict sur le conteneur
```

**3 jobs séquentiels :**

1. **Tests** — pytest avec couverture, rapport XML uploadé en artefact
2. **Build** — image taguée `latest`, nom de branche et SHA du commit, cache GitHub Actions
3. **Déploiement** — smoke tests automatiques sur le conteneur buildé (option SSH pour serveur distant disponible)

**Secrets GitHub requis** (pour déploiement sur serveur distant) :

| Secret | Description |
|---|---|
| `SSH_HOST` | IP ou domaine du serveur |
| `SSH_USER` | Utilisateur SSH |
| `SSH_PRIVATE_KEY` | Clé privée SSH |

---

##  Variables d'environnement

| Variable | Défaut | Description |
|---|---|---|
| `OPTIMAL_THRESHOLD` | `0.48` | Seuil de décision métier |
| `PYTHONUNBUFFERED` | `1` | Logs en temps réel dans Docker |
| `API_URL` | `http://127.0.0.1:8000` | URL de l'API (utilisé par certains scripts) |

---

##  Régénérer les modèles

Si les artefacts `models/` sont absents ou corrompus, exécutez le notebook ou le script d'entraînement approprié. Exemple de logique de sauvegarde :

```python
import joblib, os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

EXPECTED_FEATURES = [
    'EXT_SOURCE_1', 'EXT_SOURCE_3', 'EXT_SOURCE_2', 'AMT_CREDIT',
    'AMT_ANNUITY', 'DAYS_EMPLOYED', 'AMT_GOODS_PRICE', 'DAYS_BIRTH',
    'DAYS_LAST_PHONE_CHANGE', 'AMT_INCOME_TOTAL'
]

preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# Assurez-vous d'entraîner le préprocesseur avec vos données d'entraînement
# preprocessor.fit(X_train[EXPECTED_FEATURES])

joblib.dump(EXPECTED_FEATURES,          'models/selected_features.pkl')
joblib.dump(preprocessor,              'models/preprocessor.pkl')
joblib.dump(best_model,                'models/model.joblib')
```

> ⚠️ Vérifiez toujours les types après sauvegarde : `type(joblib.load(...))` doit retourner `Pipeline` et `LGBMClassifier`, jamais `str`.
