# ═══════════════════════════════════════════════════════════════════════════════
# Dockerfile — Credit Scoring API
# Build multi-stage : dependencies → test → app
#
# Usage :
#   Lancer les tests seuls  : docker build --target test .
#   Construire l'image prod : docker build --target app -t credit-api:latest .
#   Lancer le conteneur     : docker run -p 8000:8000 credit-api:latest
# ═══════════════════════════════════════════════════════════════════════════════


# ─── Stage 1 : dépendances ────────────────────────────────────────────────────
# Base commune aux deux stages suivants.
# On installe ici toutes les dépendances Python une seule fois.
FROM python:3.11-slim AS dependencies

WORKDIR /app

# Dépendances système minimales (build-essential pour certains packages C)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
# (séparé du COPY src/ pour profiter du cache Docker si requirements.txt n'a pas changé)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ─── Stage 2 : tests ──────────────────────────────────────────────────────────
# Hérite de `dependencies`.
# Contient le code source ET les tests — NE sera PAS dans l'image de production.
#
# Invocation : docker build --target test .
# Dans le CI/CD (GitHub Actions), cette étape bloque le build si les tests échouent.
FROM dependencies AS test

# WORKDIR identique au stage dependencies pour que les imports Python fonctionnent
WORKDIR /app

# Copier le code source, les modèles et les tests
COPY models/ ./models/
COPY src/     ./src/
COPY tests/   ./tests/

# Lancer les tests — si pytest échoue, le build Docker s'arrête ici
RUN pytest tests/ -v --tb=short


# ─── Stage 3 : application (production) ──────────────────────────────────────
# Hérite de `dependencies` (PAS de `test`).
# Image finale légère : pas de tests, pas d'outils de dev.
#
# Invocation : docker build --target app -t credit-api:latest .
FROM dependencies AS app

WORKDIR /app

# Copier uniquement ce qui est nécessaire à l'exécution
COPY models/ ./models/
COPY src/     ./src/

# Port exposé par uvicorn
EXPOSE 8000

# Vérification de santé toutes les 30s — appelle GET /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Démarrage de l'API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]