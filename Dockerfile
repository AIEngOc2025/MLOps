# ═══════════════════════════════════════════════════════════════════════════════
# Dockerfile — Credit Scoring API
# Build multi-stage : dependencies → test → app
#
# ⚠️  Port 7860 imposé par Hugging Face Spaces Docker
#
# Usage :
#   Lancer les tests seuls  : docker build --target test .
#   Construire l'image prod : docker build --target app -t credit-api:latest .
#   Lancer le conteneur     : docker run -p 7860:7860 credit-api:latest
# ═══════════════════════════════════════════════════════════════════════════════


# ─── Stage 1 : dépendances ────────────────────────────────────────────────────
FROM python:3.11-slim AS dependencies

WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
# (couche séparée pour profiter du cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ─── Stage 2 : tests ──────────────────────────────────────────────────────────
# Invocation : docker build --target test .
# Dans le CI/CD, cette étape bloque le build si les tests échouent.
FROM dependencies AS test

WORKDIR /app

COPY models/ ./models/
COPY src/     ./src/
COPY tests/   ./tests/

# Si pytest échoue → le build Docker s'arrête ici
RUN pytest tests/ -v --tb=short


# ─── Stage 3 : application (production) ───────────────────────────────────────
# Invocation : docker build --target app -t credit-api:latest .
# Image finale légère : pas de tests, pas d'outils de dev.
FROM dependencies AS app

WORKDIR /app

# Copier uniquement ce qui est nécessaire à l'exécution
COPY models/ ./models/
COPY src/     ./src/

# ⚠️  Port 7860 obligatoire pour Hugging Face Spaces Docker
EXPOSE 7860

# Vérification de santé toutes les 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Démarrage de l'API sur le port 7860
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
