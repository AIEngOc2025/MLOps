# Description:

Développez une API (Gradio, FastAPI) pour exposer votre modèle. L'API doit recevoir des données d'entrée et retourner une prédiction. Conteneurisez cette API avec Docker. Ensuite, créez un pipeline d'Intégration Continue et de Déploiement Continu (CI/CD) (ex: GitHub Actions). Ce pipeline devra automatiquement :

1.Exécuter des tests (unitaires, intégration) sur votre code API et modèle.

2.Construire l'image Docker de l'API si les tests sont concluants.

3.Déployer l'image conteneurisée sur un environnement cible (simulé ou réel).

# Prérequis:

-Avoir le code versionné sur une plateforme supportant la CI/CD.

-Avoir choisi un framework d'API.

-Avoir installé Docker.

## Résultats attendus :

-Un code source fonctionnel pour l'API.

-Un Dockerfilepour créer une image Docker de l'API.

-Un pipeline CI/CD fonctionnel et automatisé visible sur la plateforme, qui déploie l'API.

-Des tests automatisés intégrés au pipeline.

# Recommandations:

-Commencez par une API simple et un pipeline basique, puis itérez.

-Incluez une gestion des erreurs dans l'API et documentez-la (ex: Swagger).

-Séparez les étapes de build, test et déploiement dans le pipeline CI/CD.

-Utilisez des secrets pour gérer les credentials.

-Utilisez Hugging Face Spaces qui est particulièrement simple d’utilisation pour ce genre de déploiement.

# Points de vigilance:

-Assurez-vous que les tests sont fiables et couvrent les cas critiques, 
par exemple :

des entrées avec des données manquantes pour des champs obligatoires,

des valeurs hors des plages attendues (ex: un âge de -5 ans ou un revenu de 0 si ce n'est pas censé être possible),

ou des types de données incorrects (ex: du texte là où un chiffre est attendu).

Sécurisez l'API et le pipeline (gestion des secrets, validation d'entrée).

Gérez correctement le chargement du modèle dans l'API.

Lorsque vous intégrez un modèle de machine learning dans une API, il est crucial de ne pas charger le modèle à chaque requête. Cela entraînerait des lenteurs importantes voire un échec sous charge. Chargez le modèle une seule fois, au moment du démarrage de l’API, puis réutilisez le dans toutes les requêtes.

Cela permet de :

Réduire le temps de réponse de l’API.

Éviter une surcharge mémoire.

Améliorer la scalabilité.

Vérifiez que l'environnement de déploiement dispose des ressources nécessaires.

# Outils: 

-Gradio/FastAPI

-Docker

-Postman/curl

-GitHub Actions/GitLab CI/Jenkins

-Pytest

Plateformes de déploiement (Hugging Face, Heroku, Google Cloud Run...).

 

# Ressources: 

-Documentation FastAPI.

-Documentation Gradio.

-Docker.

-tutoriels sur les tests.

-Github actions.