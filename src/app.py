import gradio as gr
import requests

# URL de l'API FastAPI (locale ou Docker)
API_URL = "http://127.0.0.1:8000/predict"

def predict_credit(ext1, ext2, ext3, credit, annuity, employed, goods, birth, phone, income):
    payload = {
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "DAYS_EMPLOYED": employed,
        "AMT_GOODS_PRICE": goods,
        "DAYS_BIRTH": birth,
        "DAYS_LAST_PHONE_CHANGE": phone,
        "AMT_INCOME_TOTAL": income
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)

        # BUG CORRIGÉ : gestion explicite des erreurs HTTP (500, 422, 503...)
        if response.status_code != 200:
            return f"⚠️ Erreur API ({response.status_code}) : {response.json().get('detail', 'Erreur inconnue')}"

        res = response.json()

        status = "✅ ACCORDÉ" if res["prediction"] == 0 else "❌ REFUSÉ"
        proba  = f"Probabilité de défaut : {res['probability_default']:.2%}"
        risk   = f"Niveau de risque : {res['risk_label']}"
        # AJOUT : affichage du seuil utilisé pour la transparence
        threshold = f"Seuil de décision : {res['threshold_used']}"

        return f"{status}\n\n{proba}\n{risk}\n{threshold}"

    except requests.exceptions.ConnectionError:
        return "❌ Impossible de joindre l'API. Vérifiez qu'uvicorn est bien lancé sur le port 8000."
    except requests.exceptions.Timeout:
        return "⏱️ L'API n'a pas répondu dans les délais (timeout 5s)."
    except Exception as e:
        return f"Erreur inattendue : {e}"


demo = gr.Interface(
    fn=predict_credit,
    inputs=[
        gr.Slider(0, 1, step=0.01, value=0.5, label="EXT_SOURCE_1"),
        gr.Slider(0, 1, step=0.01, value=0.5, label="EXT_SOURCE_2"),
        gr.Slider(0, 1, step=0.01, value=0.5, label="EXT_SOURCE_3"),
        gr.Number(label="Montant Crédit (AMT_CREDIT)",          value=100000, minimum=0),
        gr.Number(label="Annuité (AMT_ANNUITY)",                value=5000,   minimum=0),
        gr.Number(label="Jours employés (négatif)",             value=-1000,  maximum=0),
        gr.Number(label="Prix des biens (AMT_GOODS_PRICE)",     value=80000,  minimum=0),
        gr.Number(label="Âge en jours (négatif)",               value=-15000, maximum=0),
        gr.Number(label="Dernier changement tél. (jours nég.)", value=-100,   maximum=0),
        gr.Number(label="Revenu Total (AMT_INCOME_TOTAL)",      value=50000,  minimum=0),
    ],
    outputs=gr.Textbox(label="Résultat de la prédiction", lines=5),
    title="🏦 Simulateur de Crédit — Scoring Client",
    description=(
        "Entrez les caractéristiques du client pour obtenir une prédiction de risque de défaut. "
        "Les champs en jours doivent être **négatifs** (ex: -15000 jours = ~41 ans)."
    ),
    flagging_mode="never",  # BUG CORRIGÉ : 'allow_flagging' est déprécié en Gradio 4+
)

if __name__ == "__main__":
    demo.launch()