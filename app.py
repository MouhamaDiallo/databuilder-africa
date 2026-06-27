# ============================================================
# DataBuilder Africa — Interface Gradio
# Déploiement : Hugging Face Spaces
# ============================================================

import gradio as gr
import joblib
import numpy as np
import os

# --- Charger le modèle ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "random_forest_ouaga_v2.pkl")
model = joblib.load(MODEL_PATH)

# ── FONCTIONS UTILITAIRES ──────────────────────────────────

def get_saison(mois):
    if mois in [11, 12, 1, 2]: return 0   # Sèche froide
    elif mois in [3, 4, 5]:    return 1   # Sèche chaude
    else:                       return 2   # Pluies

def get_periode(heure):
    if 6 <= heure <= 9:     return 1  # Matin
    elif 10 <= heure <= 14: return 2  # Midi
    elif 15 <= heure <= 17: return 3  # Après-midi
    elif 18 <= heure <= 21: return 4  # Soir
    else:                   return 0  # Nuit

def get_heat_index(temp, humidity):
    return (
        -8.78469475556
        + 1.61139411 * temp
        + 2.33854883889 * humidity
        - 0.14611605 * temp * humidity
        - 0.012308094 * temp**2
        - 0.016424828 * humidity**2
        + 0.002211732 * temp**2 * humidity
        + 0.00072546 * temp * humidity**2
        - 0.000003582 * temp**2 * humidity**2
    )

# ── FONCTION PRINCIPALE DE PRÉDICTION ─────────────────────

def predire_coupure(heure, mois, temperature, humidity, wind_speed, is_weekend):
    """
    Fonction appelée quand l'utilisateur clique sur PRÉDIRE.
    Reçoit les valeurs du formulaire, retourne le résultat.
    """

    # Calcul des features dérivées
    saison      = get_saison(int(mois))
    periode     = get_periode(int(heure))
    heat_index  = get_heat_index(temperature, humidity)
    hour_sin    = np.sin(2 * np.pi * heure / 24)
    hour_cos    = np.cos(2 * np.pi * heure / 24)
    month_sin   = np.sin(2 * np.pi * mois / 12)
    month_cos   = np.cos(2 * np.pi * mois / 12)
    weekend_val = 1 if is_weekend else 0

    # Vecteur de features (même ordre qu'à l'entraînement)
    import pandas as pd
    features = pd.DataFrame([{
        'temperature':      temperature,
        'humidity':         humidity,
        'wind_speed':       wind_speed,
        'solar_radiation':  1.0,
        'pressure':         97.9,
        'saison':           saison,
        'periode':          periode,
        'heat_index':       heat_index,
        'hour_sin':         hour_sin,
        'hour_cos':         hour_cos,
        'month_sin':        month_sin,
        'month_cos':        month_cos,
        'is_weekend':       weekend_val
    }])

    # Prédiction
    probabilite = model.predict_proba(features)[0][1]

    # Construire le résultat affiché
    saison_label  = ["🌵 Sèche froide (Nov-Fév)",
                     "🔥 Sèche chaude (Mars-Mai)",
                     "🌧️ Saison des pluies (Juin-Oct)"][saison]
    periode_label = ["🌙 Nuit", "🌅 Matin", "☀️ Midi",
                     "🌤️ Après-midi", "🌆 Soir"][periode]

    if probabilite < 0.30:
        niveau  = "🟢 FAIBLE"
        conseil = "✅ Réseau stable. Consommation normale possible."
        couleur = "✅"
    elif probabilite < 0.55:
        niveau  = "🟡 MODÉRÉ"
        conseil = "⚠️ Risque modéré. Évitez les appareils énergivores."
        couleur = "⚠️"
    elif probabilite < 0.75:
        niveau  = "🔴 ÉLEVÉ"
        conseil = "🔋 Risque élevé ! Chargez vos appareils maintenant."
        couleur = "🔋"
    else:
        niveau  = "⛔ CRITIQUE"
        conseil = "🔌 Coupure très probable. Préparez vos alternatives."
        couleur = "🔌"

    resultat = f"""
## {niveau}

### Probabilité de coupure : {probabilite:.1%}

---

### 📊 Analyse
| Paramètre | Valeur |
|-----------|--------|
| Saison | {saison_label} |
| Période | {periode_label} |
| Chaleur ressentie | {heat_index:.1f} °C |

---

### 💡 Conseil
{conseil}
"""
    return resultat


# ── INTERFACE GRADIO ──────────────────────────────────────

with gr.Blocks(
    title="DataBuilder Africa — Coupures Ouagadougou",
    theme=gr.themes.Soft(),
    css="""
        .titre { text-align: center; }
        .footer { text-align: center; font-size: 0.8em; color: gray; }
    """
) as demo:

    # En-tête
    gr.Markdown("""
    # 🔌 DataBuilder Africa
    ## Prédiction des coupures de courant — Ouagadougou, Burkina Faso
    *Développé dans le cadre du challenge Sankara AI Network (SAIN)*
    ---
    """, elem_classes="titre")

    # Formulaire + résultat côte à côte
    with gr.Row():

        # Colonne gauche : les inputs
        with gr.Column():
            gr.Markdown("### 📥 Paramètres de prédiction")

            heure = gr.Slider(
                minimum=0, maximum=23, value=13, step=1,
                label="🕐 Heure de la journée (0 = minuit, 13 = 13h)"
            )
            mois = gr.Slider(
                minimum=1, maximum=12, value=4, step=1,
                label="📅 Mois (1 = Janvier, 12 = Décembre)"
            )
            temperature = gr.Slider(
                minimum=10, maximum=45, value=35, step=0.5,
                label="🌡️ Température (°C)"
            )
            humidity = gr.Slider(
                minimum=5, maximum=95, value=20, step=1,
                label="💧 Humidité relative (%)"
            )
            wind_speed = gr.Slider(
                minimum=0, maximum=15, value=4, step=0.5,
                label="💨 Vitesse du vent (m/s)"
            )
            is_weekend = gr.Checkbox(
                label="📆 Week-end ?",
                value=False
            )

            btn = gr.Button("⚡ PRÉDIRE", variant="primary", size="lg")

        # Colonne droite : le résultat
        with gr.Column():
            gr.Markdown("### 📤 Résultat")
            resultat = gr.Markdown("*Remplissez le formulaire et cliquez sur PRÉDIRE*")

    # Exemples pré-remplis pour faciliter la démo
    gr.Markdown("### 🧪 Exemples rapides")
    gr.Examples(
        examples=[
            [13,  4, 38, 15, 3,  False],  # Pire cas
            [3,  12, 18, 45, 7,  False],  # Nuit calme
            [19,  8, 28, 70, 5,  True ],  # Soir saison pluies
            [14,  5, 42, 10, 2,  False],  # Pic absolu
        ],
        inputs=[heure, mois, temperature, humidity, wind_speed, is_weekend],
        label="Clique sur un exemple pour le charger automatiquement"
    )

    # Footer
    gr.Markdown("""
    ---
    *Modèle : Random Forest | Données : NASA POWER 2024 (8784h) | Ville : Ouagadougou 🇧🇫*
    *Projet DataBuilder Africa — Sankara AI Network*
    """, elem_classes="footer")

    # Connecter le bouton à la fonction
    btn.click(
        fn=predire_coupure,
        inputs=[heure, mois, temperature, humidity, wind_speed, is_weekend],
        outputs=resultat
    )


# ── LANCER L'APPLICATION ──────────────────────────────────
if __name__ == "__main__":
    demo.launch()
