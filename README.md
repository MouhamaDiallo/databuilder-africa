# DataBuilder Africa
## Prédiction des coupures de courant — Ouagadougou, Burkina Faso

Développé dans le cadre du challenge **Sankara AI Network (SAIN)**
*"De la donnée brute à une solution déployée"*

---

## Problème

72% des entreprises en Afrique subsaharienne subissent des coupures de courant régulières.
Au Burkina Faso, on compte en moyenne **118 coupures par an**, sans que les ménages
et les entreprises puissent les anticiper.

Ce projet répond à une question simple :
*Est-ce qu'il y a un risque de coupure de courant dans les prochaines heures à Ouagadougou ?*

---

## Solution

Une application web qui prédit le **risque de coupure de courant** à partir de données
météorologiques accessibles gratuitement via NASA POWER.

L'utilisateur renseigne quelques paramètres (heure, mois, température, humidité, vent)
et reçoit instantanément un niveau de risque, une probabilité et un conseil pratique adapté.

**Démonstration en ligne :** [huggingface.co/spaces/yeroyordan/databuilder-africa](https://huggingface.co/spaces/yeroyordan/databuilder-africa)

---

## Dataset

| Source | Détails |
|--------|---------|
| NASA POWER | Données météo horaires, Ouagadougou 2024 |
| Période | 01/01/2024 — 31/12/2024 |
| Volume | 8 784 heures |
| Variables | Température, humidité, vent, ensoleillement, pression |

Source : [power.larc.nasa.gov](https://power.larc.nasa.gov/data-access-viewer/)

---

## Feature engineering

Les variables brutes ont été enrichies avec les éléments suivants :

| Feature | Description |
|---------|-------------|
| `saison` | Saison africaine (sèche froide / sèche chaude / pluies) |
| `periode` | Période de journée (nuit / matin / midi / après-midi / soir) |
| `heat_index` | Indice de chaleur ressentie (combinaison température et humidité) |
| `hour_sin / hour_cos` | Encodage cyclique de l'heure |
| `month_sin / month_cos` | Encodage cyclique du mois |

---

## Modèle

| Paramètre | Valeur |
|-----------|--------|
| Algorithme | Random Forest Classifier |
| Nombre d'arbres | 100 |
| Profondeur maximale | 8 |
| Précision | 72.6% |
| Découpage train / test | 80% / 20% |

Importance des variables :

```
temperature       19.4%
periode           14.8%
heat_index        14.7%
pressure           8.3%
solar_radiation    7.4%
humidity           7.0%
```

---

## Résultats

- Saison sèche chaude (mars — mai) : taux de coupure de **53%**
- Midi (10h — 14h) : période la plus à risque
- Nuit (0h — 5h) : période la plus stable, environ 10% de risque

---

## Installation locale

```bash
git clone https://huggingface.co/spaces/yeroyordan/databuilder-africa
cd databuilder-africa
pip install -r requirements.txt
python app.py
```

---

## Structure du projet

```
databuilder-africa/
├── app.py                          # Interface Gradio
├── requirements.txt                # Dépendances Python
├── random_forest_ouaga_v2.pkl      # Modèle entraîné
├── notebooks/
│   ├── 01_exploration.py           # Analyse exploratoire
│   └── 02_model_training.py        # Entraînement du modèle
└── data/
    └── ouagadougou_dataset_v2.csv  # Dataset enrichi
```

---

## Pistes d'amélioration

- Intégrer des données réelles de coupures issues de la SONABEL
- Étendre le modèle à d'autres villes africaines (Dakar, Abidjan, Bamako)
- Ajouter une prévision sur 24h glissantes
- Exposer une API REST pour intégration dans des applications tierces

---

## Auteur

Projet réalisé dans le cadre du challenge DataBuilder Africa, organisé par le Sankara AI Network (SAIN).

Données : NASA POWER 2024 — Ville cible : Ouagadougou, Burkina Faso
