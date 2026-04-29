# 🎓 Prédicteur de Performance Étudiante - Système IA Hybride

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)

Bienvenue dans ce projet de pointe qui combine l'apprentissage automatique traditionnel (Machine Learning) et l'apprentissage profond (Deep Learning) pour prédire les résultats académiques des étudiants. Ce système est conçu pour être **prêt pour la production**, hautement performant et visuellement époustouflant.

---

## 🧐 Qu'est-ce que ce projet ? (Pour les nuls)

Imaginez que vous êtes un directeur d'école. Vous voulez savoir quels étudiants risquent de rater leurs examens avant même qu'ils ne les passent. 

Ce logiciel prend des données (heures d'étude, sommeil, assiduité, stress) et utilise une "intelligence" pour deviner la note finale. Mais il ne fait pas que deviner : il compare plusieurs types de cerveaux artificiels (modèles) et choisit automatiquement le plus précis.

---

## 🚀 Le Pipeline Complet : De la donnée brute à la prédiction

Voici comment les données circulent dans le système :

### 1. 🧹 Nettoyage des Données (Data Cleaning)
Le code regarde votre fichier Excel/CSV et répare les erreurs :
- **Trous dans les données** : Si une note manque, il met la moyenne.
- **Doublons** : Il supprime les lignes répétées pour ne pas fausser les résultats.
- **Anomalies** : Il détecte les valeurs bizarres (ex: 100 heures d'étude par jour).

### 2. 🧠 Ingénierie des Caractéristiques (Feature Engineering)
Le système crée de nouvelles informations intelligentes à partir des données de base :
- **Indice de Stress Académique** : Une formule magique qui combine les heures d'étude élevées avec un manque de sommeil et une faible assiduité.
- **Score d'Accès Numérique** : Combine l'accès à Internet et aux ressources pour voir si l'étudiant a les outils pour réussir.

### 3. ⚙️ Préparation (Preprocessing)
Avant de donner les données à l'IA, on les transforme :
- **Standardisation** : On met toutes les valeurs sur la même échelle (pour que les heures d'étude ne "pèsent" pas plus que le pourcentage d'assiduité).
- **Encodage** : On transforme les mots (ex: "Oui", "Non", "Homme", "Femme") en chiffres que l'ordinateur comprend.

### 4. 🏆 La Compétition des Modèles (Hybrid AI)
Le script `train.py` lance une compétition entre plusieurs modèles :
- **ML Classique** : Régression Linéaire, Random Forest (Forêt Aléatoire), Gradient Boosting.
- **Deep Learning** : Trois architectures de réseaux de neurones (Simples et Complexes) créées avec TensorFlow/Keras.
Le système calcule l'erreur (RMSE) de chaque modèle et sauvegarde **le champion** dans `models/model.pkl`.

---

## 📁 Structure du Projet (Organisation)

```text
.
├── backend/                # Le "Cerveau" (API FastAPI) qui fait les calculs.
├── frontend/               # Le "Visage" (Dashboard Streamlit) que vous utilisez.
├── src/                    # La "Bibliothèque" centrale contenant toute la logique.
├── deep_learning/          # Les plans de construction des réseaux de neurones.
├── models/                 # Là où le modèle champion est stocké après l'entraînement.
├── scripts/                # Outils de nettoyage et scripts d'analyse CRISP-DM.
├── data/                   # Vos données (Fichier CSV).
├── train.py                # Le chef d'orchestre qui entraîne et choisit le meilleur modèle.
├── requirements.txt        # La liste des ingrédients (bibliothèques Python) nécessaires.
├── docker-compose.yml      # Pour lancer tout le système en un clic avec Docker.
└── .env.example            # Modèle pour vos variables de configuration.
```

---

## 🛠️ Installation et Lancement (Pas à pas)

### Option A : Lancement Local (Recommandé pour le dev)

1.  **Préparer l'environnement** :
    ```bash
    # Créer un environnement virtuel
    py -3.11 -m venv venv
    # L'activer (Windows)
    venv\Scripts\activate
    # Installer les bibliothèques
    pip install -r requirements.txt
    ```

2.  **Entraîner l'IA** :
    ```bash
    py -3.11 train.py
    ```
    *Cela va lire les données, créer le modèle champion et générer des graphiques de performance.*

3.  **Lancer le Backend** :
    ```bash
    py -3.11 -m uvicorn backend.main:app --reload
    ```

4.  **Lancer le Frontend** :
    ```bash
    py -3.11 -m streamlit run frontend/app.py
    ```

### Option B : Lancement Docker (Le plus simple et pro)

Si vous avez Docker installé, une seule commande suffit :
```bash
docker-compose up --build
```
- **Dashboard** : [http://localhost:8501](http://localhost:8501)
- **Documentation API** : [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🎨 Le Dashboard "Cognitive Core"

Le frontend Streamlit a été conçu pour être **premium** :
- **🔮 Synthèse** : Entrez vos données et obtenez une prédiction instantanée avec des graphiques radar et des jauges.
- **🏗️ Architecture** : Visualisez quel modèle a gagné la compétition et voyez ses courbes d'apprentissage.
- **🧬 Pipeline** : Comprenez exactement comment vos données ont été transformées étape par étape.

---

## 📝 Historique des Changements Majeurs (Ce que j'ai fait)

1.  **Réorganisation Totale** : J'ai structuré le projet de manière modulaire (src, backend, frontend) pour qu'il soit propre et facile à maintenir.
2.  **Fusion des Dépendances** : Un seul `requirements.txt` pour tout le projet.
3.  **IA Hybride** : Ajout de réseaux de neurones profonds (TensorFlow) en plus des modèles Scikit-Learn.
4.  **Production Ready** : Ajout de fichiers Docker et support des variables d'environnement (`.env`).
5.  **Interface de Luxe** : Création d'un dashboard Streamlit avec un design "Glassmorphism" et des animations fluides.
6.  **Sécurité** : Ajout d'un `.gitignore` pour ne pas envoyer vos fichiers temporaires ou secrets sur GitHub.

---
*Projet Semestriel 2026 - Développé avec excellence par Louay Ben Mansour.*
