# Guide : étapes du projet, figures et code

Ce document explique **dans l’ordre** ce que fait le projet *Prédiction du risque d’échec scolaire*, **chaque figure** produite (quand les graphiques sont activés), et **où se trouve le code** correspondant.

---

## 1. Arborescence utile

| Emplacement | Rôle |
|-------------|------|
| `data/student_performance_factors.csv` | Données brutes |
| `src/config.py` | Chemins (`RAW_DATA_PATH`, `OUTPUTS_DIR`, …) et constantes (seuil 60, listes de colonnes) |
| `src/encoding.py` | Encodage ordinal + one-hot |
| `src/feature_engineering.py` | Chargement, nettoyage, variables dérivées, matrice **X** / **y** |
| `src/eda_plotting.py` | Tous les graphiques EDA + boîtes à moustaches du nettoyage + graphiques d’ingénierie |
| `src/ml_utils.py` | Sélection de variables (corrélations + Random Forest), résumé final |
| `src/pipeline.py` | Enchaînement des étapes |
| `scripts/run_pipeline.py` | Lancement **avec** figures (PNG dans `outputs/`) |
| `scripts/run_pipeline_no_figures.py` | Lancement **sans** figures (même calculs, mêmes CSV) |
| `outputs/processed/` | `prepared_X.csv`, `prepared_y.csv`, `dataset_with_engineered_features.csv` |

Le paramètre commun est `generate_figures` dans `run_full_pipeline(..., generate_figures=True|False)`.

---

## 2. Ordre des étapes (pipeline)

1. **Chargement** : lecture du CSV, conservation des 20 colonnes « source », suppression des colonnes déjà calculées dans le fichier (pour tout recalculer de façon cohérente).  
   **Code :** `load_raw_table()` dans `src/feature_engineering.py`.

2. **Nettoyage** : valeurs manquantes, doublons, types, comptage d’outliers (IQR), optionnellement **figure 01**.  
   **Code :** `data_cleaning()` dans `src/feature_engineering.py` ; graphiques via `plot_cleaning_boxplots()` dans `src/eda_plotting.py`.

3. **EDA** : univarié, bivarié, multivarié, optionnellement **figures 02 à 04**.  
   **Code :** `exploratory_data_analysis()` dans `src/eda_plotting.py`.

4. **Ingénierie** : création des colonnes dérivées + statistiques descriptives, optionnellement **figure 05**.  
   **Code :** `run_feature_engineering_section()` dans `src/feature_engineering.py` ; graphiques via `plot_engineered_features()` dans `src/eda_plotting.py`.

5. **Préparation modèle** : cible `Risk`, encodages, standardisation des numériques, construction de **X** et **y**.  
   **Code :** `build_modeling_matrix()` dans `src/feature_engineering.py`.

6. **Sélection de variables** : corrélations avec `Risk`, forêt aléatoire, liste des variables retenues, optionnellement **figure 06**.  
   **Code :** `feature_selection()` dans `src/ml_utils.py`.

7. **Synthèse** : facteurs clés + interprétation métier (texte console).  
   **Code :** `presentation_summary()` dans `src/ml_utils.py`.

8. **Export** : enregistrement des CSV dans `outputs/processed/` (toujours exécuté, avec ou sans figures).

---

## 3. Définitions importantes (pour lire le code)

- **`Exam_Score`** : note continue (0–100).  
- **`Risk`** : `1` si `Exam_Score < 60`, sinon `0` (échec / réussite selon le seuil fixé dans `src/config.py` : `FAIL_THRESHOLD = 60`).  
- **Fuite d’information** : `Exam_Score` n’est **pas** placé dans **X** lorsqu’on prédit `Risk`, car la cible binaire est dérivée directement de la note.

---

## 4. Explication de chaque figure (fichier PNG)

Les fichiers sont créés dans `outputs/` **uniquement** si `generate_figures=True` (comportement par défaut de `run_pipeline.py`).

### Figure `01_boxplots_numeric_outliers.png`

- **Quand** : après le nettoyage, dans `data_cleaning`.  
- **Contenu** : une boîte à moustaches par variable **numérique** présente à ce stade (heures d’étude, assiduité, sommeil, etc.) + **`Exam_Score`**.  
- **Intérêt** : repérer visuellement médiane, dispersion et **valeurs extrêmes** (points au-delà de 1,5 × IQR).  
- **Décision projet** : les outliers sont **recensés** mais en général **non supprimés** (peuvent correspondre à de vrais profils).  
- **Code** : `plot_cleaning_boxplots()` dans `src/eda_plotting.py`.

### Figure `02_univariate_hist_exam_hours_attendance.png`

- **Quand** : phase EDA, bloc univarié.  
- **Contenu** : histogrammes (+ courbe KDE) de **`Exam_Score`**, **`Hours_Studied`**, **`Attendance`** ; boîte à moustaches horizontale pour **`Exam_Score`**.  
- **Intérêt** : forme des distributions (symétrie, queues), position des notes par rapport au seuil 60.  
- **Code** : début de `exploratory_data_analysis()` dans `src/eda_plotting.py`.

### Figure `02b_bar_risk_distribution.png`

- **Contenu** : diagramme en barres des effectifs **`Risk = 0`** vs **`Risk = 1`**.  
- **Intérêt** : visualiser le **déséquilibre des classes** (souvent beaucoup plus de « pass » que d’« échecs »).  
- **Code** : même fonction `exploratory_data_analysis()`.

### Figure `03_bivariate_score_relationships.png`

- **Contenu** :  
  - nuages de points **`Exam_Score` vs `Hours_Studied`** et **`Exam_Score` vs `Attendance`**, couleur = `Risk` ; ligne horizontale à **60** ;  
  - boîtes à moustaches **`Exam_Score`** selon **`Parental_Education_Level`**.  
- **Intérêt** : liens simples entre comportement / contexte et performance ; repérage des points sous le seuil.  
- **Code** : section « B. Bivariate » dans `exploratory_data_analysis()`.

### Figure `04_correlation_heatmap.png`

- **Contenu** : matrice de corrélation (variables numériques + ordinales **label-encodées** + `Exam_Score`).  
- **Intérêt** : voir quelles variables **varient ensemble** et lesquelles sont le plus liées à la note.  
- **Code** : section « C. Multivariate » dans `exploratory_data_analysis()`.  
- **Sans figures** : un **aperçu texte** des plus fortes corrélations absolues avec `Exam_Score` est affiché en console.

### Figure `05_engineered_features.png`

- **Contenu** : histogramme de **`Academic_Stress_Index`** ; barres des effectifs par valeur de **`Digital_Access_Score`**.  
- **Intérêt** : vérifier que les formules produisent des distributions exploitables.  
- **Formules (rappel)** :  
  - `Academic_Stress_Index = Hours_Studied * (100 - Attendance) / max(Sleep_Hours, 0.5)`  
  - `Digital_Access_Score` : combinaison discrète d’**Internet** (oui/non) et d’**Access_to_Resources** (bas/moyen/haut).  
- **Code** : `plot_engineered_features()` dans `src/eda_plotting.py` ; formules dans `add_engineered_columns()` dans `src/feature_engineering.py`.

### Figure `06_feature_importance_rf.png`

- **Contenu** : barres horizontales des **15 plus grandes importances** de la forêt aléatoire (`RandomForestClassifier`) entraînée pour prédire **`Risk`**.  
- **Intérêt** : repérer les variables qui **réduisent le plus l’impureté** dans les arbres (effets non linéaires possibles).  
- **Code** : fin de `feature_selection()` dans `src/ml_utils.py`.  
- **Complément** : les **corrélations absolues** avec `Risk` sont aussi imprimées ; la sélection finale combine **rang de corrélation** et **rang d’importance RF**.

---

## 5. Fichiers CSV générés (avec ou sans figures)

| Fichier | Description |
|---------|-------------|
| `outputs/processed/prepared_X.csv` | Matrice des prédicteurs (encodés + numériques standardisés). |
| `outputs/processed/prepared_y.csv` | Colonne **`Risk`**. |
| `outputs/processed/dataset_with_engineered_features.csv` | Table nettoyée + **`Risk`** + colonnes ingénérées. |

Ces exports sont produits par `src/pipeline.py` à la fin de `run_full_pipeline()`, **même** avec `scripts/run_pipeline_no_figures.py`.

---

## 6. Module par module (résumé du code)

### `src/config.py`

Centralise les chemins (`PROJECT_ROOT`, `DATA_DIR`, `OUTPUTS_DIR`, `PROCESSED_DIR`), le nom du CSV, le seuil d’échec, les listes **`RAW_FEATURE_COLUMNS`**, **`ORDINAL_ORDER`**, **`NOMINAL_COLUMNS`**, **`NUMERIC_COLUMNS`**, et `ensure_directories()`.

### `src/encoding.py`

- **`label_encode_ordinals`** : transforme les colonnes ordinales en entiers 0, 1, 2, … selon un ordre fixé (évite d’inventer un ordre artificiel pour les modèles qui supposent une monotonie).  
- **`one_hot_nominals`** : crée des colonnes binaires / indicatrices pour les variables nominales (`pd.get_dummies(..., drop_first=True)`).

### `src/feature_engineering.py`

- **`load_raw_table`** : charge et filtre les colonnes.  
- **`add_engineered_columns`** : ajoute `Academic_Stress_Index` et `Digital_Access_Score`.  
- **`data_cleaning`** : imputation, doublons, types, IQR, puis boîtes à moustaches si `generate_figures`.  
- **`build_modeling_matrix`** : chaîne encodage → **ColumnTransformer** (imputation médiane + **StandardScaler** sur les numériques) → **DataFrame** `X_scaled` et série `y`.

### `src/eda_plotting.py`

Contient toute la logique **matplotlib / seaborn** décrite aux sections 4.1–4.6.

### `src/ml_utils.py`

- **`feature_selection`** : `corrwith`, **`RandomForestClassifier`** (`class_weight='balanced_subsample'`), fusion des rangs, figure optionnelle.  
- **`presentation_summary`** : liste de facteurs + texte d’interprétation métier.

### `src/pipeline.py`

Appelle les fonctions dans l’ordre logique et passe **`generate_figures`** partout où c’est nécessaire.

---

## 7. Comment lancer

```bash
# Avec toutes les figures PNG
python scripts/run_pipeline.py

# Sans figures (plus rapide, adapté CI / serveur)
python scripts/run_pipeline_no_figures.py
```

Dans un notebook ou un script perso :

```python
from src.pipeline import run_full_pipeline

run_full_pipeline(generate_figures=False)
```

---

## 8. Suite possible du projet (hors périmètre actuel)

- Notebook **`02_Modeling.ipynb`** : charger `prepared_X` / `prepared_y`, stratifier selon `Risk`, métriques adaptées au déséquilibre (PR-AUC, etc.).  
- Document LaTeX : `docs/rapport_avancement_projet.tex` (intègre les figures si compilé après une exécution **avec** graphiques).

---

*Document généré pour accompagner le dépôt `ps2final`. Mettez à jour ce guide si vous ajoutez de nouvelles étapes ou figures.*
