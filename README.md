# Système de Recommandation de Films

PicquePoule est une application web interactive de recommandation cinématographique exploitant une chaîne de traitement de données (ETL) hybride (SQL/Pandas) et des algorithmes de Machine Learning avancés.

Le projet se distingue par son catalogue rigoureusement filtré (10 000 titres premium) et son double moteur de recommandation : sémantique (NLP) et collaboratif (SVD).


## Défis Techniques & Solutions

- Bases de données lourdes : Passage par DuckDB et SQL pour éviter les crashs RAM rencontrés sur Colab.

- Persistance des données : Utilisation intensive du format Pickle et Parquet pour conserver les types complexes.

- Authentification : Développement d'un flux manuel car l'outil standard ne permettait pas l'envoi direct des mots de passe vers Sheets.

- Stabilité Streamlit : Utilisation de st.session_state et des clés (keys) pour empêcher la disparition des variables lors du rafraîchissement des widgets.

- Compatibilité Python : Gestion précise des versions (downgrade nécessaire pour la bibliothèque surprise).

- Recherche textuelle : Nettoyage des accents et caractères spéciaux pour assurer la correspondance entre la saisie utilisateur et la BDD.

## Installation et Configuration
Ce projet utilise uv pour la gestion des dépendances et de l'environnement virtuel.

### Prérequis
Cloner le dépôt.


⚠️ Télécharger la base de données tmdb, la base de données Movielens et le parquet, et la placer dans un dossier data/raw
lien : https://drive.google.com/file/d/1VB5_gl1fnyBDzcIOXZ5vUSbCY68VZN1v/view?usp=sharing

https://drive.google.com/file/d/1kLFsmvvgNUaZv3MQuGlKujAlYwz_V4oS/view?usp=sharing

https://files.grouplens.org/datasets/movielens/ml-32m.zip

S'assurer que uv est installé.

### Installation des dépendances
```
uv sync
```

### Téléchargement du modèle NLP
Le projet nécessite le modèle de langue anglaise de la librairie SpaCy :

```
uv run python -m spacy download en_core_web_sm
```
## Architecture du Pipeline de Données
Les scripts de transformation se trouvent dans le dossier src/. L'exécution doit suivre l'ordre chronologique ci-dessous pour garantir l'intégrité des données.

### 1. Découverte et Extraction Initiale
- Exploration : src/explore/decouverte_bdd_duckBDD.py

  - Exploration initiale des fichiers sources via DuckDB.

- Création du DataFrame Principal : src/preprocessing/create_df.py

  - Jointures SQL et filtrage sur les tables title.basics, title.crew et title.ratings.

  - Sortie : data/processed/dataframe_v1.pkl

### 2. Enrichissement (Casting et Production)
- Traitement des Principals : src/preprocessing/df_actors.py

  - Import de la table title.principals.

  - Agrégation (groupby) par catégorie et ID de film, puis pivot des données.

  - Sortie : data/processed/actors_producers_pivot.parquet

- Fusion Initiale : src/preprocessing/join_df_v1_with_df_actors.py

  - Jointure entre le dataframe_v1 et le fichier pivot parquet.

  - Sortie : data/processed/dataframe_v2.pkl

### 3. Transformation et Consolidation
- Résolution des Identifiants : src/preprocessing/create_dictionnary_actor.py

  - Remplacement des identifiants alphanumériques (nconst) par les noms réels des personnes (acteurs, actrices, producteurs).

  - Sortie : data/processed/dataframe_v3.pkl

- Fusion IMDB/TMDB et Filtrage : src/preprocessing/join_and_languages.py

  - Jointure finale entre les données IMDB et la base TMDB.

  - Application d'un filtre culturel (langues occidentales).

  - Sortie : data/processed/dataframe_v4.csv

### 4. Nettoyage et Préparation ML
- Standardisation : src/preprocessing/nettoyage_df_v4.py

  - Regroupement des colonnes (fusion actor + actress).

  - Typage des données et réorganisation des colonnes.

  - Sortie : data/processed/dataframe_ML_final.csv (et .pkl)

- Traitement NLP : src/NLP_training/NLP.py

  - Lemmatisation des résumés (overview) via SpaCy.

  - Nettoyage des entités nommées (suppression des espaces pour bradpitt, sciencefiction...) afin d'optimiser la vectorisation.

  - Sortie Finale : data/processed/dataframe_ready_for_ML.pkl


### 5. Modèles de Marchine learning
- Le projet embarque deux moteurs distincts situés dans model/ :

  -Recommandation par Titre (KNN) : Entraîné via src/training_ml/train_nn.py. Il utilise la similarité cosinus sur un mélange de genres, réalisateurs et sémantique (NLP).

- "Surprends-moi !" (SVD) : Entraîné via src/training_ml/ml_svd.py.

  -Innovation : Pour éviter un réentraînement lourd à chaque nouvel utilisateur, le système calcule un profil utilisateur en temps réel et le projette dans l'espace latent du modèle (produit scalaire avec la matrice qi).

### 6. Création de bdd Streamlit

- export_title_language.py et nettoyage_fusion_title.py permettent de récupérer une base de données épurés, en francais avec possibilité de limiter la casse.

## Application Streamlit

L'interface gère l'expérience utilisateur complète :

- Authentification Hybride : Système sur-mesure pour synchroniser les mots de passe hachés avec une base Google Sheets via API.

- Affichage des détails du film via l'API TMDB (affiches, résumés).

- Onboarding : Les nouveaux utilisateurs doivent noter 5 films pour initialiser leur profil IA.

- Moteur de recherche : Filtrage avancé (genres, années, durée), recherche par titre avec nettoyage Unicode ou fonction "Surprends moi".

- Popups de Notation : Utilisation de st.dialog et st.popover pour noter les films vus sans rafraîchir la page inutilement.

- Fonctionnalités distinctes : L'utilisateur enregistré bénéficie d'options supplementaires, comme la recommandation du modèle SVD

- Statistiques : Visualisations de la base de données via Altair.

### Lancement de l'application
Depuis la racine du projet :

```
streamlit run app.py
```

## Structure du Projet



```mon_projet/
├── app.py                     # Point d'entrée Streamlit
├── pyproject.toml             # Gestion des dépendances (uv)
├── style.css                  # Design personnalisé
├── data/
│   ├── raw/                   # IMDB, TMDB, Movielens
│   └── processed/             # Fichiers .pkl et .csv finaux
├── model/                     # Modèles .joblib (SVD, KNN)
└── src/
    ├── create_bdd_streamlit/  # Nettoyage FR et titres
    ├── explore/               # Scripts DuckDB
    ├── NLP_training/          # SpaCy et vectorisation
    ├── preprocessing/         # Pipeline ETL complet
    └── training_ml/           # Scripts d'entraînement des modèles

```
