# Système de Recommandation de Films
Ce projet met en œuvre une chaîne de traitement de données (ETL) complète et une application web interactive pour recommander des films. Il combine des données provenant des bases IMDB et TMDB, traitées via DuckDB et Pandas, et utilise des techniques de Traitement du Langage Naturel (NLP) pour alimenter un modèle de Machine Learning.

## Installation et Configuration
Ce projet utilise uv pour la gestion des dépendances et de l'environnement virtuel.

### Prérequis
Cloner le dépôt.

[!WARNING]
Télécharger la base de données tmdb, et la placer dans un dossier data/raw
lien : https://drive.google.com/file/d/1VB5_gl1fnyBDzcIOXZ5vUSbCY68VZN1v/view?usp=sharing

S'assurer que uv est installé.

### Installation des dépendances
'''
uv sync
'''

### Téléchargement du modèle NLP
Le projet nécessite le modèle de langue anglaise de la librairie SpaCy :

'''uv run python -m spacy download en_core_web_sm'''
## Architecture du Pipeline de Données
Les scripts de transformation se trouvent dans le dossier src/. L'exécution doit suivre l'ordre chronologique ci-dessous pour garantir l'intégrité des données.

### 1. Découverte et Extraction Initiale
- Exploration : src/explore/decouverte_bdd_duckBDD.py

  - Exploration initiale des fichiers sources via DuckDB.

- Création du DataFrame Principal : src/preprocessing/create_df.py

  - Jointures SQL et filtrage sur les tables title.basics, title.crew et title.ratings.

  - Sortie : data/processed/dataframe_v1.csv

### 2. Enrichissement (Casting et Production)
- Traitement des Principals : src/preprocessing/df_actors.py

  - Import de la table title.principals.

  - Agrégation (groupby) par catégorie et ID de film, puis pivot des données.

  - Sortie : data/processed/actors_producers_pivot.parquet

- Fusion Initiale : src/preprocessing/join_df_v1_with_df_actors.py

  - Jointure entre le dataframe_v1 et le fichier pivot parquet.

  - Sortie : data/processed/dataframe_v2.csv

### 3. Transformation et Consolidation
- Résolution des Identifiants : src/preprocessing/create_dictionnary_actor.py

  - Remplacement des identifiants alphanumériques (nconst) par les noms réels des personnes (acteurs, actrices, producteurs).

  - Sortie : data/processed/dataframe_v3.csv

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

  - Sortie Finale : data/processed/dataframe_ready_for_ML.parquet

## Application Streamlit

L'interface utilisateur permet de visualiser les données et d'obtenir des recommandations.

### Lancement de l'application
Depuis la racine du projet :

'''uv run streamlit run app.py'''

## Fonctionnalités

- Système de filtrage avancé par genre, année et note.

- Moteur de recommandation basé sur le contenu (similarité cosinus sur les résumés lemmatisés, casting et métadonnées).

- Affichage des détails du film via l'API TMDB (affiches, résumés).

## Structure du Projet

'''
mon_projet/
├── app.py                  # Point d'entrée de l'application Streamlit
├── pyproject.toml          # Gestion des dépendances (uv)
├── README.md               # Documentation du projet
├── data/
│   ├── raw/                # Données brutes (IMDB .tsv, TMDB .csv)
│   └── processed/          # Fichiers intermédiaires (v1, v2...) et finaux
├── models/                 # Pipeline ML entraîné (.pkl ou .parquet)
└── src/
    ├── explore/            # Scripts d'exploration
    ├── preprocessing/      # Scripts de transformation (create_df, cleaning...)
    ├── NLP_training/       # Scripts NLP et Entraînement modèle
    └── utils/              # Fonctions utilitaires (si applicable)'''