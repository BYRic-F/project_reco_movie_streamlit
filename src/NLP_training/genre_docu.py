import pandas as pd
import numpy as np
import os
pkl_path = os.path.join('data', 'processed', 'dataframe_ML_final.pkl')

df = pd.read_pickle(r'C:\Users\frede\Vs_Code\dossier_projets\projet_test\data\processed\dataframe_ML_final.pkl')
print('import terminé')

def classer_documentaires_par_genre_secondaire(df, colonne_genres='genres', colonne_titre='originalTitle'):
    """
    filtre les films documentaires et crées une deuxieme colonne avec les autres genres
    
    Return :
        pd.DataFrame: DataFrame filtré avec la colonne 'genres_secondaires'.
    """
    
    #recup les genres qui ont docmentaires
    df_documentaires = df[df[colonne_genres].apply(lambda genres: 'Documentary' in genres)]
        
    print(f" filtrage fini : {len(df_documentaires)} avec un genre 'Documentary'.")
    
    #creation 2e colonne
    # isoler les genres secondaires (prendre toute la liste sauf)
    df_documentaires['genres_secondaires'] = df_documentaires[colonne_genres].apply(
        lambda genres: [g for g in genres if g != 'Documentary'])
    
    # Gérer les cas où le documentaire n'a aucun autre genre
    df_documentaires['genres_secondaires_affinee'] = df_documentaires['genres_secondaires'].apply(
        lambda g: g if g else ['Other']
    )
    
    # --- ÉTAPE 3 : Sélection des colonnes (CORRECTION APPLIQUÉE ICI) ---
    df_resultat = df_documentaires[[colonne_titre, colonne_genres, 'genres_secondaires_affinee']]
    
    return df_resultat

# application fonction sur df :
df_documentaires_classes = classer_documentaires_par_genre_secondaire(df, colonne_genres='genres', colonne_titre='originalTitle')

print("\nApercu resultat")
print(df_documentaires_classes.head(10))

# explde pour voir
df_exploded_secondaires = df_documentaires_classes.explode('genres_secondaires_affinee')
comptage_secondaire = df_exploded_secondaires['genres_secondaires_affinee'].value_counts()
print("\n--- Distribution des Genres Secondaires ---")
print(comptage_secondaire.head(10))

#beaucoup de other
