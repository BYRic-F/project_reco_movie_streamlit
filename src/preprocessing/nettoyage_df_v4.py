import pandas as pd
import os

path_v4 = os.path.join('data', 'processed', 'dataframe_v4.csv')
path_csv = os.path.join('data', 'processed', 'dataframe_ML_final.csv')
path_pkl = os.path.join('data', 'processed', 'dataframe_ML_final.pkl')

dataframe_v4 = pd.read_csv(path_v4)
print("DataFrame chargé avec succès")


# Conversion des colonnes 'actor' et 'actress' de chaînes de caractères en listes
dataframe_v4['actor'] = dataframe_v4['actor'].apply(
    lambda x: eval(x) if pd.notnull(x) else [])

dataframe_v4['actress'] = dataframe_v4['actress'].apply(
    lambda x: eval(x) if pd.notnull(x) else [])

#Conversion des colonnes directors de chaînes de caractères en listes
dataframe_v4['directors'] = dataframe_v4['directors'].apply(
    lambda x: eval(x) if pd.notnull(x) else [])

# conversion des colonnes writers de chaînes de caractères en listes
dataframe_v4['writers'] = dataframe_v4['writers'].apply(
    lambda x: eval(x) if pd.notnull(x) else [])

# conversion colonnes production_companies de chaînes de caractères en listes
dataframe_v4['production_companies_name'] = dataframe_v4['production_companies_name'].apply(
    lambda x: eval(x) if pd.notnull(x) else [])

#conversion colonnes producer de chaînes de caractères en listes
dataframe_v4['producer'] = dataframe_v4['producer'].apply(
    lambda x: eval(x) if pd.notnull(x) else []
)

#conversion colonnes genres de chaînes de caractères en listes
dataframe_v4['genres'] = dataframe_v4['genres'].apply(
    lambda x: x.split(",") if pd.notnull(x) else []
)

print("conversion des colonnes en liste effectuée")

# conversion colonnes tconst, primatryTitle, originalTitle, overview en chaînes de caractères, poster_path, original_language en chaînes de caractères
dataframe_v4['tconst'] = dataframe_v4['tconst'].astype('string')
dataframe_v4['primaryTitle'] = dataframe_v4['primaryTitle'].astype('string') 
dataframe_v4['originalTitle'] = dataframe_v4['originalTitle'].astype('string')
dataframe_v4['overview'] = dataframe_v4['overview'].astype('string')
dataframe_v4['poster_path'] = dataframe_v4['poster_path'].astype('string')
dataframe_v4['original_language'] = dataframe_v4['original_language'].astype('string')
dataframe_v4['id'] = dataframe_v4['id'].astype('int')

print("conversion des colonnes en chaînes de caractères effectuée")

#conversion des colonnes budget , revenue en int64
for col in ['budget', 'revenue']:
    dataframe_v4[col] = pd.to_numeric(dataframe_v4[col], errors='coerce').astype('int64')

print("conversion des colonnes 'budget' et 'revenue' en int64 effectuée")

# Création de la nouvelle colonne 'actor_actress' en combinant les listes des colonnes 'actor' et 'actress'
dataframe_v4['actor_actress'] = dataframe_v4[['actor', 'actress']].apply(
    lambda x: sum(x.tolist(), []), axis=1)

print("Colonne 'actor_actress' créée avec succès")

print(dataframe_v4[['actor_actress', 'actor', 'actress']].head(5))

#drop colonnes actor et actress
dataframe_v4 = dataframe_v4.drop(columns=(['actor', 'actress']))

print("Colonnes 'actor' et 'actress' supprimées")

#nouvel ordre des colonnes
nouvel_ordre =['tconst',
                'id',
                'primaryTitle',
                'originalTitle',
                'genres', 
                'startYear',
                'runtimeMinutes',
                'averageRating', 
                'numVotes', 
                'popularity',
                'actor_actress',
                'producer',
                'writers',
                'directors',
                'production_companies_name',
                'budget', 
                'revenue', 
                'original_language', 
                'overview', 
                'poster_path']

#changement de l'ordre des colonnes
dataframe_v4 = dataframe_v4[nouvel_ordre]

print("Ordre des colonnes modifié")

# Sauvegarde du DataFrame nettoyé dans un nouveau fichier CSV
dataframe_v4.to_csv(path_csv, index=False)

#Sauvegarde en fichier pickle pour pouvoir garder les bons formats dee colonnes
dataframe_v4.to_pickle(path_pkl)
print("DataFrame nettoyé sauvegardé avec succès dans 'dataframe_ML_final.pkl'")

