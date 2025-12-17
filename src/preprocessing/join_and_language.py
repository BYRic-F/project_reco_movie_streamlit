import os
import pandas as pd
path_v3 = os.path.join('data', 'processed', 'dataframe_v3.pkl')
output_path = os.path.join('data', 'processed', 'dataframe_v4.pkl')
path_tmdb = os.path.join('data', 'raw', 'tmdb_full.csv')

#import dataframe v3
dataframe_v3 = pd.read_pickle(path_v3)
dataframe_v3.head()


print(f" shape = {dataframe_v3.shape}")




#import tmdb

data_tmdb = pd.read_csv(path_tmdb)


print(data_tmdb.head(5).to_markdown())

print('----------------------------------------------------------')

#verif des Nan pour le drop des colonens
print(f" Nan : {data_tmdb.isna().sum()}")

print('----------------------------------------------------------')

#liste des colonnes pour voir lesquelles droper
print(f"columns = {data_tmdb.columns.to_list()}")


#créationd de la liste a droper
columns_to_drop = ['adult',
 'genres',
 'homepage',
  'backdrop_path',
 'original_title',
 'production_countries',
 'release_date',
 'runtime',
 'spoken_languages',
 'status',
 'title',
 'video',
 'vote_average',
 'vote_count',
 'tagline',
 'production_companies_country']



#drop des colonnes inutiles
data_tmdb_final = data_tmdb.drop(columns = columns_to_drop)
print(data_tmdb_final.head(5).to_markdown())


#Rename des colonnes pour le merge
data_tmdb_final = data_tmdb_final.rename(columns = {'imdb_id' : 'tconst'})



#Merge final imdb tmdb
dataframe_merge = pd.merge(dataframe_v3, data_tmdb_final, how = 'left', on = 'tconst')


print(dataframe_merge.head(5).to_markdown())


#verif du merge
print(f"columns = {dataframe_merge.columns}")

print('----------------------------------------------------------')

#verificationd des valeurs nulles
print(f"Nan = {dataframe_merge.isna().sum()}")

print('----------------------------------------------------------')

# Etude des valeurs nulles
df_mask =  dataframe_merge.isnull().any(axis = 1)
dataframe_row_empty = dataframe_merge[df_mask]


print(dataframe_row_empty.sample(10).to_markdown())
#- > il s'avère que beaucoup de films étrangers low cost ait passer le filtre (indien etc)
# Pays très peuplé (donc bcp de vote) avec une culture du cinema très différente de la notre


#Etude des différentes langues presentes
print(f"language : {dataframe_merge['original_language'].value_counts()}")

print('----------------------------------------------------------')

#Creation de la liste des langues que l'on souhaite garder (occidental)
iso_codes_europe_anglais = [
    'en',  # Anglais (Royaume-Uni, États-Unis, Irlande)
    'fr',  # Français (France, Belgique, Luxembourg)
    'de',  # Allemand (Allemagne, Autriche)
    'es',  # Espagnol (Espagne)
    'it',  # Italien (Italie)
    'nl',  # Néerlandais (Pays-Bas, Belgique)
    'pt',  # Portugais (Portugal)
    'fi',  # Finnois (Finlande)
    'ga',  # Irlandais (Irlande)
      'ja',  #Japonais
  ]

print(iso_codes_europe_anglais)

print('----------------------------------------------------------')

#création du mask
mask_langue = dataframe_merge['original_language'].isin(iso_codes_europe_anglais)

#Creation du df final
df_filtre = dataframe_merge[mask_langue]


#Shape on commence a être dans les normes demandées
print(f" shape : {df_filtre.shape}")
print('----------------------------------------------------------')

print(f" Nan :{df_filtre.isna().sum()}")

print('----------------------------------------------------------')
#drop des films sans poster path ni overview
df_filtre = df_filtre.dropna(subset=['poster_path', 'overview'])
df_filtre = df_filtre.reset_index(drop= True)

print(f" Nan :{df_filtre.isna().sum()}")

print('----------------------------------------------------------')
# On est bien
print(f" shape : {df_filtre.shape}")
print('----------------------------------------------------------')


print(df_filtre.head(5).to_markdown())


# Export du fichier v 4

df_filtre.to_pickle(output_path)


