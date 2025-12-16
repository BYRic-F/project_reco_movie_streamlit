import pandas as pd
import os


df_final = pd.read_pickle(r'C:\Users\frede\Vs_Code\dossier_projets\projet_test\data\processed\dataframe_ML_final.pkl')

df_title = pd.read_csv(r'C:\Users\frede\Vs_Code\dossier_projets\projet_test\data\raw\title_fr.csv')

print('import des df')
print(f"le dataframe dataframe_ML_finale a {df_final.shape[0]} lignes")

#certains films sont en doublons
df_title_clean = df_title.sort_values(by=['titleId', 'ordering'], ascending=[True, True])
#On trie grace a ordering et on garde la premiere
df_title_clean = df_title_clean.drop_duplicates(subset=['titleId'], keep='first')
df_title_clean = df_title_clean.rename(columns={'title': 'title_fr_akas'})


print("Debut du merge")

df_final = pd.merge(df_final, df_title_clean, how = 'left' , left_on= 'tconst', right_on = "titleId")

# on remplace les valeurs manquantes "titre en francais" par le titre inital

df_final['title_final'] = df_final['title_fr_akas'].fillna(df_final['primaryTitle'])


print("merge et netooyage fini")

print(df_final.columns)

df_final = df_final.drop(columns = ['primaryTitle', 'originalTitle','averageRating', 'numVotes', 'popularity',
        'writers', 'directors',
        'production_companies_name', 'budget', 'revenue','overview','Unnamed: 0' ,'titleId',  'region', 'ordering', 'title_fr_akas'])

print(df_final.columns)
print(f" le df steamlit a {df_final.shape[0]} lignes")

print(f" verif nombre de valeurs nulles sur title_final {df_final['title_final'].isnull().sum()}")

df_final.to_parquet('data/processed/dataframe_streamlit.parquet')