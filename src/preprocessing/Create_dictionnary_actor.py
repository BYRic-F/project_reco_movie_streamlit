import pandas as pd
import duckdb
import os

path_v2 = os.path.join('data', 'processed', 'dataframe_v2.csv')
path_v3 = os.path.join('data', 'processed', 'dataframe_v3.csv')

conn = duckdb.connect()

def SQL(string) :
  return conn.execute(string).fetchdf()


#Load database imdb name.basics.tsv.gz
df_name_basics_imdb = SQL(
    "SELECT * FROM read_csv_auto('https://datasets.imdbws.com/name.basics.tsv.gz', delim='\t')"
)
print('database loaded')

#Create dictionnary id - name

dict_id_name = df_name_basics_imdb.set_index('nconst')['primaryName'].to_dict()
print('dictionnary created')
print(dict_id_name['nm0000206'])  

#import dataframe_v2
dataframe_v2 = pd.read_csv(path_v2)
print('dataframe_v2 imported')

#transform actor id to actor name in dataframe_v2

colonnes = ['directors', 'writers', 'actor', 'actress' , 'producer'	]

def transform_list_ids_to_names(id_list):
    if isinstance(id_list, list):
        # Remplacer chaque ID par le nom si présent dans le dictionnaire, sinon garder l'ID
        return [dict_id_name.get(i) for i in id_list]
    return id_list


for col in colonnes : 
  dataframe_v2[col] = dataframe_v2[col].apply(lambda x: x.split(',') if pd.notnull(x) else x) #split string to list
  dataframe_v2[col] = dataframe_v2[col].apply(transform_list_ids_to_names) #transform id to name using the dictionnary
  
print('actor id transformed to actor name in dataframe_v2')

#export new dataframe_v3
dataframe_v2.to_csv(path_v3, index = False)
print('dataframe_v3 exported')

# End of the script



