import duckdb
import urllib.request
import pyarrow
import os
print(pyarrow.__version__)
print('finish')

output_dir = os.path.join('data', 'raw')
os.makedirs(output_dir, exist_ok=True)

url = "https://datasets.imdbws.com/title.principals.tsv.gz"
local_file = "title.principals.tsv.gz"
urllib.request.urlretrieve(url, local_file)

conn = duckdb.connect()

df_actors = conn.execute(f"""
    SELECT tconst, nconst, category
    FROM read_csv_auto('{local_file}', delim='\t')
    WHERE category IN ('actor','actress','producer')
""").fetchdf()
print("Chargement DuckDB terminé, nb lignes:", len(df_actors))
print(df_actors.head())

df_actors_grouped = (
    df_actors
    .groupby(['tconst', 'category'])['nconst']
    .apply(lambda x: ','.join(x))
    .reset_index()
)
print("Groupement terminé, nb lignes:", len(df_actors_grouped))

print(df_actors_grouped.head())

df_actors_pivot = df_actors_grouped.pivot(index='tconst', columns='category', values='nconst').reset_index()

print("Pivot terminé, nb lignes:", len(df_actors_pivot))
print(df_actors_pivot.head())


file_path = os.path.join(output_dir, 'actors_producers_pivot.parquet')
df_actors_pivot.to_parquet(file_path)

print("Fichier actors_producers_pivot.parquet écrit avec succès.")

