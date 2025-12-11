

import duckdb

conn = duckdb.connect()

def SQL(string) :
    return conn.execute(string).fetchdf()



df_name_basics_imdb = SQL(
    "SELECT * FROM read_csv_auto('https://datasets.imdbws.com/name.basics.tsv.gz', delim='\t'), Limit 5000"
)
print(df_name_basics_imdb.head(5).to_markdown())

print('---------------------------------------------')

df_akas_imdb = SQL(
    "SELECT * FROM read_csv_auto('https://datasets.imdbws.com/title.akas.tsv.gz', delim='\t'), Limit 5000"
)

print(df_akas_imdb.sample(5).to_markdown())

print('---------------------------------------------')

df_title_basics = SQL(
    "SELECT * FROM read_csv_auto('https://datasets.imdbws.com/title.basics.tsv.gz', delim='\t'), Limit 5000"
)
print(df_title_basics.head(5).to_markdown())

print('---------------------------------------------')

df_crew =  SQL(
    "SELECT * FROM read_csv_auto('https://datasets.imdbws.com/title.crew.tsv.gz', delim='\t'), Limit 5000"
)
print(df_crew.head(5).to_markdown())

print('---------------------------------------------')

df_episode = SQL(
    "SELECT * FROM read_csv_auto('https://datasets.imdbws.com/title.episode.tsv.gz', delim='\t'), Limit 5000"
)
print(df_episode.head(5).to_markdown())

print('---------------------------------------------')

df_principals = SQL(
    "SELECT * FROM read_csv_auto('https://datasets.imdbws.com/title.principals.tsv.gz', delim='\t'), Limit 5000"
)
print(df_principals.sample(20).to_markdown())

print('---------------------------------------------')

df_ratings = SQL(
    "SELECT * FROM read_csv_auto('https://datasets.imdbws.com/title.ratings.tsv.gz', delim='\t'), Limit 5000"
)
print(df_ratings.head(5).to_markdown())

print('---------------------------------------------')

import pandas as pd

df_tmbd = pd.read_csv(r'C:\Users\frede\Vs_Code\dossier_projets\projet_test\data\raw\tmdb_full.csv')


print(df_tmbd.head(5).to_markdown())


