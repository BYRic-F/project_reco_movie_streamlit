
import duckdb
import pandas as pd
import os

output_path = os.path.join('data', 'processed', 'dataframe_v1.csv')

conn = duckdb.connect()

def SQL(string) :
  return conn.execute(string).fetchdf()


# Recuperation de la BDD principale + jointure facile

df_total = SQL(
    """SELECT tconst,
          	primaryTitle,
            originalTitle,
            startYear,
            runtimeMinutes,
            genres,
            directors,
            writers,
            averageRating,
            numVotes
    FROM read_csv_auto('https://datasets.imdbws.com/title.basics.tsv.gz', delim='\t') basics
    JOIN read_csv_auto('https://datasets.imdbws.com/title.crew.tsv.gz') crew USING (tconst)
    JOIN read_csv_auto('https://datasets.imdbws.com/title.ratings.tsv.gz', delim='\t') USING (tconst)
    WHERE TRY_CAST(basics.startYear AS INTEGER) > 1990
    AND titleType = 'movie'
    AND TRY_CAST(averageRating AS Float) > 6.5
    AND TRY_CAST(runtimeMinutes AS INTEGER) > 75
    AND TRY_CAST(numVotes AS INTEGER)  > 500

    ;
     """
)


print(df_total.head(5))


print(df_total.shape)


# Export du fichier temporaire v1
df_total.to_csv(output_path, index=False)


