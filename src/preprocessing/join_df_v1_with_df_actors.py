import pandas as pd
import os

path_parquet = os.path.join('data', 'raw', 'actors_producers_pivot.parquet')
path_csv_base = os.path.join('data', 'processed', 'dataframe_v1.csv')
path_output = os.path.join('data', 'processed', 'dataframe_v2.csv')

actors_producers = pd.read_parquet(path_parquet)
print(actors_producers.head(5))

dataframe_base =pd.read_csv(path_csv_base)
print(dataframe_base.head(5))
print(dataframe_base.shape)

dataframe_finale = pd.merge(dataframe_base, actors_producers, how='left', on='tconst')
print(dataframe_finale.head(5))
print(dataframe_finale.shape)

dataframe_finale.to_csv(path_output, index=False)