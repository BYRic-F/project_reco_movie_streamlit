import pandas as pd
import os
from sklearn.preprocessing import FunctionTransformer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors


path_df_ml = os.path.join('data', 'processed', 'dataframe_ready_for_ML.pkl')
df = pd.read_pickle(path_df_ml)
print("Import réussi")

print(df.head(5).to_markdown())
print('--------------------------------------------------------')

print(f" columns = {df.columns}")

#selection des colonnes
cols_knn = [
    #id tmdb
    'id', 

    # texte
    'genres_clean',
    'directors_clean',
    'actor_actress_clean',
    'NLP',  

    # num
    'startYear',
    'averageRating',
    'numVotes'
]

df_nn = df[cols_knn].copy()
# on verifie que pas de Nan
print(df_nn.isna().sum())

# Créer un transformateur personnalisé pour pondérer les features
def weight_features(X, weight):
  return X * weight

# Appliquer des poids différents à chaque type de feature
preprocessor = ColumnTransformer(
    transformers=[  
            
            # Text
        ('genres', Pipeline([
            ("tfifd", TfidfVectorizer()),
            ("weight", FunctionTransformer(weight_features, kw_args={"weight": 2.2}))
            ]), 'genres_clean'),
        
        ('directors', Pipeline([
            ("tfifd", TfidfVectorizer(min_df=2)),
            ("weight", FunctionTransformer(weight_features, kw_args={"weight": 1.8}))
            ]),'directors_clean'),
            
        
        ('actors', Pipeline([
            ("tfifd", TfidfVectorizer(min_df=3)),
            ("weight", FunctionTransformer(weight_features, kw_args={"weight": 2}))
            ]), 'actor_actress_clean'),
        

        ('NLP', Pipeline([
            ("tfifd", TfidfVectorizer(min_df=5, max_features=5000, ngram_range=(1,2))),
            ("weight", FunctionTransformer(weight_features, kw_args={"weight": 1.5}))
            ]),'NLP'),
        
        
        # Chiffres
        ('num', Pipeline([
            ('scaler', MinMaxScaler()),
            ('weight', FunctionTransformer(weight_features, kw_args={'weight': 1}))
            ]), ['startYear', 'averageRating', 'numVotes'])
    ],
)

print(preprocessor)

pipeline = Pipeline([
    ("prep" , preprocessor ),
    ("nn", NearestNeighbors(metric = "cosine", n_neighbors = 5))
])

print(pipeline)
print("Début du train")
pipeline.fit(df_nn)
print("Train reussi")

distances, indices = pipeline[1].kneighbors(
    pipeline[0].transform(df_nn.iloc[[5000]])
)

print(df_nn.iloc[indices[0]][[ 
    'id', 
    # texte
    'genres_clean',
    'directors_clean',
    'actor_actress_clean',
    # num
    'startYear',
    'averageRating',
    'numVotes'
]].to_markdown())


