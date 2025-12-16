
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVDpp ,SVD
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.model_selection import GridSearchCV
import os
import joblib

path_movielens_ratings = os.path.join('data', 'raw', 'ratings.csv')
path_links= os.path.join('data', 'raw', 'links.csv')
path_movies = os.path.join('data', 'raw', 'movies.csv')
path_tags = os.path.join('data', 'raw', 'tags.csv')
path_df_ml_final = os.path.join('data', 'processed', 'dataframe_ML_final.csv')

print("Import dataframe")
df_movielens_ratings = pd.read_csv(path_movielens_ratings)
df_dataframe_final = pd.read_csv(path_df_ml_final)
df_links = pd.read_csv(path_links)
df_movies = pd.read_csv(path_movies)
df_tags = pd.read_csv(path_tags)


print(df_movies.head(5).to_markdown())
print('--------------------------------------------------------------------------')
print(df_links.head(5).to_markdown())
print('-------------------------------------------------------------------------')
print(df_tags.head(5).to_markdown())
print('---------------------------------------------------------------------')
print(df_dataframe_final.head(5).to_markdown())
print('---------------------------------------------------------------------------------')
print(df_movielens_ratings.head(5).to_markdown())


# merge pour garder uniquement userId, movieId, rating, tmdbId
print("debut du merge")

df_complete = pd.merge(
    df_movielens_ratings,
    df_links,
    on='movieId',
    how='left'
)
df_complete = df_complete.drop(columns =['movieId','timestamp', 'imdbId'])

print(df_complete.head(5).to_markdown())
print(f"shape : {df_complete.shape[0]}")


# recuperer les id tmtb de notre dataset pour garder les meme films
#creation de la liste pour recuperer toutes les id tmdb
print('On recupere les listes de films')
movies_to_keep = set(df_dataframe_final['id'].unique())

#on filtre pour ne garder que les mêmes films (facilité pour l'insertion dans streamlit)
df_ratings_filtered = df_complete[df_complete['tmdbId'].isin(movies_to_keep)].copy()
# Vérif
print(f"Nombre de lignes avant filtrage : {len(df_complete):,}")
print(f"Nombre de lignes après filtrage : {len(df_ratings_filtered):,}")


# Début du ML
# Définit le format des notes (de 0.5 à 5.0) (Readme)
reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(
    df_ratings_filtered[["userId", "tmdbId", "rating"]],
    reader)

# train _test
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

print("'trainset' et 'testset' créés avec succès.")

#hyperparametres réglés
param_grid = {
    "n_factors": [40],
    "n_epochs": [30],
    "lr_all": [0.005],
    "reg_all": [0.02, 0.025]
}

gs = GridSearchCV(
    SVD,
    param_grid,
    measures=["rmse", "mae"],
    cv=3,
    n_jobs=-1
)

print("Début du GridSearch...")
gs.fit(data)
print("GridSearch terminé")


print("Meilleure RMSE :", gs.best_score["rmse"])
print("Meilleurs paramètres :", gs.best_params["rmse"])

best_algo = gs.best_estimator["rmse"]
#entrainement sur meilleur modèle
best_algo.fit(trainset)



predictions = best_algo.test(testset)

# RMSE
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)
print(f"\nLe modèle SVD a une RMSE : {rmse} sur test")
print(f"\nLe modèle a une MAE : {mae}")




# export
path_model_export = os.path.join('model', 'svd_model_final.joblib')

joblib.dump(best_algo, path_model_export, compress=3)




#---------------test---------------------------------
#choisir un user
user_id = df_ratings_filtered["userId"].iloc[0]
#tous les films
all_items = df_ratings_filtered["tmdbId"].unique()
#films deja notés
rated_items = df_ratings_filtered[
    df_ratings_filtered["userId"] == user_id
]["tmdbId"].values

#films non notés
unseen_items = [item for item in all_items if item not in rated_items]

predictions = []
#predire notes
for item in unseen_items:
    pred = best_algo.predict(user_id, item)
    predictions.append((item, pred.est))
#top 5
top_5 = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
print(top_5)



df_ratings_filtered["tmdbId"].iloc[880]


df_test = pd.merge(df_movies, df_links, how = 'left', on = 'movieId')


df_merge_total = pd.merge(df_ratings_filtered, df_test, how = 'left', on = 'tmdbId' )


print(df_test[df_test['tmdbId'] == 5548])


