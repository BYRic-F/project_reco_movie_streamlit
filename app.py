# import des librairies
from surprise import Dataset, Reader
import streamlit as st
from streamlit_authenticator import Authenticate, Hasher
from streamlit_option_menu import option_menu
import requests
import json
import pandas as pd
import os
import joblib
import numpy as np
import altair as alt
import random

API_KEY = "99dea9a9dd2b4d7d7f47e9a0cdd160f7"

# Configuration de la page Streamlit
st.set_page_config(layout="wide",
                initial_sidebar_state="auto")

#-------------------------------------- CHARGEMENT DES BASES DES BDD-----------------------------------------------#
#Chargerment fichier Css
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# definition de quelques listes :
# liste de films les plus populaires et variés pour créer les préférences des utilisateurs
csv_path = os.path.join("data", "processed", "movies_db.csv")
dataframe_path = os.path.join("data", "processed", 'dataframe_streamlit.pkl')

#Chargement des bases de données
@st.cache_data(show_spinner="Chargement du catalogue films...")
def load_dataframe_streamlit(path):
    """Charge et met en cache le DataFrame principal (parquet)."""
    return pd.read_pickle(path)

@st.cache_data(show_spinner="Chargement de la base de données de notation...")
def load_movies_db(path):
    """Charge et met en cache le DataFrame des films pour la notation (csv)."""
    return pd.read_csv(path)

df_streamlit = load_dataframe_streamlit(dataframe_path)
df_movies = load_movies_db(csv_path)

#---------chargement des modeles ------------
#   ----- Nearest neighboors + nlp + bdd entrainement  ---------

path_pipeline = os.path.join('model', 'pipeline_knn.joblib')
path_df_ml = os.path.join('data', 'processed', 'dataframe_ready_for_ML.pkl')
#importation de la fonction weight sinon ca ne amrche pas
def weight_features(X, weight):
  return X * weight
@st.cache_data(show_spinner="Chargement du modèle de recommandation...")
def load_knn_pipeline(path):
    """Charge le pipeline KNN une seule fois et le met en cache."""
    pipeline = joblib.load(path)
    return pipeline
ML_FEATURES = ['genres_clean','directors_clean','actor_actress_clean','NLP',  'startYear','averageRating','numVotes','production_companies_name_clean']
# Appeler la fonction mise en cache pour charger le pipeline
pipeline_knn = load_knn_pipeline(path_pipeline)
df_knn = pd.read_pickle(path_df_ml)

# --- Chargement du modèle et des foncitons SVD ---
path_svd = os.path.join('model', 'model_svd.joblib')

@st.cache_resource 
def load_svd_model(path):
    return joblib.load(path)

model_svd = load_svd_model(path_svd)

#-----------------------------  Creation fonction svd : récupérer note + pred ------------------------------

# le svd model ne marchait pas sur des utilisateurs qui n'etait pas dans la BDD d'entrainement, on utilise donc un vecteur qui aime ce film, aime aussi ce film
def generer_reco_via_qi(model_svd, ratings_dict):
    """
    Génère des recommandations basées sur les vecteurs latents (QI).
    Récupère le Top 100 pour permettre le filtrage des films déjà vus ensuite.
    """
    # On récupere les matrices d'entrainement de notre svd (carte complete des films)
    matrice_items = model_svd.qi
    n_factors = matrice_items.shape[1]
    
    # vecteur user nul
    user_vector = np.zeros(n_factors)
    count = 0
    
    # Construction profil user
    # ratings_dict, films notés sur Sheets
    for movie_id, rating in ratings_dict.items():
        try:
            # Conversion ID externe (TMDB) -> ID interne (Surprise)
            # On force int() car les clés JSON arrivent souvent en string
            iid = model_svd.trainset.to_inner_iid(int(movie_id))
            # Pondération : On donne plus d'importance aux films bien notés
            poids = float(rating) / 5.0
            # Ajout au vecteur utilisateur
            user_vector += matrice_items[iid] * poids
            count += 1
        except (ValueError, KeyError):
            # Cas où le film noté n'est pas dans le dataset d'entraînement
            continue
            
    # Normalisation car sinon plus on noterait plus le vecteur serait long = distance faussée
    user_vector = user_vector / count
    # le vecteur user est calculé, on va chercher les films qui s'alignent
    # Calcul de similarité 
    # Cela génère un score de pertinence pour chaque film .dot permet de comparer le vecteur au 10000 vecteurs
    scores = np.dot(matrice_items, user_vector)
    
    # On récupère les plus grands indices
    # argsort trie du plus petit au plus grand, [::-1] inverse pour avoir le décroissant
    # 100 meilleurs, le random est plus pertinent
    top_indices = np.argsort(scores)[::-1][:100]
    
    # 7. Convertir les IDs internes (Surprise) en IDs réels (TMDB) comme le modele retient des index, ca permet de retourner en arriere
    #et trouver le bon tmdbid avec l'index
    reco_ids = []
    for iid in top_indices:
        try:
            raw_id = model_svd.trainset.to_raw_iid(iid)
            reco_ids.append(raw_id)
        except:
            pass
            
    return reco_ids
#----------------------------------------

BASE_URL = "https://image.tmdb.org/t/p/w500"

ALL_DOC_GENRES = [
    "Biographie",
    "Histoire","Nature et Environnement","Science et Technologie",
    "Société et Culture","Voyage et Exploration",
    "Affaires Criminelles","Sport","Musique et Art","Alimentation et Cuisine","Guerre et Conflit"]



#----------------------------------------Creation fonction Api--------------------------------- 

# Fonction pour récupérer les détails d'un film par ID
def obtenir_details_film(film_id):
    details_url = f"https://api.themoviedb.org/3/movie/{film_id}?api_key={API_KEY}&language=fr-FR"
    return requests.get(details_url).json()
def obtenir_cast(film_id):
    credits_url = f"https://api.themoviedb.org/3/movie/{film_id}/credits?api_key={API_KEY}&language=fr-FR"
    return requests.get(credits_url).json().get("cast", [])
#------------------------------- Creation des fonctions des differentes pages ---------------------------------------------------------------
#
#--------------------------------- page config new user--------------------------------------------------------

def page_new_user1():
    """ Affiche la page de première configuration avec une sélection de films et pagination. """

    # Intro
    col1_h, col2_h, col3_h = st.columns([2,5,1])
    st.markdown('---')
    with col2_h : 
        st.markdown("## 🤩 Préparez-vous à l'expérience PicquePoule ! 🤩")
    st.markdown("Bienvenue ! Pour que vous puissiez profiter au maximum de notre application, nous allons vous demander de prendre juste un petit moment.")
    st.markdown(
        """
        **Ne vous inquiètez pas, ça ira très vite, et vous ne le regretterez pas !**
        
        En quelques clics seulement, vous bénéficierez de **recommandations personnalisées** spécialement conçues pour vous. Fini la perte de temps à chercher quoi regarder ; PicquePoule vous sert le meilleur contenu sur un plateau.
        
        Commençons tout de suite !
        """
    )
    st.markdown("---")
    st.info("""
    **Étape 1 : Notation des films**
    
    Cliquez sur les étoiles pour noter les films que vous avez vus.
    
    **Il nous faut au moins 5 notes** pour cerner vos goûts !
    """)
    
    
    with st.form("SVD_form") :
        COLUMNS_PER_ROW = 6
        cols = st.columns(COLUMNS_PER_ROW)
        rating_keys = {} #stock les clés
        #Boucle pour affichage
        for i,(index, row) in enumerate(df_movies.iterrows()):
            with cols[i % COLUMNS_PER_ROW]:
                
                #affichage films
                st.markdown(f"**{row['title']}**")
                st.image(BASE_URL + row['poster_path'], width='stretch')
                col1ji, col2ji, col3ji = st.columns([1,3,1])
                #crée des id uniques pour retenir la variable
                key_name = f"rate_{row['tmdb_id']}"
                rating_keys[row['tmdb_id']] = key_name
                with col2ji :
                    st.feedback("stars", key=key_name)
                st.markdown("---")
        
        st.info("Définissez vos préférences : Films (genres) et Documentaires (thèmes).")
        col1pref, col2pref = st.columns(2)
        with col1pref : 
            st.markdown("**Préférences du genre de films :**")
            new_genres = st.multiselect(
                        "",
                        ["Action", "Comédie", "Drame", "Horreur", "Science-Fiction"]
                        ,key="update_genres"
                    )
        with col2pref : 
            st.markdown("**Préférences du thème des documentaires :**")
            new_doc_genres = st.multiselect(
                "",
                options=ALL_DOC_GENRES,
                key="update_doc_genres")
        st.markdown("---")
        
        
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info2 :
            st.info("Ces choix seront toujours réinitialisables dans la section Profil")
        
        col1,col2,col3 = st.columns([5,5,1])
        with col2 :    
            submit = st.form_submit_button("Je valide mes choix")    

    # Envoi des infos a Sheets        
    if submit:
        # le feedback calcule de 0-4 doncon passe de 1a 5
        ratings_to_send = {}
        
        for movie, key in rating_keys.items():
            raw_score = st.session_state.get(key)
            if raw_score is not None:
                final_score = raw_score + 1 
                ratings_to_send[movie] = final_score
        
        # Verifpour avoir les 5 notes où message erreur
        if len(ratings_to_send) < 5:
            st.error(f"🛑 Il manque des notes ! Vous n'avez noté que **{len(ratings_to_send)}** films. Veuillez en noter au moins **5** pour continuer.")
        else:
            try:
                st.info("Envoi de vos notes à l'algorithme...")
                
                # Envoi des notes
                payload_ratings = {
                    "username": st.session_state['username'],
                    "action": "submit_ratings",
                    "ratings": ratings_to_send
                }
                res_ratings = requests.post(SHEETS_API_URL, json=payload_ratings)
                res_ratings.raise_for_status()
                
                # Envoi du profil
                payload_profile = {
                    "username": st.session_state['username'],
                    "genres_pref": json.dumps(new_genres),
                    "doc_genres_pref": json.dumps(new_doc_genres),
                    "action": "update_profile"
                }
                res_profile = requests.post(SHEETS_API_URL, json=payload_profile)
                res_profile.raise_for_status()

                if res_ratings.json().get('success') and res_profile.json().get('success'):
                    st.success("Profil configuré avec succès ! Bienvenue.")
                    st.session_state["is_new_user"] = False
                    
                    # Nettoyage
                    for key in rating_keys.values():
                        if key in st.session_state:
                            del st.session_state[key]
                            
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Erreur lors de l'enregistrement côté serveur.")

            except Exception as e:
                st.error(f"Erreur de connexion : {e}")
  #--------------------------- Def verifier films a noter--------          
@st.dialog("🍿 C'est l'heure du verdict !")
def afficher_notations_popup(pending_ids, username):
    st.write(f"Vous avez **{len(pending_ids)} film(s)** en attente de note.")
    
    # On affiche le premier film de la liste
    for f_id in pending_ids[:1]: 
        details = obtenir_details_film(f_id)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            poster = details.get("poster_path")
            if poster:
                st.image(f"https://image.tmdb.org/t/p/w200{poster}", width='stretch')
        with col2:
            st.subheader(details.get('title'))
            # Slider
            note = st.feedback("stars", key=f"slider_pop_{f_id}")
        
        # Bouton Valider
        if st.button("Valider la note", key=f"btn_pop_{f_id}", width='stretch'):
            if note is None:
                st.warning("Veuillez sélectionner au moins une étoile ⭐")
            else :
                with st.spinner("Enregistrement..."):
                    try:
                        clean_id = str(int(float(f_id)))
                        note_reelle = note + 1
                        payload = {
                            "action": "submit_ratings",
                            "username": username,
                            "ratings": {clean_id: int(note_reelle)}
                        }
                        requests.post(SHEETS_API_URL, json=payload)
                        st.toast("C'est enregistré !")
                        st.rerun() 
                    except Exception as e:
                        st.error(f"Erreur : {e}")


#FOnciton appel
def verifier_films_a_noter():
    # Sécurité connexion
    if not st.session_state.get("authentication_status"):
        return

    username = st.session_state['username']
    
    # Récupération API
    try:
        url = f"{SHEETS_API_URL}?action=get_pending_movies&username={username}"
        response = requests.get(url)
        pending_ids = response.json() 
    except:
        pending_ids = []

    # Si des films sont trouvés
    if pending_ids:
        # On affiche un BANDEAU D'ALERTE joli et stable
        # st.warning crée un fond jaune/orange qui attire l'oeil
        with st.container(border=True):
            col_msg, col_btn = st.columns([3, 1])
            
            with col_msg:
                st.info(f"🔔 **Hey {st.session_state.get('name')} !** Vous avez **{len(pending_ids)} film(s)** vus récemment à noter.")
            
            with col_btn:
                # C'est ce bouton qui déclenche la modale
                st.markdown("<div style='height: 2px;'></div>", unsafe_allow_html=True)
                if st.button("⭐ Noter maintenant", width='stretch'):
                    afficher_notations_popup(pending_ids, username)
                                                                                                                                        
# ------------------fonction SIDE BAR-----------------------------
def sidebar(authenticator):
    with st.sidebar :
        # definition des options de la sidebar
        base_option = ["Accueil",
                    "Recherche de films",
                    "Recherche de documentaires"]
        base_icons = [
                    "house",
                    "film",
                    "journal-text"]
        
        # ajoute les options si l'utilisateur est authentifié
        if st.session_state["authentication_status"] is True:
            base_option.append("Mon profil")
            base_icons.append("person-circle")
    
    # Creation de la sidebar "modulabe"
        # condition d'affichage selon le statut
        if st.session_state["authentication_status"] is True:
            st.write(f"Bienvenu : {st.session_state['name']}")
        if st.session_state["is_guest"] is True:
            st.write("Bienvenu : Invité")
        page_selection = option_menu(
        menu_title = None, options =
        base_option,
        icons = base_icons)
        if st.session_state["authentication_status"] is True:
            add_deco = authenticator.logout("Déconnexion")
        if st.session_state["is_guest"] is True:
            st.info("Vous êtes en mode Invité. Certaines fonctionnalités sont limitées.")
            if st.button("Se connecter / S'inscrire", key="guest_to_login_btn"):
                st.session_state["is_guest"] = False
                st.rerun()
        return page_selection
        
        
# --------- fonction page d'accueil--------

def page_accueil() :
    """ affiche la page d'accueil"""
    col1_h, col2_h, col3_h = st.columns([2,5,1])
    with col2_h :
        st.header("🎬Bienvenue sur votre plateforme PicquePoule🎬")
    st.markdown("---")
    st.markdown("### Pourquoi notre sélection est unique ?")
    
    st.markdown(
        """
        Chez PicquePoule, nous ne vous montrons que le meilleur du cinéma. Notre catalogue est filtré de manière rigoureuse pour vous garantir une expérience de visionnage de haute qualité.
        
        Voici les critères que chaque film doit respecter pour figurer dans nos résultats :
        """
    )
#### qualité pop
    with st.expander("✨ Critères de qualité et popularité"): 
        st.markdown("""
        Nous exigeons des scores de public élevés pour éliminer les contenus de faible qualité :
        
        * **Large choix :** Environ 10 000 films.
        * **Note Moyenne :** Supérieure à **6.5/10**.
        * **Nombre de Votes :** Plus de **500 votes** enregistrés.
        """)

#---- Année---
    with st.expander("📅 Critères de pertinence et format"):
        st.markdown("""
        Nous assurons une sélection de contenu adapté à une séance cinéma :
        
        * **Sortie Récente :** Films produits après **1990**.
        * **Durée Minimale :** Plus de **75 minutes**.
        * **Type de Contenu :** Uniquement des **longs-métrages**.
        """)

    #-----Origine film-----
    with st.expander("🌍 Cinéma occidental"):
        st.markdown("""
        Nous avons accès aux données d'un catalogue mondial :
        
        * **Cinéma de Référence :** Films produits principalement aux **États-Unis**, au **Royaume-Uni**, en **France**, en **Allemagne**, en **Espagne** et en **Italie**.
        * **Autres Origines Diversifiées :** Nous incluons également des œuvres significatives produites au **Japon**, aux **Pays-Bas**, au **Portugal**, en **Irlande** et en **Finlande**.
        """)
        
    #---Docu--------------------------------------
    with st.expander("📚 Documentaires incontournables"):
        st.markdown("""
        Notre sélection rassemble le meilleur du cinéma documentaire. :
        
        * Élargissez vos horizons avec notre sélection de documentaires triés sur le volet. Nous vous proposons des œuvres **de haute qualité** et des histoires puissantes pour satisfaire votre curiosité et approfondir votre compréhension du monde.
        """)
        #--------------------- NOtre BDD-------------------------------------------
    with st.expander("📖 Notre base de données"):
        total_films_B = df_streamlit['title_final'].nunique()
        total_pays = df_streamlit['original_language'].nunique()

        st.write("### Quelques infos sur nos films...")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Nombre total de films uniques",
                value=f"{total_films_B:,}".replace(",", " "))
        with col2:
            st.metric(
                label="Nombre d'acteurs et d'actrices",
                value=f"{len(df_streamlit.explode('actor_actress')):,}".replace(",", " "))
        with col3:
            st.metric(
                label="Nombre de pays de production",
                value=f"{total_pays:,}".replace(",", " "))
        def plot_genre_counts_adapted(dataframe):

    # Explosion
            df_exploded = dataframe.explode('genres')
            #  Compter la fréquence de chaque genre
            genre_counts = df_exploded['genres'].value_counts().reset_index()
            genre_counts.columns = ['Genre', 'Nombre de Films']
            return genre_counts

        genre_counts_df = plot_genre_counts_adapted(df_streamlit)
        st.header("Distribution des films par catégorie")

        # 1. Création du graphique Altair avec rotation
        chart = alt.Chart(genre_counts_df).mark_bar().encode(
            x=alt.X('Genre',axis=alt.Axis(labelOverlap=False,labelAngle=45),sort='-y'),
            y='Nombre de Films',
            tooltip=['Genre', 'Nombre de Films']
        ).properties(
            title='Nombre Total de Films par Genre'
        ).interactive()

        # 2. Affichage du graphique Altair dans Streamlit
        st.altair_chart(chart, use_container_width=True)
        st.markdown(f"**Nombre total de genres uniques :** {len(genre_counts_df)}")
        st.dataframe(genre_counts_df)


    st.markdown("---")
    
    col1, col2, col3 = st.columns([5,2,6])
    with col2 :
        st.image("https://www.gif-maniac.com/gifs/50/50384.gif", width=500)
    

# ------------------fonction page film ---------------

def page_film() :
    """ affiche la page film """    
    #initialiser les etats avant pour eviter les erreurs
    submit_titre = False
    submit_surprise = False
    submit_filtre = False
    error_films = False
    # ajout des options disponibles communes aux 2 authentifs
    radio_choix = ["Recherche par titre","Recherche par filtres"]
    
    # Ajout des options disponibles aux utilisateurs enregistrés
    if st.session_state["authentication_status"] is True:
        radio_choix.append("Surprends moi !")
    header_col1, header_col2, header_col3 = st.columns([4, 6, 1])
    with header_col2:
        st.header("🎥 Recherche de films 🎥")
    st.markdown("---")    
    radio_col1, radio_col2, radio_col3= st.columns(3)
    with radio_col1: 
        st.markdown("### Comment souhaitez-vous rechercher votre film ?")
        choix_filtres = st.radio("",
        (radio_choix
        ))
    if choix_filtres != "Surprends moi !" and 'film_surprise_actuel' in st.session_state:
        del st.session_state['film_surprise_actuel']
    # Recherche par titre
    search_col1, search_col2, search_col3 = st.columns(3)   
    if choix_filtres == "Recherche par titre":
        all_titles = sorted(df_streamlit['title_final'].dropna().astype(str).str.strip().unique(),key=str.lower)
        with radio_col2 :
            st.markdown("<div style='height:85px'></div>",unsafe_allow_html=True)
            query = st.text_input(label ="Entrez le titre du film")           
            film_write = None
            if query:
                q = query.lower()
                exact = [t for t in all_titles if t.lower() == q]
                starts = [t for t in all_titles if t.lower().startswith(q) and t not in exact]
                contains = [t for t in all_titles if q in t.lower() and t not in exact + starts]

                results = exact + sorted(starts, key=str.lower) + sorted(contains, key=str.lower)
    # selectbox sur les film filtré
                if results:
                    film_write = st.selectbox("Résultats", options=results[:20])
                else:
                    error_films = st.error("Aucun film trouvé, merci de réessayer.")

            if not submit_titre and query and error_films == False: #and film_write == '-- Sélectionner un film --': 
                st.info("Sélectionnez un titre et cliquez sur 'Lancer la recherche'.")
        with radio_col2 :
            button1, button2, button3 = st.columns([2,3,1])
            with button2 :
                submit_titre = st.button("Lancer la recherche") 
        
    # Creation des filtres de recherche( genre, année, pays de production , acteurs, realisateurs, durée)
    genres_traduits = {
    "Action": "Action",
    "Adult": "Adulte",
    "Adventure": "Aventure",
    "Animation": "Animation",
    "Biography": "Biographie",
    "Comedy": "Comédie",
    "Crime": "Crime",
    "Documentary": "Documentaire",
    "Drama": "Drame",
    "Family": "Famille",
    "Fantasy": "Fantastique",
    "History": "Histoire",
    "Horror": "Horreur",
    "Music": "Musique",
    "Musical": "Comédie musicale",
    "Mystery": "Mystère",
    "News": "Actualités",
    "Romance": "Romance",
    "Sci-Fi": "Science-fiction",
    "Sport": "Sport",
    "Thriller": "Thriller",
    "War": "Guerre",
    "Western": "Western"
}
    if choix_filtres == "Recherche par filtres" :
        filtre_col1, filtre_col2, filtre_col3 = st.columns(3)           
        with filtre_col1:
    # affiche en fr
            choix_fr = st.selectbox("Genre", sorted(genres_traduits.values()))
    
    # On recup en ang pour pas causer d'erreurs dans le df
            genre_anglais = [ang for ang, fr in genres_traduits.items() if fr == choix_fr][0]
            
            actor = st.selectbox("Acteurs", df_streamlit['actor_actress'].explode().unique())
        with filtre_col2 :
            annee = st.selectbox("Année de sortie", df_streamlit['startYear'].unique())
            producteur = st.selectbox("Producteurs", df_streamlit['producer'].explode().unique())
            st.info("Sélectionnez vos filtres ! ")
            button1, button2, button3 = st.columns([2,3,1])
            with button2 :
                submit_filtre = st.button("Lancer la recherche")  
        with filtre_col3 :
            pays = st.selectbox("Pays de production", df_streamlit['original_language'].unique())
            duration = st.select_slider("Durée", options=["Toutes", "Moins de 90 min", "90-120 min", "Plus de 120 min"])  
        
    if choix_filtres == "Surprends moi !":
        with radio_col2:
            st.markdown("<div style='height:110px'></div>",unsafe_allow_html=True)
            st.info("Voici une recommandation personnalisée surprise pour vous !")
            col_but1, colbut2, colbut3 = st.columns([3,6,1])
            with colbut2 :
                submit_surprise = st.button("Nouveau film surprise")
    st.markdown("---")
    
    #si on clique sur recherche par titre ATTENTION IL FAUDRA INTEGRE CONDITIONS FILTRES AUSSI
    if submit_titre and film_write:
        # 1. CALCUL KNN
        index = df_streamlit[df_streamlit['title_final'].str.lower() == film_write.strip().lower()].index[0]
        movie_data_for_ml = df_knn.iloc[[index]][ML_FEATURES]
        distances, indices = pipeline_knn[1].kneighbors(pipeline_knn[0].transform(movie_data_for_ml))
        
        # On récupère les 4 films les plus proches (index 1 à 4)
        reco_df = df_streamlit.iloc[indices[0][1:5]]

        st.markdown(f"### 🎯 Films recommandés pour : {film_write}")
        st.markdown("---")

        # boucle pour afficher
        for i, (idx, row) in enumerate(reco_df.iterrows()):
            # Utilisation de l'ID TMDB présent dans le pickle
            f_id = row['id']
            
            # appel fonctions api
            details = obtenir_details_film(f_id)
            
            if details and "id" in details:
                # Récupération du cast et crew
                credits_url = f"https://api.themoviedb.org/3/movie/{f_id}/credits?api_key={API_KEY}&language=fr-FR"
                credits = requests.get(credits_url).json()
                
                # Extraction des infos
                annee = details.get('release_date', '????')[:4]
                genres = ", ".join([g['name'] for g in details.get('genres', [])])
                directors = [m["name"] for m in credits.get("crew", []) if m["job"] == "Director"]
                cast = [a["name"] for a in credits.get("cast", [])[:5]]
                
                # Mise en page
                col1, col2 = st.columns([1, 3]) 
                
                with col1:
                    poster = details.get("poster_path")
                    if poster:
                        st.image(f"https://image.tmdb.org/t/p/w500{poster}", width='stretch')
                
                with col2:
                    col1mov, col2mov, col3mov = st.columns([0.2, 4, 0.2])
                    with col2mov:
                        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
                        st.subheader(f"{i+1}. {details.get('title')}")
                        
                        st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
                        st.write(f"📅 **Année :** {annee} | 🎭 **Genres :** {genres}")
                        st.write(f"🎬 **Réalisateur :** {', '.join(directors)}")
                        st.write(f"👥 **Acteurs :** {', '.join(cast)}")
                        
                        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

                        # Gestion du résumé car certaion apparaissent pas
                        resume = details.get('overview')
                        is_english = False

                        if not resume or resume.strip() == "":
                            details_en = requests.get(f"https://api.themoviedb.org/3/movie/{f_id}?api_key={API_KEY}&language=en-US").json()
                            resume = details_en.get('overview', 'Aucun résumé disponible.')
                            is_english = True

                        if is_english:
                            st.caption("🇬🇧 *Résumé uniquement disponible en anglais*")
                        
                        st.write(f"📖 **Résumé :** {resume}")

            st.markdown("---")


    #si on clique sur recherche par filtre
    
    if submit_filtre :
                #Boucle pour affichage
        st.write(df_streamlit[df_streamlit['startYear']== annee])
    
    #si on clique sur surprends moi 
# ... dans la section "Surprends moi !" de page_film ...

    # 1 on calcule quand on clique sur le bouton
    if submit_surprise:
        if model_svd is None:
            st.error("Modèle SVD non chargé.")
        else:
            with st.spinner("Récupération de vos notes..."):
                try:
                    # Appel API
                    username = st.session_state['username']
                    url_api = f"{SHEETS_API_URL}?action=get_user_ratings&username={username}"
                    response = requests.get(url_api)
                    response.raise_for_status()
                    user_real_ratings = response.json()
                except Exception as e:
                    st.error(f"Erreur API : {e}")
                    user_real_ratings = {}

                # Calcul SVD
                if user_real_ratings:
                    reco_ids = generer_reco_via_qi(model_svd, user_real_ratings)
                    
                    if reco_ids:
                        # Filtrage car les notes revenaient sous forme de float et n'etaitjamais pareil
                        def nettoyer_id(val):
                            try: return str(int(float(val)))
                            except: return str(val).strip()
                        #Pour eviter les doublons daja vus
                        ids_deja_vus_propres = set(nettoyer_id(k) for k in user_real_ratings.keys())
                        reco_ids_new = []
                        for rid in reco_ids:
                            rid_clean = nettoyer_id(rid)
                            if rid_clean not in ids_deja_vus_propres:
                                reco_ids_new.append(rid)
                        
                        if not reco_ids_new: reco_ids_new = reco_ids 

                        # Sauvegarde du choix en memoire
                        # Random pour pas toujours afficher les meme
                        film_choisi = random.choice(reco_ids_new)
                        st.session_state['film_surprise_actuel'] = film_choisi
                        st.session_state['nb_notes_user'] = len(user_real_ratings) # Juste pour l'info
                        
                    else:
                        st.warning("Aucune correspondance trouvée.")
                else:
                    st.error("Impossible de récupérer vos notes.")

    # condition si un film est affiché
    if 'film_surprise_actuel' in st.session_state :
        
        # On récupère l'ID stocké
        f_id = st.session_state['film_surprise_actuel']
        nb_notes = st.session_state.get('nb_notes_user', '?')

        # On récupère les détails (API TMDB)
        details = obtenir_details_film(f_id)
        
        if details and "id" in details:
            if submit_surprise: 
                st.balloons()
            
            st.success(f"✨ Basé sur vos {nb_notes} films notés !")

            # Récupération Crédits
            credits_url = f"https://api.themoviedb.org/3/movie/{f_id}/credits?api_key={API_KEY}&language=fr-FR"
            try: credits_data = requests.get(credits_url).json()
            except: credits_data = {}

            # Variables d'affichage
            i = 0
            annee = details.get('release_date', '????')[:4]
            genres = ", ".join([g['name'] for g in details.get('genres', [])])
            directors = [m["name"] for m in credits_data.get("crew", []) if m["job"] == "Director"]
            cast = [a["name"] for a in credits_data.get("cast", [])[:5]]

            # --- MISE EN PAGE ---
            col1, col2 = st.columns([1, 3]) 
            with col1:
                poster = details.get("poster_path")
                if poster:
                    st.image(f"https://image.tmdb.org/t/p/w500{poster}", width='stretch')
            
            with col2:
                col1mov, col2mov, col3mov = st.columns([0.2, 4, 0.2])
                with col2mov:
                    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
                    st.subheader(f"{i+1}. {details.get('title')}")
                    
                    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                    st.write(f"📅 **Année :** {annee} | 🎭 **Genres :** {genres}")
                    st.write(f"🎬 **Réalisateur :** {', '.join(directors)}")
                    st.write(f"👥 **Acteurs :** {', '.join(cast)}")
                    
                    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

                    # Résumé
                    resume = details.get('overview')
                    if not resume or resume.strip() == "":
                        try:
                            details_en = requests.get(f"https://api.themoviedb.org/3/movie/{f_id}?api_key={API_KEY}&language=en-US").json()
                            resume = details_en.get('overview', 'Aucun résumé disponible.')
                            st.caption("🇬🇧 *Résumé en anglais*")
                        except: resume = "Pas de résumé."
                    
                    st.write(f"📖 **Résumé :** {resume}")
                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                    st.success("🎯 **Améliorez vos résultats :** Cliquez sur 'Je regarde ce film' pour que l'algorithme comprenne ce que vous aimez !")

                    # --- LE BOUTON "JE REGARDE"  et envoie de la requete a la sheets
                    if st.session_state.get("authentication_status") is True:
                        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                        
                        # Création de deux colonnes pour aligner les boutons
                        col_btn_watch, col_btn_seen = st.columns(2)
                        
                        # COLONNE 1 : BOUTON WATCHLIST
                        with col_btn_watch:
                            if st.button("🍿 Je regarde !", key=f"btn_watch_{f_id}", use_container_width=True):
                                with st.spinner("Ajout à votre historique..."):
                                    try:
                                        payload = {
                                            "action": "add_watchlist",
                                            "username": st.session_state['username'],
                                            "movie_id": str(int(float(f_id)))
                                        }
                                        requests.post(SHEETS_API_URL, json=payload)
                                        st.toast("Ajouté à votre liste 'À voir' !", icon="✅")
                                    except Exception as e:
                                        st.error(f"Erreur connexion : {e}")

                        # COLONNE 2 : BOUTON DEJA VU + NOTATION
                        with col_btn_seen:
                            # petit popover
                            with st.popover("✅ J'ai déjà vu", width='stretch'):
                                st.markdown(f"**Notez :** {details.get('title')}")
                                
                                # Les étoiles (0-4 par défaut, donc on ajoutera +1)
                                rating_val = st.feedback("stars", key=f"feed_rate_{f_id}")
                                
                                if st.button("Valider la note", key=f"btn_send_rate_{f_id}", type="primary"):
                                    if rating_val is not None:
                                        final_score = rating_val + 1  # vonversion 0-4 vers 1-5
                                        
                                        with st.spinner("Envoi..."):
                                            try:
                                                payload_rate = {
                                                    "action": "submit_ratings",
                                                    "username": st.session_state['username'],
                                                    "ratings": {str(int(float(f_id))): final_score}
                                                }
                                                res = requests.post(SHEETS_API_URL, json=payload_rate)
                                                if res.status_code == 200:
                                                    st.toast(f"Note de {final_score}/5 enregistrée !")
                                                else:
                                                    st.error("Erreur API")
                                            except Exception as e:
                                                st.error(f"Erreur : {e}")
                                    else:
                                        st.warning("Veuillez sélectionner au moins une étoile.")
            st.markdown("---")
        
#----------------------------------------fonciton Page_docu-------------------------------------

    
def page_docu():
    """ Affichage page docu"""
    #Initalise etat
    submit_doc_fil = False
    submit_doc_sur = False
    #Selection selon l'authentif
    
    radio_docu = ['Recherche par filtres']
    # Ajout des options disponibles aux utilisateurs enregistrés
    if st.session_state["authentication_status"] is True:
        radio_docu.append("Surprends moi !")
    col_head_doc, col2_head_doc, col3_head_doc = st.columns([3,6,1])
    
    ## affichage titre
    with col2_head_doc:
        st.header("📚 Recherche de documentaires 📚")
    st.markdown("---")  
    
    ## affichage comment....    
    col1_search_t, col2_search_t, col3_search_t = st.columns(3)
    with col1_search_t: 
        st.markdown("### Comment souhaitez-vous rechercher votre documentaire ?")
        
        choix_filtres = st.radio("",
        radio_docu
        )
        # Contenu 1ere colonne : Le choix par mot-clé 
    if choix_filtres == "Recherche par filtres":
        st.info("Veuillez choisir le thème de votre documentaire")
        col_filt1, col_filt2= st.columns(2)
        
        with col_filt1:
            recherche_genre = genre = st.selectbox("Thème", ALL_DOC_GENRES)
            duration = st.select_slider("Durée", options=["Toutes", "Moins de 90 min", "90-120 min", "Plus de 120 min"])
    # Contenu 2e colonne : tri date ancien ou recent
        with col_filt2:
            annee_docu_range = st.selectbox(
            "Période de sortie", 
            ["Toutes les années","Moins de 1 an (Très Récent)",
    "Moins de 5 ans (Contemporain)","Moins de 10 ans",
    "Après 2000","1990 - 2000 (Classique)"])
                  
        submit_doc_fil = st.button("Lancer la recherche")
    
    # Contenu surprend moi (uniquement user enregistré)    
    if choix_filtres == "Surprends moi !":
        col_surprise, col_surprise2, col_surpri_3 = st.columns(3)
        with col_surprise :
            st.info("Voici une recommandation personnalisée surprise pour vous !")
        submit_doc_sur = st.button("Nouveau documentaire surprise")
    st.markdown("---") 
    
    
    #### SI recherche par filtre ATTENtion il faudra rajouter les conditions filtres
    if submit_doc_fil: 
        LIST_TEST_DOCU = ["coucou", 'blou']
        COLUMNS_PER_ROW = 2
        cols = st.columns(COLUMNS_PER_ROW)

        for i, movie in enumerate(LIST_TEST_DOCU):
            with cols[i % 2]:
                
                #affichage films
                st.markdown(f"**{movie}**")
                st.image("https://m.media-amazon.com/images/I/71-B0aUFxYL._AC_SL1191_.jpg", 
                            width=400 )
                
                st.markdown("---")
                
    ### SI Surprends moi 
    if submit_doc_sur: 
        LIST_TEST_DOCU = ["coucou", 'blou']
        COLUMNS_PER_ROW = 2
        cols = st.columns(COLUMNS_PER_ROW)

        for i, movie in enumerate(LIST_TEST_DOCU):
            with cols[i % 2]:
                
                #affichage films
                st.markdown(f"**{movie}**")
                st.image("https://m.media-amazon.com/images/I/71-B0aUFxYL._AC_SL1191_.jpg", 
                            width=400 )
                
                st.markdown("---")
#----------------------------------------- Page profil -----------------------------------------------------------    

def page_profil() : 
    profil_utilisateur = {
        "Identifiant": st.session_state['username'],
        "Nom": st.session_state['name'],
        "Email": st.session_state['email'],
    }

    col_left, col_center, col_right = st.columns(3)
    # Affichage du titre
    with col_center:
        
        st.header("👤 Mon Profil 👤")

    st.markdown("---") 
#-------- Partie information de base-----------------

    st.subheader("Informations de Base")

    col_id, col_nom, col_email = st.columns(3)
    with col_id:
        st.markdown("**Identifiant :**")
        st.text(profil_utilisateur['Identifiant'])
    with col_nom:
        st.markdown("**Nom et prénom :**")
        st.text(profil_utilisateur['Nom'])


    with col_email :
        st.markdown("**Email :**")
        st.text(profil_utilisateur['Email'])

    st.markdown("---")

#-------------------- partie préférences-----------
    #Préférences 
    user_genres = user_data.get('genres_pref', []) 
    user_docs =  user_data.get('doc_genres_pref', [])
    with st.form("profile_update_form"):
        st.subheader("Préférences (pour les recommandations)")
        col1, col2= st.columns(2)
        with col1:
        # Affichage des listes (Genre, Films, Documentaires)
            st.markdown("**Préférences du genre de films :**")
            new_genres = st.multiselect(
                        "",
                        ["Action", "Comédie", "Drame", "Horreur", "Science-Fiction"]
                        ,key="update_genres",
                        default=user_genres
                    )
        
        with col2:
            st.markdown("**Préférences du thème des documentaires :**")
            new_doc_genres = st.multiselect(
                "",
                options=ALL_DOC_GENRES,
                key="update_doc_genres",
                default= user_docs
                )
        save_submitted = st.form_submit_button("Sauvegarder les modifications du profil")
    st.markdown("---")
    
# si on clique sur sauvegardes des modifs, ca envoie tout a l'api
    if save_submitted:
            try:
                # Préparation des données pour l'API (Conversion en chaînes JSON/texte brut) 
                
                # Préparation du dictionnaire de mise à jour
                update_payload = {
                    "username": st.session_state['username'], 
                    "genres_pref": json.dumps(new_genres), 
                    "doc_genres_pref": json.dumps(new_doc_genres), 
                    "action": "update_profile" 
                }
                
                # Appel POST à l'API Sheets
                st.info("Sauvegarde des modifications du profil...")
                write_response = requests.post(SHEETS_API_URL, json=update_payload)
                write_response.raise_for_status()
                
                response_json = write_response.json()
                
                if response_json.get('success'):
                    st.success("Profil mis à jour avec succès ! Rafraîchissement...")
                    st.cache_data.clear()
                    st.rerun() 
                else:
                    st.error(f"Échec de la sauvegarde du profil : {response_json.get('error', 'Erreur inconnue')}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur de connexion à l'API lors de la sauvegarde : {e}")
            except Exception as e:
                st.error(f"Erreur inattendue : {e}")


# -------------------- Partie SVD (Notation & Recommandations) -------------------
    st.subheader("Intelligence Artificielle & Recommandations")
    
    
    st.info("""
        **Optimiser vos suggestions de films**
        
        L'algorithme apprend de vos notes. Si vous trouvez que les recommandations ne sont pas assez précises, 
        vous pouvez relancer l'étape de notation pour affiner votre profil IA.
        """)
        

    col_svd_1, col_svd_2, col_svd_3 = st.columns(3)
    with col_svd_2:
        
        if st.button("Recommencer la notation ⭐️", use_container_width=True):
            #on repasse sur page new_user
            st.session_state["is_new_user"] = True
            st.rerun()



#------------------------------Chargement données depuis Sheets et Authentification-----------------------------------------------------------------------------#
#
#--------------------------------------------------------------------------------------------------------------------------------------------#

#--- Initialisation de l'état du visieur ---
if "is_guest" not in st.session_state:
    st.session_state["is_guest"] = False
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "is_new_user" not in st.session_state:
    st.session_state["is_new_user"] = False
    
        # url api google sheets
        
SHEETS_API_URL = "https://script.google.com/macros/s/AKfycbyQaRK6AYD4zCtkn8DvHO6mO4E7GJiG5x9YOh_C1nt7U59eZCREsGFyFnOmewMmd2g5aA/exec"

        # Donnees des comptes
# --- 1. Fonction de Chargement et de Formatage des Données ---

@st.cache_data(show_spinner="Chargement des données utilisateurs depuis notre base de données..")
def charger_donnees_sheets():
    """Charge les données d'identification depuis notre base de données."""
    
    try:
        # --- APPEL L'API SHEETS (GET) ---
        response = requests.get(SHEETS_API_URL)
        response.raise_for_status() 
        
        data_from_sheets = response.json()
        
        # Vérification si la réponse est une liste 
        if not isinstance(data_from_sheets, list):
            st.error("Erreur de format de l'API: la réponse n'est pas une liste d'utilisateurs.")
            return {'usernames': {}}
            
    except requests.exceptions.RequestException as e:
        st.error(f"Échec de la connexion à l'API Sheets. Vérifiez l'URL ou les permissions. Détail: {e}")
        return {'usernames': {}}


    # --- Transformation des Données au format Streamlit Authenticator ---
    usernames = {}
    for user_data in data_from_sheets:
        username = user_data.get('username')
        if username and user_data.get('password'):
            try:
                # Utilise json.loads pour convertir la chaîne de caractères en liste Python
                genres_pref = json.loads(user_data.get('genres_pref', '[]'))
                doc_genres_pref = json.loads(user_data.get('doc_genres_pref', '[]'))
            except json.JSONDecodeError:
                genres_pref = []
                doc_genres_pref = []
            usernames[username] = {
                'name': user_data.get('name'),
                'password': user_data.get('password'), 
                'email': user_data.get('email'),
                'failed_login_attemps': 0,
                'logged_in': False,
                'role': user_data.get('role', 'utilisateur'),
                'genres_pref': genres_pref, 
                'doc_genres_pref': doc_genres_pref
            }
    return {'usernames': usernames}

donnees_comptes = charger_donnees_sheets()

#-----------------------------------Authentification des utilisateurs--------------------------------------------------------------#

authenticator = Authenticate(
    donnees_comptes,
    "cookie_name_unique",
    "cookie_key_secure",
    30,
)

# --- Determine le statut de connexion  et charge les données---

user_data = {} 
user_status = 'Non connecté'
is_logged_in = False

if st.session_state["authentication_status"] is True:
    username = st.session_state["username"]
    
    # Sécurisation contre les données vides et initialisation de user_data
    if username in donnees_comptes.get('usernames', {}):
        user_data = donnees_comptes['usernames'][username] 
        user_status = user_data['role']
        is_logged_in = True
    else:
        st.session_state["authentication_status"] = None
        user_status = 'Non connecté'
        is_logged_in = False
    
elif st.session_state["is_guest"] is True:
    user_status = 'Invité'
    is_logged_in = False
    user_data = {}
    
#-----------------------------Fin chargement données depuis Sheets et Authentification-----------------------------------------------------------------------------#
#
#
#------------------------------Creation du menu de navigation-----------------------------------------------------------------------------#

# si la session est co en invité ou utilisateur affiche la sidebar
if st.session_state["authentication_status"] is True or st.session_state["is_guest"] is True:
    
    # page config new user
    if st.session_state["authentication_status"] is True and st.session_state["is_new_user"] is True:
        page_new_user1()
    
    else :
        page_selection = sidebar(authenticator)
        verifier_films_a_noter()
    # affichage des pages selon le choix de la sidebar
        
        if page_selection == "Accueil":
            page_accueil()
        
        
        elif page_selection == "Recherche de films":
            page_film()
        
        elif page_selection == "Recherche de documentaires" : 
            page_docu()
        
        elif page_selection == "Mon profil":
            page_profil()
        
    
    
    
    
    
    
    


# ------------------------------------------------------------------------------PAGE DE LOGIN -----------------------------------------------------------------
#
#---------------------------------------------------------------page d'accueil non co---------------------------------------------------------------------------------------------
else:
    col1,col2,col3 = st.columns([7,8,5])
    with col2:  
        st.header("🎬Bienvenue sur PicquePoule🎬")
        
    left_col, center_col, right_col = st.columns([1, 2, 1])
    with center_col:
        st.markdown("---")
        
    # onglets d'authentification
        tab_register, tab_login, tab_visiteur = st.tabs(["S'inscrire", "Se connecter", "Visiteur"])

#------------------------ Onglet S'INSCRIRE --------------------
    with tab_register:
        st.subheader("Créer un compte gratuit")
        st.info("Inscrivez vous pour accéder à toutes les fonctionnalités de PicquePoule !")

    # 1. Création du formulaire standard (Fait avec IA car authenticator.register_user("main") ne permet pas de capturer le mot de passe en clair)
        with st.form("registration_form"):
            st.write("Veuillez saisir vos informations :")
            
            # Capture du Nom complet
            col_name1, col_name2 = st.columns(2)
            with col_name1:
                first_name = st.text_input("Prénom", key="reg_first_name")
            with col_name2:
                last_name = st.text_input("Nom de famille", key="reg_last_name")
                
            # Champs essentiels
            email = st.text_input("Email", key="reg_email")
            username = st.text_input("Username", key="reg_username")
            
            # Capture du Mot de passe en clair
            password_raw = st.text_input("Mot de passe", type='password', key="reg_password")
            repeat_password = st.text_input("Répéter le mot de passe", type='password', key="reg_repeat_password")
            
            submitted_inscription = st.form_submit_button("S'inscrire")

    # 2. Traitement du formulaire
    if submitted_inscription :
        # 2a. Vérifications de base
        if password_raw != repeat_password:
            st.error("Les mots de passe ne correspondent pas.")
        elif not username or not email or not password_raw:
            st.error("Veuillez remplir tous les champs obligatoires.")
        else:
            try:
                # 🚨 ÉTAPE CRUCIALE : Hachage du mot de passe
                hasher = Hasher()
                hashed_password = hasher.hash(password_raw)
                
                # Concaténation du Nom complet
                name_of_registered_user = f"{first_name} {last_name}".strip()
                
                # --- Enregistrement du Nouvel Utilisateur dans Sheets (POST) ---
                new_user_data = {
                    "username": username, 
                    "password": hashed_password,       
                    "name": name_of_registered_user,    
                    "email": email,
                    "role": "utilisateur",
                    "failed_login_attemps": 0,
                    "logged_in": False,
                }
                
                st.info("Tentative d'enregistrement dans notre base de données...")
                
                # Appel POST à l'API Sheets
                write_response = requests.post(SHEETS_API_URL, json=new_user_data)
                write_response.raise_for_status() 
                
                response_json = write_response.json()
                if response_json.get('success'):
# renvoie vers la page de configuration
                    st.success("Inscription réussie ! Redirection vers la configuration du profil...")
                    st.cache_data.clear()
                    st.session_state["authentication_status"] = True # Force la connexion
                    st.session_state["username"] = username         # Définit le username
                    st.session_state["name"] = name_of_registered_user # Définit le nom complet
                    st.session_state["is_new_user"] = True        # Active la page Onboarding
                    st.rerun()
                else:
                    st.error(f"Échec de l'enregistrement : {response_json.get('error', 'Erreur inconnue')}")

            except requests.exceptions.HTTPError as e:
                st.error(f"Erreur HTTP lors de l'appel à l'API Sheets: Vérifiez l'URL ou les logs Apps Script. Détail: {e}")
            except Exception as e:
                st.error(f"Erreur inattendue lors de l'inscription: {e}")
                
# ---------------  Onglet SE CONNECTER--------------------------------
    with tab_login:
        st.subheader("Connexion")
        # Code pour la connexion (login)
        authenticator.login(location = 'main', key = 'login_main')
            
        # Gestion de l'échec de la connexion
        if st.session_state["authentication_status"] == False:
            st.error('Identifiants incorrects.')

# ---------------------Onglet VISITEUR -------------------------
    with tab_visiteur:
        st.subheader("Mode Invité / Visiteur")
        st.info("Accédez immédiatement à la démo avec des fonctionnalités limitées.")
        
        if st.button("Continuer en tant que visiteur", key="guest_button"):
            # Définir l'état de l'invité à True et recharger la page
            st.session_state["is_guest"] = True
            st.rerun()
            
            
# ------------------