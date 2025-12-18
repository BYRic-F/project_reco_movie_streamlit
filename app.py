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
import unicodedata
import re

# cle api et secrets
# On récupère les clés depuis les secrets (fonctionne en Local ET sur le Cloud)
API_KEY = st.secrets["tmdb_api_key"]
SHEETS_API_URL = st.secrets["sheets_api_url"]


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
    """Charge et met en cache le DataFrame principal (parquepickle)."""
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

#-------------------- FOnciton filtre genre ---------------------------------------------------------
def filtrer_films(df, genre, annee, pays, acteur, producteur, duree):
    df_res = df.copy()
    if genre and genre != "Tous":
        df_res = df_res[df_res['genres'].apply(lambda x: genre in x)]
    if acteur and acteur != "Tous":
        df_res = df_res[df_res['actor_actress'].apply(lambda x: acteur in x)]
    if producteur and producteur != "Tous":
        df_res = df_res[df_res['producer'].apply(lambda x: producteur in x)]
    if annee and annee != "Tous":
        df_res = df_res[df_res['startYear'] == annee]
    if pays and pays != "Tous":
        df_res = df_res[df_res['original_language'] == pays]
    if duree and duree != "Toutes":
        if duree == "Moins de 90 min":
            df_res = df_res[df_res['runtimeMinutes'] < 90]
        elif duree == "90-120 min":
            df_res = df_res[(df_res['runtimeMinutes'] >= 90) & (df_res['runtimeMinutes'] <= 120)]
        elif duree == "Plus de 120 min":
            df_res = df_res[df_res['runtimeMinutes'] > 120]
    return df_res

#---------------------------- def nettoyage input film -----------------------------------
def retirer_accents(texte):
    """
    Poru tout nettoyer
    """
    if not isinstance(texte, str):
        return str(texte)
        #  On enleve les accents
    texte_norm = unicodedata.normalize('NFKD', texte)
    texte_sans_accent = "".join([c for c in texte_norm if not unicodedata.combining(c)])
    # minuscule
    texte_lower = texte_sans_accent.lower()   
    # Regex quete
    texte_clean = re.sub(r'[^a-z0-9]', ' ', texte_lower)   
    #On nettoie les espaces multiples (ex: "arrete  moi" -> "arrete moi")
    return " ".join(texte_clean.split())

#-----------------------------  Creation fonction svd : récupérer note + pred ------------------------------

# le svd model ne marchait pas sur des utilisateurs qui n'etait pas dans la BDD d'entrainement, on utilise donc un vecteur qui aime ce film, aime aussi ce film
def generer_reco_via_qi(model_svd, ratings_dict):
    """
    Utilisation vecteur latent por predictions
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
            # Conversion ID externe (TMDB) -> ID interne (Surprise) fonciton to_inner_iid
            # On force int() car les clés JSON arrivent souvent en string
            iid = model_svd.trainset.to_inner_iid(int(movie_id))
            # Pondération : On donne plus d'importance aux films bien notés comme ca on evite navets
            poids = float(rating) / 5.0
            # Ajout au vecteur utilisateur
            user_vector += matrice_items[iid] * poids
            count += 1
        except (ValueError, KeyError):
            # gestionn erreur
            continue
            
    # Normalisation car sinon plus on noterait plus le vecteur serait long = distance faussée
    user_vector = user_vector / count
    # le vecteur user est calculé, on va chercher les films qui s'alignent
    # Calcul de similarité 
    # Cela génère un score de pertinence pour chaque film .dot permet de comparer le vecteur au 10000 vecteurs
    scores = np.dot(matrice_items, user_vector)
    
    # On récupère les plus grands indices
    # argsort trie du plus petit au plus grand, [::-1] inverse pour avoir le décroissant
    
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

# Dico de genres traduits ------------------------
#film
genres_traduits = { 
    "Action": "Action", "Adult": "Adulte", "Adventure": "Aventure", "Animation": "Animation",
    "Biography": "Biographie", "Comedy": "Comédie", "Crime": "Crime", "Documentary": "Documentaire",
    "Drama": "Drame", "Family": "Famille", "Fantasy": "Fantastique", "History": "Histoire",
    "Horror": "Horreur", "Music": "Musique", "Musical": "Comédie musicale", "Mystery": "Mystère",
    "News": "Actualités", "Romance": "Romance", "Sci-Fi": "Science-fiction", "Sport": "Sport",
    "Thriller": "Thriller", "War": "Guerre", "Western": "Western"
    }
#-------- docu
genres_traduits_docu = { 
        "Action": "Action", "Adult": "Adulte", "Adventure": "Aventure", "Animation": "Animation",
        "Biography": "Biographie", "Comedy": "Comédie", "Crime": "Crime", "Documentary": "Autres",
        "Drama": "Drame", "Family": "Famille", "Fantasy": "Fantastique", "History": "Histoire",
        "Horror": "Horreur", "Music": "Musique", "Musical": "Comédie musicale", "Mystery": "Mystère",
        "News": "Actualités", "Romance": "Romance", "Sci-Fi": "Science-fiction", "Sport": "Sport",
        "Thriller": "Thriller", "War": "Guerre"}
#--------------------------------------Fonction sauvegarde note----------------------------

def envoyer_note(movie_id, score, movie_title, username):
    payload_rate = {
        "action": "submit_ratings",
        "username": username,
        "ratings": {str(movie_id): int(score)}
    }
    try:
        res = requests.post(SHEETS_API_URL, json=payload_rate)
        if res.status_code == 200:
            st.toast(f"Note de {score}/5 enregistrée pour {movie_title} !", icon="✅")
        else:
            st.error("Erreur lors de l'envoi.")
    except Exception as e:
        st.error(f"Erreur de connexion : {e}")

#----------------------------------------Creation fonction Api tmdb--------------------------------- 

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
    """ Affiche la page de première configuration a la 1ere co"""

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
            options_films = sorted(list(genres_traduits.keys()), key=lambda x: genres_traduits[x])
            new_genres = st.multiselect(
                        "",
                        options=options_films,
                        format_func=lambda x: genres_traduits[x]  #traduction
                        ,key="update_genres"
                    )
        with col2pref : 
            st.markdown("**Préférences du thème des documentaires :**")
            options_docs = sorted(list(genres_traduits_docu.keys()), key=lambda x: genres_traduits_docu[x])
            new_doc_genres = st.multiselect(
                "",
                options=options_docs,
                format_func=lambda x: genres_traduits_docu[x], #traduction
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
        # On affiche un bandeau alerte user
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
            colleft, colcen, colrig = st.columns([1,10,1])
            with colcen :
                st.write(f"Bienvenu : {st.session_state['name']}")
        if st.session_state["is_guest"] is True:
            colleft1, colcen1, colrig1= st.columns([1,5,1])
            with colcen1 :
                st.write("Bienvenu : Invité")
        page_selection = option_menu(
        menu_title = None, options =
        base_option,
        icons = base_icons)
        if st.session_state["authentication_status"] is True:
            colleft1, colcen1, colrig1= st.columns([1.5,5,1])
            with colcen1 :
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
    col1_h, col2_h, col3_h = st.columns([1.5,5,1])
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
    with st.expander("📖 Statistiques de la base"):
        st.write("### Coup d'œil sur le catalogue")
        
        # Métriques
        c1, c2, c3 = st.columns(3)
        total_films = df_streamlit['title_final'].nunique()
        nb_acteurs = len(df_streamlit.explode('actor_actress'))
        total_pays = df_streamlit['original_language'].nunique()
        
        c1.metric("Films", f"{total_films:,}".replace(",", " "))
        c2.metric("Acteurs/Actrices", f"{nb_acteurs:,}".replace(",", " "))
        c3.metric("Pays", f"{total_pays:,}".replace(",", " "))

        st.markdown("---")
        
        # --- GRAPHIQUES 
        
        graph_col1, graph_col2 = st.columns(2, gap="medium")

        with graph_col1:
            st.subheader("Par genre")
            # Préparation des données Genres
            df_exploded = df_streamlit.explode('genres')
            genre_counts = df_exploded['genres'].value_counts().reset_index()
            genre_counts.columns = ['Genre', 'Count']
            
            # Chart Altair
            chart_genres = alt.Chart(genre_counts).mark_bar(color='#E10600').encode(
                x=alt.X('Genre', sort='-y', title=None),
                y=alt.Y('Count', title='Nombre de films'),
                tooltip=['Genre', 'Count']
            ).properties(
                height=300 # Hauteur fixe
            ).interactive()
            
            st.altair_chart(chart_genres, use_container_width=True)

            with graph_col2:
                st.subheader("Répartition des notes")
                chart_ratings = alt.Chart(df_knn).mark_bar(color='#E10600').encode(
                    x=alt.X('averageRating', bin=alt.Bin(maxbins=30), title='Note moyenne'),
                    y=alt.Y('count()', title='Nombre de films'),
                    # Tooltip adapté pour afficher l'intervalle et le compte
                    tooltip=[alt.Tooltip('averageRating', bin=True, title='Note approx.'), 'count()']
                ).properties(
                    height=300
                ).interactive()
                
                st.altair_chart(chart_ratings, use_container_width=True)

    st.markdown("---")
    
    col1, col2, col3 = st.columns([5,2,6])
    with col2 :
        st.image("https://www.gif-maniac.com/gifs/50/50384.gif", width=500)
    

# ------------------fonction page film ---------------

def page_film():
    """ affiche la page film """    
    #initialiser les etats avant pour eviter les erreurs
    submit_surprise = False
    submit_filtre = False
    error_films = False
    if 'search_active' not in st.session_state:
        st.session_state.search_active = False

    
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
        st.markdown("### Comment souhaitez-vous rechercher votre prochain film ?")
        choix_filtres = st.radio("", (radio_choix))
    
    # Si on change de mode de recherche, on réinitialise la mémoire sinon affiche persiste
    if choix_filtres != "Recherche par titre":
        st.session_state.search_active = False
        
    if choix_filtres != "Recherche par filtres":
        if 'resultats_filtre_memoire' in st.session_state:
            del st.session_state['resultats_filtre_memoire']

    if choix_filtres != "Surprends moi !" and 'film_surprise_actuel' in st.session_state:
        del st.session_state['film_surprise_actuel']
        
# AFFIcgha ge  Recherche par titre--------
    search_col1, search_col2, search_col3 = st.columns(3)   
    if choix_filtres == "Recherche par titre":
        # --- MISE EN CACHE DES DONNÉES DE RECHERCHE ---
        if 'search_data_cache' not in st.session_state:
            # On prend directement les colonnes depuis votre DataFrame
            # Index 0 = Titre nettoyé (pour chercher)
            # Index 1 = Titre affiché (pour le selectbox)
            st.session_state['search_data_cache'] = df_streamlit[['title_search', 'title_final']].values.tolist()

        search_data = st.session_state['search_data_cache']
        
        with radio_col2 :
            st.markdown("<div style='height:85px'></div>",unsafe_allow_html=True)
            query = st.text_input(label ="Votre film coup de cœur ?")           
            film_write = None
            
            if query:
                # 1. Nettoyage de la saisie
                q_clean = retirer_accents(query)
                
                # 2. Algorithme de recherche (List comprehension rapide)
                exact = [original for clean, original in search_data if clean == q_clean]
                
                starts = [original for clean, original in search_data if clean.startswith(q_clean) and original not in exact]
                
                contains = [original for clean, original in search_data if q_clean in clean and original not in exact + starts]

                results = exact + sorted(starts, key=str.lower) + sorted(contains, key=str.lower)
                
                if results:
                    film_write = st.selectbox("Choisis parmi la liste : ", options=results[:20])
                else:
                    error_films = st.error("Aucun film trouvé, merci de réessayer.")

            # Si la recherche n'est pas active, on affiche le message d'info
            if not st.session_state.search_active and query and error_films == False: 
                st.info("Sélectionnez un titre et cliquez sur 'Propose moi des suggestions'.")
        
        with radio_col2 :
            message_placeholder = st.empty()
            button1, button2, button3 = st.columns([1.5,3,1])
            with button2 :
                # bouton lance et active memoire pour search_active
                if st.button("Propose moi des suggestions"):
                    if not query:            
                            message_placeholder.error("Veuillez saisir un titre de film.")
                    else :
                        st.session_state.search_active = True

#----- Affichage recherche par filre
    
    if choix_filtres == "Recherche par filtres" :
        #pror eviter refresh
        with st.form(key= 'form_filtres_films') :
            filtre_col1, filtre_col2, filtre_col3 = st.columns(3)   
            #systeme de tra        
            with filtre_col1:
                choix_fr = st.selectbox("Genre", ['Tous'] + sorted(genres_traduits.values()))
                if choix_fr == "Tous":
                    genre_anglais = "Tous"
                else :
                    genre_anglais = [ang for ang, fr in genres_traduits.items() if fr == choix_fr][0]
                actor = st.selectbox("Acteurs", ['Tous'] + sorted(df_streamlit['actor_actress'].explode().dropna().unique().tolist()))
            with filtre_col2 :
                annee = st.selectbox("Année de sortie", ["Tous"] + sorted(df_streamlit['startYear'].dropna().unique().tolist(), reverse=True))
                producteur = st.selectbox("Producteurs", ["Tous"] + sorted(df_streamlit['producer'].explode().dropna().unique().tolist()))
            with filtre_col3 :
                pays = st.selectbox("Pays de production",['Tous'] + sorted(df_streamlit['original_language'].dropna().unique().tolist()))
                duration = st.select_slider("Durée", options=["Toutes", "Moins de 90 min", "90-120 min", "Plus de 120 min"])  
            filtreselec1, filtreselec2 , filtreselec3 = st.columns(3)
            with filtreselec2 : 
                st.info("Sélectionnez vos filtres ! ")
                button1, button2, button3 = st.columns([2,3,1])
                with button2 :
                    submit_filtre = st.form_submit_button("Lancer la recherche", type="primary")
    
    
    
#---------------- Affichage Surprends moi page film
        
    if choix_filtres == "Surprends moi !":
        with radio_col2:
            st.markdown("<div style='height:110px'></div>",unsafe_allow_html=True)
            st.info("Voici une recommandation personnalisée surprise pour vous !")
            col_but1, colbut2, colbut3 = st.columns([3,6,1])
            with colbut2 :
                submit_surprise = st.button("Nouveau film surprise")
    st.markdown("---")
#------- Calcul Recherche titre Knn
    # condtion memoire
    if st.session_state.search_active and film_write:
        
        # 1. CALCUL KNN
        index = df_streamlit[df_streamlit['title_final'].str.lower() == film_write.strip().lower()].index[0]
        movie_data_for_ml = df_knn.iloc[[index]][ML_FEATURES]
        distances, indices = pipeline_knn[1].kneighbors(pipeline_knn[0].transform(movie_data_for_ml))
        
        reco_df = df_streamlit.iloc[indices[0][1:5]]
        
        st.success(
            f"🎯 **Films recommandés pour : {film_write}** \n\n"
            "Résultats obtenus par **Nearest Neighbors** (IA) : comparaison de la sémantique des résumés (NLP), des genres, réalisateurs et acteurs.")

        # chargement user note comme surprends moi
        user_real_ratings = {}
        ids_deja_vus = set()
        if st.session_state.get("authentication_status") is True:
            try:
                username = st.session_state['username']
                url_api = f"{SHEETS_API_URL}?action=get_user_ratings&username={username}"
                response = requests.get(url_api)
                if response.status_code == 200:
                    user_real_ratings = response.json()
                    def nettoyer_id(val):
                        try: return str(int(float(val)))
                        except: return str(val).strip()
                    ids_deja_vus = set(nettoyer_id(k) for k in user_real_ratings.keys())
            except: pass

        # affichage 4 films
        for i, (idx, row) in enumerate(reco_df.iterrows()):
            f_id = row['id']
            details = obtenir_details_film(f_id)
            
            if details and "id" in details:
                # Récup API TMDB 
                credits_url = f"https://api.themoviedb.org/3/movie/{f_id}/credits?api_key={API_KEY}&language=fr-FR"
                credits = requests.get(credits_url).json()
                annee = details.get('release_date', '????')[:4]
                genres = ", ".join([g['name'] for g in details.get('genres', [])])
                directors = [m["name"] for m in credits.get("crew", []) if m["job"] == "Director"]
                cast = [a["name"] for a in credits.get("cast", [])[:5]]
                
                col1, col2 = st.columns([1, 3]) 
                with col1:
                    poster = details.get("poster_path")
                    if poster: st.image(f"https://image.tmdb.org/t/p/w500{poster}", width='stretch')
                
                with col2:
                    col1mov, col2mov, col3mov = st.columns([0.2, 4, 0.2])
                    with col2mov:
                        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
                        st.subheader(f"{i+1}. {details.get('title')}")
                        st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
                        st.write(f"📅 **Année :** {annee} | 🎭 **Genres :** {genres} | 🕒 **Durée :** {details.get('runtime')} minutes")
                        st.write(f"🎬 **Réalisateur :** {', '.join(directors)}")
                        st.write(f"👥 **Acteurs :** {', '.join(cast)}")
                        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

                        resume = details.get('overview')
                        if not resume or resume.strip() == "":
                            details_en = requests.get(f"https://api.themoviedb.org/3/movie/{f_id}?api_key={API_KEY}&language=en-US").json()
                            resume = details_en.get('overview', 'Aucun résumé.')
                        st.write(f"📖 **Résumé :** {resume}")
                        
                        # focntion premium "regarder cefilm"
                        if st.session_state.get("authentication_status") is True:
                            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                            st.success("🎯 **Améliorez vos résultats :** Cliquez sur 'Je regarde ce film' pour que l'algorithme comprenne ce que vous aimez !")
                            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                            f_id_str = str(int(float(f_id)))
                            
                            #  Déjà vu ? on l'affiche
                            if f_id_str in ids_deja_vus:
                                note_mise = user_real_ratings.get(f_id_str, "?")
                                st.success(f"✅ Déjà vu (Note : {note_mise}/5)")
                            
                            # 2. Pas vu ? Affiche les boutons de choix
                            else:
                                col_btn_w, col_btn_s = st.columns(2)
                                
                                # Bouton Watchlist
                                with col_btn_w:
                                    if st.button("🍿 Je regarde !", key=f"reco_w_{f_id}_{i}", width="stretch"):
                                        payload = {"action": "add_watchlist", "username": st.session_state['username'], "movie_id": f_id_str}
                                        requests.post(SHEETS_API_URL, json=payload)
                                        st.toast("Ajouté !")

                                # Bouton Notation formulaire pour eviter rerun 
                                with col_btn_s:
                                    with st.popover("✅ J'ai déjà vu", width='stretch'):
                                        st.write(f"Notez **{details.get('title')}**")
                                        
                                        # Le formulaire bloque le rechargement quand on clique sur l'étoile sinon ca rerun
                                        with st.form(key=f"frm_rt_{f_id}_{i}"):
                                            rating_val = st.feedback("stars", key=f"reco_rate_{f_id}_{i}")
                                            submitted_search = st.form_submit_button("Valider", type="primary")
                                        #envoi
                                        if submitted_search:
                                            if rating_val is not None:
                                                try:
                                                    payload_rate = {
                                                        "action": "submit_ratings",
                                                        "username": st.session_state['username'],
                                                        "ratings": {f_id_str: rating_val + 1}
                                                    }
                                                    requests.post(SHEETS_API_URL, json=payload_rate)
                                                    st.toast("Note enregistrée !")
                                                    st.rerun() # Pour rafraîchir l'affichage en "Déjà vu"
                                                except Exception as e:
                                                    st.error(f"Erreur : {e}")

            st.markdown("---")

                    # ----------------------------------------------



    #-----------Calcul recherche filtre-------------------
    
    if submit_filtre:
        # On calcule les résultats
        resultats = filtrer_films(
            df_streamlit, 
            genre=genre_anglais, 
            annee=annee, 
            pays=pays, 
            acteur=actor, 
            producteur=producteur, 
            duree=duration
        )
        
        # Gestion des résultats vides
        if resultats.empty:
            st.warning("Aucun film ne correspond exactement à tous ces critères combinés. Essayez d'en enlever un !")
            # On nettoie la mémoire pour ne pas laisser de vieux résultats
            if 'resultats_filtre_memoire' in st.session_state:
                del st.session_state['resultats_filtre_memoire']
        else:
            #
            nb = len(resultats)
            nb_a_afficher = min(nb, 20)
            #Gestkl de sessions_state pour l'envoi de notes
            st.session_state['resultats_filtre_memoire'] = resultats.sample(n=nb_a_afficher)
            st.session_state['nb_resultats_filtre'] = nb

    if 'resultats_filtre_memoire' in st.session_state:
        
        films_affi = st.session_state['resultats_filtre_memoire']
        nb_total = st.session_state.get('nb_resultats_filtre', 0)
        
        st.success(f"🎯 {nb_total} film(s) trouvé(s) !")

        # chargement user et notes
        user_real_ratings = {}
        ids_deja_vus = set()
        
        if st.session_state.get("authentication_status") is True:
            try:
                username = st.session_state['username']
                url_api = f"{SHEETS_API_URL}?action=get_user_ratings&username={username}"
                response = requests.get(url_api)
                if response.status_code == 200:
                    user_real_ratings = response.json()
                    def nettoyer_id(val):
                        try: return str(int(float(val)))
                        except: return str(val).strip()
                    ids_deja_vus = set(nettoyer_id(k) for k in user_real_ratings.keys())
            except: 
                pass

        # affichage comme dh'ab
        for i, (idx, row) in enumerate(films_affi.iterrows()):
            try: f_id = str(int(float(row['id'])))
            except: continue

            # Appel API TMDB
            details = obtenir_details_film(f_id)
            
            if details and "id" in details:
                # Récup Crédits
                try:
                    credits_url = f"https://api.themoviedb.org/3/movie/{f_id}/credits?api_key={API_KEY}&language=fr-FR"
                    credits_data = requests.get(credits_url).json()
                    directors = [m["name"] for m in credits_data.get("crew", []) if m["job"] == "Director"]
                    cast = [a["name"] for a in credits_data.get("cast", [])[:5]]
                except:
                    directors, cast = [], []

                annee_film = details.get('release_date', '????')[:4]
                genres_film = ", ".join([g['name'] for g in details.get('genres', [])])
                
                # Mise en page Colonnes
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
                        st.write(f"📅 **Année :** {annee_film} | 🎭 **Genres :** {genres_film} | 🕒 **Durée :** {details.get('runtime')} minutes")
                        st.write(f"🎬 **Réalisateur :** {', '.join(directors)}")
                        st.write(f"👥 **Acteurs :** {', '.join(cast)}")
                        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

                        # Résumé
                        resume = details.get('overview')
                        if not resume or resume.strip() == "":
                            try:
                                details_en = requests.get(f"https://api.themoviedb.org/3/movie/{f_id}?api_key={API_KEY}&language=en-US").json()
                                resume = details_en.get('overview', 'Aucun résumé.')
                            except: resume = "Pas de résumé."
                        
                        # Tronquer résumé si trop long
                        if len(resume) > 400: resume = resume[:400] + "..."
                        st.write(f"📖 **Résumé :** {resume}")
                        
                        # --- ZONE ACTIONS (Utilisateur connecté) ---
                        if st.session_state.get("authentication_status") is True:
                            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                            st.success("🎯 **Améliorez vos résultats :** Cliquez sur 'Je regarde ce film' pour que l'algorithme comprenne ce que vous aimez !")
                            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                            # 1. Déjà vu ?
                            if f_id in ids_deja_vus:
                                note_mise = user_real_ratings.get(f_id, "?")
                                st.success(f"✅ Déjà vu (Note : {note_mise}/5)")
                            
                            # 2. Pas vu ? Boutons d'action
                            else:
                                col_btn_w, col_btn_s = st.columns(2)
                                
                                # BOUTON WATCHLIST
                                with col_btn_w:
                                    if st.button("🍿 Je regarde !", key=f"reco_w_filt_{f_id}_{i}", width="stretch"):
                                        try:
                                            payload = {"action": "add_watchlist", "username": st.session_state['username'], "movie_id": f_id}
                                            requests.post(SHEETS_API_URL, json=payload)
                                            st.toast("Ajouté !", icon="✅")
                                        except Exception as e:
                                            st.error(f"Erreur : {e}")

                                # BOUTON NOTATION
                                with col_btn_s:
                                    with st.popover("✅ J'ai déjà vu", width='stretch'):
                                        st.write(f"Notez **{details.get('title')}**")
                                        
                                        with st.form(key=f"frm_rt_filt_{f_id}_{i}"):
                                            rating_val = st.feedback("stars", key=f"reco_rate_filt_{f_id}_{i}")
                                            submitted_rate = st.form_submit_button("Valider", type="primary")
                                            
                                            if submitted_rate:
                                                if rating_val is not None:
                                                    try:
                                                        payload_rate = {
                                                            "action": "submit_ratings",
                                                            "username": st.session_state['username'],
                                                            "ratings": {f_id: rating_val + 1}
                                                        }
                                                        res = requests.post(SHEETS_API_URL, json=payload_rate)
                                                        if res.status_code == 200:
                                                            st.toast("Note enregistrée !")
                                                            st.rerun()
                                                    except Exception as e:
                                                        st.error(f"Erreur : {e}")
            
            st.markdown("---")
            
            
    
    #-------Calcules surprends moi ------


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
        
        # On récupère id film chiusui
        f_id = st.session_state['film_surprise_actuel']
        nb_notes = st.session_state.get('nb_notes_user', '?')

        # On récup detail
        details = obtenir_details_film(f_id)
        
        if details and "id" in details:
            st.success(f"✨ Basé sur vos {nb_notes} films notés !")

            # Récup tmdb
            credits_url = f"https://api.themoviedb.org/3/movie/{f_id}/credits?api_key={API_KEY}&language=fr-FR"
            try: credits_data = requests.get(credits_url).json()
            except: credits_data = {}

            # Variables d'affichage
            i = 0
            annee = details.get('release_date', '????')[:4]
            genres = ", ".join([g['name'] for g in details.get('genres', [])])
            directors = [m["name"] for m in credits_data.get("crew", []) if m["job"] == "Director"]
            cast = [a["name"] for a in credits_data.get("cast", [])[:5]]

            # --MISE EN PAGE -
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
                    st.write(f"📅 **Année :** {annee} | 🎭 **Genres :** {genres}  | 🕒 **Durée :** {details.get('runtime')} minutes")
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
                            if st.button("🍿 Je regarde !", key=f"btn_watch_{f_id}", width="stretch"):
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
                            # On garde le popover
                            with st.popover("✅ J'ai déjà vu", width='stretch'):
                                st.markdown(f"**Notez :** {details.get('title')}")
                                
                                # --- AJOUT DU FORMULAIRE ICI ---
                                # Le formulaire empêche le rerun immédiat au clic sur les étoiles
                                with st.form(key=f"form_rating_{f_id}"):
                                    
                                    # L'étoile ne rechargera plus la page grâce au formulaire
                                    rating_val = st.feedback("stars", key=f"feed_rate_{f_id}")
                                    
                                    # Ce bouton déclenche l'envoi et le rerun en une seule fois
                                    submitted = st.form_submit_button("Valider la note", type="primary")
                                    
                                # --- LOGIQUE APRÈS VALIDATION ---
                                if submitted:
                                    if rating_val is not None:
                                        final_score = rating_val + 1
                                        
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
                                                    #pour relancer
                                                    if 'film_surprise_actuel' in st.session_state:
                                                        del st.session_state['film_surprise_actuel']
                                                    st.rerun() 
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
    #initialisation pour eviter erreurs
    submit_doc_fil = False
    submit_surprise = False
    #Selection selon l'authentif
    # On filtre pour n'avoir que les documentaires
    df_documentaire = df_streamlit[df_streamlit['genres'].apply(lambda genres: 'Documentary' in genres)]
    radio_docu = ['Recherche par filtres']
    col_head_doc, col2_head_doc, col3_head_doc = st.columns([3,6,1])
    if st.session_state["authentication_status"] is True:
        radio_docu.append("Surprends moi !")
    ## affichage titre
    with col2_head_doc:
        st.header("📚 Recherche de documentaires 📚")
    st.markdown("---")  
    
    ## affichage comment....    
    col1_search_t, col2_search_t, col3_search_t = st.columns(3)
    with col1_search_t: 
        st.markdown("### Comment souhaitez-vous rechercher votre prochain documentaire ?")
        
        choix_filtres = st.radio("",
        radio_docu
        )
        # Contenu 1ere colonne : Le choix par mot-clé 
    if choix_filtres == "Recherche par filtres":
        df_documentaire = df_streamlit[df_streamlit['genres'].apply(lambda genres: 'Documentary' in genres)]
        st.info("Veuillez choisir le thème de votre documentaire")

        with st.form(key='form_filtres_docu'):
                col_filt1, col_filt2 = st.columns(2)

                with col_filt1:
                    # 2. Utilisation de format_func pour la traduction
                    choix_fr = st.selectbox("Genre", ['Tous'] + sorted(genres_traduits_docu.values()))
                    if choix_fr == "Tous":
                        choix_genre_docu = "Tous"
                    else :
                        choix_genre_docu = [ang for ang, fr in genres_traduits_docu.items() if fr == choix_fr][0]
                    duration_docu = st.select_slider("Durée", options=["Toutes", "Moins de 90 min", "90-120 min", "Plus de 120 min"])  
                
                with col_filt2:
                    annee_docu = st.selectbox("Année de sortie", ["Tous"] + sorted(df_documentaire['startYear'].dropna().unique().tolist(), reverse=True))
                
                col_filt1button, col_filt2button, col_filt3button = st.columns([2.3,1,2])   
                
                with col_filt2button:        
                    # 3. Changement en bouton de soumission de formulaire
                    submit_doc_fil = st.form_submit_button("Lancer la recherche")


    #### SI recherche par filtre ATTENtion il faudra rajouter les conditions filtres
    if submit_doc_fil: 
        # On calcule les résultats
        resultats_docu = filtrer_films(
            df_documentaire, 
            genre=choix_genre_docu, 
            annee=annee_docu, 
            pays=None, 
            acteur=None, 
            producteur=None, 
            duree=duration_docu
        )
        
        # Gestion des résultats vides
        if resultats_docu.empty:
            st.warning("Aucun film ne correspond exactement à tous ces critères combinés. Essayez d'en enlever un !")
            # On nettoie la mémoire pour ne pas laisser de vieux résultats
        else:
            # On compte le nombre total de résultats
            nb = len(resultats_docu)
            
            # combien de film trouvé
            st.success(f"{nb} documentaires trouvés (Affichage des 20 premiers)")

            # IMPORTANT : On coupe le tableau pour ne garder que les 20 premiers
            nb_max = min(len(resultats_docu), 20)
            films_affi = resultats_docu.sample(n=nb_max)
            
            # On boucle sur la version coupée (films_affi) et non pas sur tout le tableau
            for i, (idx, row) in enumerate(films_affi.iterrows()):
                try: 
                    f_id = str(int(float(row['id'])))
                except: 
                    continue

                # Appel API TMDB
                details = obtenir_details_film(f_id)
                
                if details and "id" in details:
                    # Récup Crédits
                    try:
                        credits_url = f"https://api.themoviedb.org/3/movie/{f_id}/credits?api_key={API_KEY}&language=fr-FR"
                        credits_data = requests.get(credits_url).json()
                        directors = [m["name"] for m in credits_data.get("crew", []) if m["job"] == "Director"]
                        cast = [a["name"] for a in credits_data.get("cast", [])[:5]]
                    except:
                        directors, cast = [], []

                    annee_film = details.get('release_date', '????')[:4]
                    genres_film = ", ".join([g['name'] for g in details.get('genres', [])])
                    
                    # colonnes
                    col1, col2 = st.columns([1, 3]) 
                    
                    with col1:
                        poster = details.get("poster_path")
                        if poster: 
                            st.image(f"https://image.tmdb.org/t/p/w500{poster}", width='stretch')
                    
                    with col2:
                        # affichage
                        col1mov, col2mov, col3mov = st.columns([0.2, 4, 0.2])
                        with col2mov:
                            st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
                            st.subheader(f"{i+1}. {details.get('title')}")
                            st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
                            st.write(f"📅 **Année :** {annee_film} | 🎭 **Genres :** {genres_film} | 🕒 **Durée :** {details.get('runtime')} minutes")
                            st.write(f"🎬 **Réalisateur :** {', '.join(directors)}")
                            st.write(f"👥 **Acteurs :** {', '.join(cast)}")
                            st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

                            resume = details.get('overview')
                            if not resume or resume.strip() == "":
                                details_en = requests.get(f"https://api.themoviedb.org/3/movie/{f_id}?api_key={API_KEY}&language=en-US").json()
                                resume = details_en.get('overview', 'Aucun résumé.')
                            st.write(f"📖 **Résumé :** {resume}")
                    
                    st.markdown("----")
    # Surrends moi docu est basé uniquement sur les thèmes de l'utilisteur 
                
    if choix_filtres == "Surprends moi !":
        with col2_search_t:
            st.markdown("<div style='height:110px'></div>",unsafe_allow_html=True)
            st.info("Voici une recommandation personnalisée surprise pour vous !")
            col_but1, colbut2, colbut3 = st.columns([1.9,6,1])
            with colbut2 :
                submit_surprise = st.button("Nouveau documentaire surprise")
                
    st.markdown("---")
    if submit_surprise:
            # recup pref user
            user_prefs = user_data.get('doc_genres_pref', [])
            
            # filtrage avec genres user
            if user_prefs:
                # filtre sur les docs selong genres user
                df_filtered = df_documentaire[df_documentaire['genres'].apply(
                    lambda g_list: any(pref in g_list for pref in user_prefs))]
                
                if not df_filtered.empty:
                    selection = df_filtered.sample(1)
                    st.success(f"🎯 Sélectionné parmi vos thèmes préférés")
            else:
                # si 0 pref random
                selection = df_documentaire.sample(1)
                
            # 3. Sauvegarde en session
            st.session_state['docu_surprise_actuel'] = str(int(float(selection.iloc[0]['id'])))

        # --- AFFICHAGE DU RESULTAT ---
            if 'docu_surprise_actuel' in st.session_state:
                f_id = st.session_state['docu_surprise_actuel']
            
            # Récup infos API TMDB
            details = obtenir_details_film(f_id)
            
            # Récupération de l'historique utilisateur pour savoir si "Déjà vu"
            # On le fait ici pour être sûr d'avoir l'info à jour
            user_seen_ids = set()
            if st.session_state.get("authentication_status"):
                try:
                    res_ratings = requests.get(f"{SHEETS_API_URL}?action=get_user_ratings&username={st.session_state['username']}")
                    if res_ratings.status_code == 200:
                        # On récupère juste les IDs des films notés
                        raw_ratings = res_ratings.json()
                        # Nettoyage des IDs (str/float/int)
                        user_seen_ids = set(str(int(float(k))) for k in raw_ratings.keys())
                except:
                    pass

            if details and "id" in details:
                # Affichage
                col1, col2 = st.columns([1, 3]) 
                
                with col1:
                    poster = details.get("poster_path")
                    if poster: st.image(f"https://image.tmdb.org/t/p/w500{poster}", width='stretch')
                
                with col2:
                    col1mov, col2mov, col3mov = st.columns([0.2, 4, 0.2])
                    with col2mov :
                        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
                    st.subheader(f"✨ {details.get('title')}")
                    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
                    annee = details.get('release_date', '????')[:4]
                    genres = ", ".join([g['name'] for g in details.get('genres', [])])
                    st.write(f"📅 **Année :** {annee} | 🎭 **Genres :** {genres} | 🕒 **Durée :** {details.get('runtime')} min")
                    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
                    resume = details.get('overview')
                    if not resume: 
                        try: resume = requests.get(f"https://api.themoviedb.org/3/movie/{f_id}?api_key={API_KEY}&language=en-US").json().get('overview', 'Pas de résumé.')
                        except: resume = "Pas de résumé."
                    st.write(f"📖 **Résumé :** {resume}")
                    
                    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
                    
                    # --- ZONE ACTIONS SIMPLIFIÉE ---
                    
                    # 1. Vérification si déjà vu
                    f_id_clean = str(int(float(f_id)))
                    
                    if f_id_clean in user_seen_ids:
                        # Juste l'info visuelle
                        st.success("✅ **Déjà vu** (Vous avez déjà noté ce documentaire)")
                st.markdown("---")   
#----------------------------------------- Page profil -----------------------------------------------------------    

def page_profil() : 
    profil_utilisateur = {
        "Identifiant": st.session_state['username'],
        "Nom": st.session_state['name'],
        "Email": st.session_state['email'],
    }

    col_left, col_center, col_right = st.columns([5,3,5])
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
            options_films = sorted(list(genres_traduits.keys()), key=lambda x: genres_traduits[x])
            st.markdown("**Préférences du genre de films :**")
            new_genres = st.multiselect(
                        "",
                        options=options_films,           
                        format_func=lambda x: genres_traduits[x], #
                        key="update_genres",
                        default = user_genres
                    )
        
        with col2:
            st.markdown("**Préférences du thème des documentaires :**")
            options_docs = sorted(list(genres_traduits_docu.keys()), key=lambda x: genres_traduits_docu[x])
            new_doc_genres = st.multiselect(
                "",
                options=options_docs,
                format_func=lambda x: genres_traduits_docu[x],
                key="update_doc_genres",
                default= user_docs
                )
        col_biu1, col_biu2, col_biu3 = st.columns([3.7,3,3])
        with col_biu2 :
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
        
        if st.button("Recommencer la notation ⭐️", width="stretch"):
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
        elif not any(char.isalpha() for char in username):
            st.error("⛔ Le nom d'utilisateur doit contenir au moins une lettre (ex: 'Moi10' et non '10').")
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