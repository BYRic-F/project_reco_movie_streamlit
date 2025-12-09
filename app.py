# import des librairies

import streamlit as st
from streamlit_authenticator import Authenticate, Hasher
from streamlit_option_menu import option_menu
import requests
import json
import pandas as pd

# Configuration de la page Streamlit
st.set_page_config(layout="wide",
                initial_sidebar_state="auto")

#Chargerment fichier Css
with open(r"C:\Users\frede\Vs_Code\dossier_projets\projet_test\style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# definition de quelques listes :
# liste de films les plus populaires et variés pour créer les préférences des utilisateurs
ALL_MOVIES = [
                "Inception", "The Dark Knight", "Interstellar", "Pulp Fiction", "The Matrix", 
                "Fight Club", "Parasite", "Mad Max: Fury Road", "La La Land", "Joker", 
                "Avatar", "Titanic", "Forrest Gump", "Dune", "Arrival", "Blade Runner 2049", 
                "The Godfather", "The Shawshank Redemption", "A Beautiful Mind", 
                "Gladiator",
                "Spirited Away", "The Exorcist", "City of God", "Amélie", 
                "Kill Bill: Vol. 1", "Eternal Sunshine of the Spotless Mind", 
                "Whiplash", "No Country for Old Men", "Get Out","Narnia",
                    
                    ]

ALL_DOC_GENRES = [
    "Biographie",
    "Histoire","Nature et Environnement","Science et Technologie",
    "Société et Culture","Voyage et Exploration",
    "Affaires Criminelles","Sport","Musique et Art","Alimentation et Cuisine","Guerre et Conflit"]


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
        for i, movie in enumerate(ALL_MOVIES):
            with cols[i % COLUMNS_PER_ROW]:
                
                #affichage films
                st.markdown(f"**{movie}**")
                st.image("https://m.media-amazon.com/images/I/71-B0aUFxYL._AC_SL1191_.jpg", 
                            use_container_width=True )
                col1ji, col2ji, col3ji = st.columns([1,3,1])
                key_name = f"rate_{movie}"
                rating_keys[movie] = key_name
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
        # 1. Récupération et conversion des notes (0-4 -> 1-5)
        ratings_to_send = {}
        
        for movie, key in rating_keys.items():
            raw_score = st.session_state.get(key)
            if raw_score is not None:
                final_score = raw_score + 1 
                ratings_to_send[movie] = final_score
        
        # 2. VÉRIFICATION STRICTE (Le "if" que vous avez demandé)
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
        
    #---Docu
    with st.expander("🌍 Documentaires incontournables"):
        st.markdown("""
        Notre sélection rassemble le meilleur du cinéma documentaire. :
        
        * Élargissez vos horizons avec notre sélection de documentaires triés sur le volet. Nous vous proposons des œuvres **de haute qualité** et des histoires puissantes pour satisfaire votre curiosité et approfondir votre compréhension du monde.
        """)

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
    # ajout des options disponibles communes aux 2 authentifs
    radio_choix = ["Recherche par titre",]
    
    # Ajout des options disponibles aux utilisateurs enregistrés
    if st.session_state["authentication_status"] is True:
        radio_choix.append("Surprends moi !")
    
    header_col1, header_col2, header_col3 = st.columns([3, 6, 1])
    with header_col2:
        st.header("🎥 Recherche de films 🎥")
    st.markdown("---")    
    radio_col1, radio_col2, radio_col3= st.columns(3)
    with radio_col1: 
        st.markdown("### Comment souhaitez-vous rechercher votre film ?")
        choix_filtres = st.radio("",
        (radio_choix
        ))
    # Creation des filtres de recherche( genre, année, pays de production , acteurs, realisateurs, durée)
    search_col1, search_col2, search_col3 = st.columns(3)   
    if choix_filtres == "Recherche par titre":
        with search_col1 :
            film_write = st.text_input("Entrez le titre du film que vous recherchez :")
            
        with st.expander("Appliquer un filtre"):
            filtre_col1, filtre_col2, filtre_col3 = st.columns(3)           
            with filtre_col1 :    
                genre = st.selectbox("Genre", ["Tous", "Action", "Comédie", "Drame", "Horreur", "Science-Fiction"])
                duration = st.select_slider("Durée", options=["Toutes", "Moins de 90 min", "90-120 min", "Plus de 120 min"])
            with filtre_col2 :
                annee = st.selectbox("Année de sortie", ["Toutes", "2020", "2019", "2018", "2017", "2016"])
            with filtre_col3 :
                pays = st.selectbox("Pays de production", ["Tous", "USA", "France", "Royaume-Uni", "Canada", "Allemagne"])
        submit_titre = st.button("Lancer la recherche")
        
    if choix_filtres == "Surprends moi !":
        with search_col1 :
            st.info("Voici une recommandation personnalisée surprise pour vous !")
        submit_surprise = st.button("Nouveau film surprise")
    #gère erreur si pas defilm
    if submit_titre and not film_write:
        st.error("Attention : **Entrez le nom d'un film pour lancer la recherche.**")
    st.markdown("---")
    
    #si on clique sur recherche par titre ATTENTION IL FAUDRA INTEGRE CONDITIONS FILTRES AUSSI
    if submit_titre and film_write: 
                #Boucle pour affichage
        COLUMNS_PER_ROW = 4
        cols = st.columns(COLUMNS_PER_ROW)
        LISTE_test = ['Plop', 'Babar', 'Jm', "lalal"]
        for i, movie in enumerate(LISTE_test):
            with cols[i % 4]:
                
                #affichage films
                st.markdown(f"**{movie}**")
                st.image("https://m.media-amazon.com/images/I/71-B0aUFxYL._AC_SL1191_.jpg", 
                            use_container_width=True )
                
                st.markdown("---")

    
    #si on clique sur surprends moi 
    if submit_surprise : 
                #Boucle pour affichage
        COLUMNS_PER_ROW = 4
        cols = st.columns(COLUMNS_PER_ROW)
        LISTE_test = ['Plop', 'Babar', 'Jm', "lalal"]
        for i, movie in enumerate(LISTE_test):
            with cols[i % 4]:
                
                #affichage films
                st.markdown(f"**{movie}**")
                st.image("https://m.media-amazon.com/images/I/71-B0aUFxYL._AC_SL1191_.jpg", 
                            use_container_width=True )
                
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
        st.markdown("**Nom et prénom:**")
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