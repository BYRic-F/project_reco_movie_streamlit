import streamlit as st
import requests

API_KEY = "99dea9a9dd2b4d7d7f47e9a0cdd160f7"

# Fonction pour rechercher un film et retourner le premier résultat
def rechercher_film(nom_film):
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={nom_film}&language=fr-FR"
    response = requests.get(search_url).json()
    if response["results"]:
        return response["results"][0]
    return None

# Fonction pour récupérer les détails d'un film par ID
def obtenir_details_film(film_id):
    details_url = f"https://api.themoviedb.org/3/movie/{film_id}?api_key={API_KEY}&language=fr-FR"
    return requests.get(details_url).json()

# Fonction pour récupérer le casting d'un film par ID
def obtenir_cast(film_id):
    credits_url = f"https://api.themoviedb.org/3/movie/{film_id}/credits?api_key={API_KEY}&language=fr-FR"
    return requests.get(credits_url).json().get("cast", [])

# Fonction pour afficher l'affiche du film
def afficher_affiche(poster_path, titre):
    if poster_path:
        image_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
        st.image(image_url, caption=titre, use_container_width=True)
    else:
        st.write("Affiche non disponible")

# Fonction principale pour afficher toutes les informations d'un film
def afficher_infos_film(nom_film):
    film = rechercher_film(nom_film)
    if not film:
        st.write("Film non trouvé")
        return

    film_id = film["id"]
    details = obtenir_details_film(film_id)
    cast_list = obtenir_cast(film_id)

    # Affichage
    afficher_affiche(film.get("poster_path"), film.get("title"))
    
    st.subheader("Résumé")
    st.write(details.get("overview", "Pas de résumé disponible."))

    st.write(f"**Date de sortie :** {details.get('release_date', 'Date non disponible')}")

    if cast_list:
        top_cast = [actor["name"] for actor in cast_list[:5]]
        st.write("**Acteurs principaux :**", ", ".join(top_cast))
    else:
        st.write("Acteurs non disponibles")

# Interface Streamlit
st.title("Recherche d'affiche de film")
film = st.text_input("Tape le nom du film:")

if film:
    afficher_infos_film(film)


