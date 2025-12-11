
import pandas as pd
import nltk
import spacy
import os 

pkl_path = os.path.join('data', 'processed', 'dataframe_ML_final.pkl')
path_export_pkl = os.path.join('data', 'processed', 'dataframe_ready_for_ML.pkl')

nlp = spacy.load('en_core_web_sm')
nltk.download('stopwords')
stopwordsenglish = nltk.corpus.stopwords.words("english")


df = pd.read_pickle(pkl_path)
print(df.head(5).to_markdown())

#lemma sur overview

def clean(text) :# Spacy découpe automatiquement en tokens avec cette syntaxe :
    sent_tokens = nlp(text)
  # Create list
    words_lemma = [
        token.lemma_.lower()
        for token in sent_tokens
        if not token.is_punct
        and not token.is_space
        and token.lemma_.lower() not in stopwordsenglish
    ]
    return " ".join(words_lemma)


print('Application lemma')
df_ml = df.copy()
df_ml['NLP'] = df['overview'].apply(clean)

print('Lemmatize finie')
print(df_ml.head(5).to_markdown())

# Clean les espaces entre les acteurs, producer, writers, directors 
def nettoyage_espace(liste_mots):
    """
    Transforme ['Brad Pitt', 'Sci Fi'] en 'bradpitt scifi'
    """
    if not isinstance(liste_mots, list):
        return ''

    # On met en minuscule et on colle les morceaux de chaque mot
    return " ".join([str(mot).lower().replace(" ", "") for mot in liste_mots])


colonnes_a_traiter = ['genres' , 'actor_actress', 'producer', 'writers', 'directors',
        'production_companies_name']

for col in colonnes_a_traiter:
    df_ml[f'{col}_clean'] = df_ml[col].apply(nettoyage_espace)
    

df_ml.to_pickle(path_export_pkl)

