import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def clean(text,lang='en',html=False):
    if html:
        text = re.sub(CLEANR, '', text.lower()) # Quitamos etiquetas HTML
    text = re.sub(r'[^\w\s]', '', text.lower()) # Quitamos signos de puntuación y símbolos
    text = re.sub('[0-9]', '', text.lower())  # Quitamos números
    if lang=='en':
        SW = stopwords.words('english') # Leemos la lista de stopwords del inglés
    elif lang=='es':
        SW = stopwords.words('spanish') # Leemos la lista de stopwords del inglés
    else:
        print(f"'{lang}' is not a valid option")
    tokens_no_sw = [word for word in word_tokenize(text) if not word in SW] # Quitamos stopwords
    stems = []
    for w in tokens_no_sw:
        stems.append(lemmatizer.lemmatize(w))
    return stems