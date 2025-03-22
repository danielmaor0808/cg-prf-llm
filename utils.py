import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from config import VERBOSE_LOGGING


def log(message: str):
    """
    Print a message only if VERBOSE_LOGGING is set to True.
    """
    if VERBOSE_LOGGING:
        print(message)


# Setup resources for preprocessing functions
def setup_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')


def get_stopwords():
    return set(stopwords.words('english'))


def get_stemmer():
    return PorterStemmer()


# Define a preprocessing function
def preprocess_text(text):

    # Tokenize the text
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize

    # Retain only alphabetic words (filter out punctuation, numbers, etc.)
    tokens = [word for word in tokens if word.isalpha()]

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Apply stemming
    tokens = [stemmer.stem(word) for word in tokens]

    # Join tokens back into a single string
    return " ".join(tokens)


def concat_top_docs(docs: List[str]) -> str:
    """
    Join a list of docs into a single string for further expansion or BM25 scoring.
    """
    return " ".join(docs)

def binarize_ratings(ratings: List[float], threshold=1.0) -> List[int]:
    """
    Convert numeric ratings to binary for average_precision_score, etc.
    rating <= threshold becomes 0, else 1.
    """
    return [1 if rating > threshold else 0 for rating in ratings]