import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os


class TextCleaner:
    def __init__(self, remove_stopwords=True, lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize_words = lemmatize
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text):
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        words = text.split()

        if self.remove_stopwords:
            words = [w for w in words if w not in self.stop_words]

        if self.lemmatize_words:
            words = [self.lemmatizer.lemmatize(w) for w in words]

        return " ".join(words)


def build_tfidf_vectorizer(max_features=20000, ngram_range=(1,2)):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )
    return vectorizer


def save_vectorizer(vectorizer, path="tfidf_vectorizer.joblib"):
    joblib.dump(vectorizer, path)
