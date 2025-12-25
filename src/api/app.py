#main code
import sys
import os

sys.path.append(os.path.abspath("."))

# ---------------------------
# Fix Python path
# ---------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)

# ---------------------------
# Imports
# ---------------------------
import streamlit as st
import joblib

from src.preprocessing.text_cleaner import TextCleaner


# ---------------------------
# Load model & vectorizer
# ---------------------------
MODEL_PATH = "models/baseline_logreg.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

cleaner = TextCleaner()

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.write("Paste a news article or headline below to classify it.")

text = st.text_area("News text", height=250)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        clean_text = cleaner.clean(text)
        X = vectorizer.transform([clean_text])

        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        label = "REAL 🟢" if pred == 1 else "FAKE 🔴"
        confidence = proba[pred]

        st.subheader("Prediction")
        st.write(label)

        st.subheader("Confidence")
        st.write(f"{confidence:.2f}")
