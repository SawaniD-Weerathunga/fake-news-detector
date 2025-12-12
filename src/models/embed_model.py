import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer

def main():
    print("✔ Loading MiniLM model (this takes 2–5 seconds)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("✔ Loading cleaned dataset...")
    df = pd.read_csv("data/processed/cleaned.csv")

    # Drop missing rows
    df = df.dropna(subset=["clean_text"])

    texts = df["clean_text"].tolist()
    labels = df["label"].values

    print(f"✔ Total samples: {len(texts)}")
    print("✔ Generating embeddings in fast batches...")

    # Batch embeddings (very fast even on CPU)
    X = model.encode(texts, batch_size=64, show_progress_bar=True)

    print("✔ Embeddings created!")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, stratify=labels, random_state=42
    )

    print("✔ Training classifier...")
    classifier = LogisticRegression(max_iter=3000)
    classifier.fit(X_train, y_train)

    print("✔ Evaluating model...")
    preds = classifier.predict(X_test)

    print("\n==============================")
    print("   CLASSIFICATION REPORT")
    print("==============================")
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    main()
