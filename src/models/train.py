import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from src.preprocessing.text_cleaner import TextCleaner, build_tfidf_vectorizer


def load_data():
    fake = pd.read_csv("data/raw/Fake.csv")
    true = pd.read_csv("data/raw/True.csv")

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], ignore_index=True)
    df = df.sample(frac=1, random_state=42)  # shuffle
    return df


def preprocess(df):
    cleaner = TextCleaner()
    df["clean_text"] = df["text"].apply(cleaner.clean)
    return df


def vectorize(train_texts, test_texts):
    vectorizer = build_tfidf_vectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # Save the vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

    return X_train, X_test


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    joblib.dump(model, "models/baseline_logreg.joblib")
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    roc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    os.makedirs("reports", exist_ok=True)

    with open("reports/baseline_report.txt", "w") as f:
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write(f"ROC-AUC: {roc:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    print("Evaluation complete. See reports/baseline_report.txt")


def main():
    df = load_data()
    df = preprocess(df)

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.15,
        stratify=df["label"],
        random_state=42
    )

    X_train, X_test = vectorize(X_train_texts, X_test_texts)

    model = train_model(X_train, y_train)

    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()
