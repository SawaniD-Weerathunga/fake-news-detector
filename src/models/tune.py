import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

from src.preprocessing.text_cleaner import TextCleaner

def main():
    df = pd.read_csv("data/processed/cleaned.csv")   # or your file path

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))

    # Remove NaN values from training data
    X_train = X_train.fillna("")
    X_test = X_test.fillna("")

    X_train_tfidf = tfidf.fit_transform(X_train)

    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "class_weight": ["balanced", None]
    }

    model = LogisticRegression(max_iter=2000)
    grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring="f1_macro")
    grid.fit(X_train_tfidf, y_train)

    print("Best parameters:", grid.best_params_)

    # Save tuned model
    joblib.dump(grid.best_estimator_, "models/log_reg_tuned.pkl")
    joblib.dump(tfidf, "models/tfidf_tuned.pkl")

if __name__ == "__main__":
    main()
