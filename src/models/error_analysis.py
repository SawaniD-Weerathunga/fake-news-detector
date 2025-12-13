import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap

# =============================
# CONFIG
# =============================
DATA_PATH = "data/processed/cleaned.csv"
MODEL_PATH = "models/log_reg_tuned.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"

FP_PATH = "reports/false_positives.csv"
FN_PATH = "reports/false_negatives.csv"

# =============================
# LOAD DATA
# =============================
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["clean_text", "label"])

print(f"✓ Data loaded: {len(df)} rows")

# =============================
# LABELS
# =============================
y = df["label"].values

# =============================
# TRAIN / TEST SPLIT (DATAFRAME SAFE)
# =============================
df_train, df_test, y_train, y_test = train_test_split(
    df,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =============================
# LOAD MODEL & VECTORIZER
# =============================
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# =============================
# VECTORIZE TEXT
# =============================
X_train = vectorizer.transform(df_train["clean_text"])
X_test = vectorizer.transform(df_test["clean_text"])

# =============================
# PREDICT
# =============================
preds = model.predict(X_test)

# =============================
# BUILD ERROR DATAFRAME (NO INDEX BUGS)
# =============================
errors = pd.DataFrame({
    "text": df_test["clean_text"].values,
    "true": y_test,
    "pred": preds,
    "subject": df_test["subject"].values if "subject" in df_test.columns else "unknown"
})

# =============================
# TEXT LENGTH ANALYSIS
# =============================
errors["length"] = errors["text"].str.split().str.len()

# =============================
# FALSE POSITIVES / FALSE NEGATIVES
# =============================
fp = errors[(errors["true"] == 0) & (errors["pred"] == 1)]
fn = errors[(errors["true"] == 1) & (errors["pred"] == 0)]

# =============================
# SAVE RESULTS
# =============================
fp.to_csv(FP_PATH, index=False)
fn.to_csv(FN_PATH, index=False)

print(f"\nSaved {len(fp)} False Positives → {FP_PATH}")
print(f"Saved {len(fn)} False Negatives → {FN_PATH}")

# =============================
# CLASSIFICATION REPORT
# =============================
print("\n=== Classification Report ===")
print(classification_report(y_test, preds))

# =============================
# LENGTH ANALYSIS
# =============================
print("\nAverage text length (words):")
print("All:", errors["length"].mean())
print("FP :", fp["length"].mean())
print("FN :", fn["length"].mean())

# =============================
# SUBJECT ANALYSIS
# =============================
print("\nErrors by subject:")
print(errors.groupby("subject").size().sort_values(ascending=False))

# =============================
# SHAP EXPLAINABILITY (SAFE MODE)
# =============================
print("\nRunning SHAP analysis (first 5 errors)...")

explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)

for i in range(min(5, len(errors))):
    print("\n------------------------------")
    print("TEXT:")
    print(errors.iloc[i]["text"][:500], "...")

# =============================
# LIME EXPLANATION (TEXT-LEVEL)
# =============================
from lime.lime_text import LimeTextExplainer

print("\nRunning LIME explanation...")

# Class names must match label encoding
explainer_lime = LimeTextExplainer(class_names=["Real", "Fake"])

# Wrapper function for LIME
def predict_proba_lime(texts):
    X = vectorizer.transform(texts)
    return model.predict_proba(X)

# Choose one error to explain
if len(fp) > 0:
    i = fp.index[0]
    error_type = "FALSE POSITIVE"
elif len(fn) > 0:
    i = fn.index[0]
    error_type = "FALSE NEGATIVE"
else:
    print("No errors found for LIME explanation.")
    exit()

print(f"\nExplaining {error_type}")
print("TEXT:\n", errors.loc[i, "text"][:500], "...")

# Generate explanation
exp = explainer_lime.explain_instance(
    errors.loc[i, "text"],
    predict_proba_lime,
    num_features=10
)

# Open explanation in browser

# =============================
# LIME EXPLANATION (SAVE HTML)
# =============================
from lime.lime_text import LimeTextExplainer
import webbrowser
import os

print("\nRunning LIME explanation...")

explainer_lime = LimeTextExplainer(class_names=["Real", "Fake"])

def predict_proba_lime(texts):
    X = vectorizer.transform(texts)
    return model.predict_proba(X)

# Pick one error
if len(fp) > 0:
    i = fp.index[0]
    error_type = "FALSE POSITIVE"
elif len(fn) > 0:
    i = fn.index[0]
    error_type = "FALSE NEGATIVE"
else:
    print("No errors found for LIME explanation.")
    exit()

print(f"\nExplaining {error_type}")
print(errors.loc[i, "text"][:500], "...")

exp = explainer_lime.explain_instance(
    errors.loc[i, "text"],
    predict_proba_lime,
    num_features=10
)

# Save explanation
os.makedirs("reports", exist_ok=True)
lime_path = "reports/lime_explanation.html"
exp.save_to_file(lime_path)

print(f"\n✓ LIME explanation saved to {lime_path}")

# Open in browser
webbrowser.open("file://" + os.path.abspath(lime_path))

