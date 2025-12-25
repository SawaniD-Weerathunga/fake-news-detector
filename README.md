# 📰 Fake News Detection System (NLP + Machine Learning)

An end-to-end **Fake News Detection** project using **Natural Language Processing (NLP)** and **Machine Learning**, featuring **error analysis**, **model explainability (SHAP & LIME)**, and a **Streamlit web application** for real-time predictions.

---

## 🚀 Project Overview

Fake news spreads rapidly across digital platforms and can strongly influence public opinion.  
This project builds a **complete fake news detection pipeline** that classifies news articles as **Real** or **Fake**, while also explaining *why* the model makes its predictions.

Key goals:
- Build a robust NLP classification model
- Identify and analyze model weaknesses
- Improve fake-news recall using threshold tuning
- Provide explainable AI insights
- Deploy a simple demo UI

---

## 📂 Project Structure
fake-news-detector/
│
├── data/
│ ├── raw/
│ │ ├── Fake.csv
│ │ └── True.csv
│ └── processed/
│ └── cleaned.csv
│
├── src/
│ ├── preprocessing/
│ │ └── text_cleaner.py
│ │
│ ├── models/
│ │ ├── train.py
│ │ └── error_analysis.py
│ │
│ └── api/
│ └── app.py
│
├── models/
│ ├── baseline_logreg.joblib
│ └── tfidf_vectorizer.joblib
│
├── reports/
│ ├── baseline_report.txt
│ ├── false_positives.csv
│ ├── false_negatives.csv
│ └── lime_explanation.html
│
├── requirements.txt
└── README.md


---

## 🧠 Model Architecture

- **Algorithm:** Logistic Regression  
- **Text Representation:** TF-IDF (unigrams & bigrams)  
- **Class Balancing:** Enabled  
- **Decision Threshold:** Tuned for higher fake-news recall  
- **Explainability:** SHAP (global) + LIME (local)

---

## 📊 Model Performance

After class balancing and threshold tuning:

- **Accuracy:** ~99%
- **Fake News Recall:** ~1.00
- **Real News Precision:** ~1.00

> ⚠️ Note: High performance is expected due to strong linguistic and stylistic differences in the dataset.  
> These characteristics are analyzed in the error analysis section.

---

## 🔍 Error Analysis

Automatically generated outputs:
- `reports/false_positives.csv`
- `reports/false_negatives.csv`

Additional insights:
- Text length analysis
- Subject-wise error distribution
- False Positive & False Negative inspection
- Identification of misleading patterns

---

## 🧠 Model Explainability

### 🔹 SHAP
- Explains which words push predictions toward **Fake** or **Real**
- Used for analyzing the most critical misclassifications

### 🔹 LIME
- Generates human-readable explanations
- Highlights influential words and phrases
- Output saved as:


## Reports/lime_explanation.html
---

## 🧪 Training the Model

Run the training script:
python -m src.models.train

This will:
   - Load and preprocess the dataset
   - Train the model
   - Save the trained model & vectorizer
   - Generate evaluation metrics


## Running Error Analysis

Run the script:
python -m src.models.error_analysis

This performs:
   - Probability diagnostics
   - Threshold-based predictions
   - Error analysis (FP/FN)
   - SHAP explanations
   - LIME explanations


## Streamlit Web App

Run the demo UI:
streamlit run src/api/app.py

Features:
   - Paste any news article or headline
   - Predict Fake or Real
   - Display confidence score


## Testing Samples

Long fake and real news articles were used to:
   - Evaluate robustness
   - Record demo videos
   - Validate explainability outputs


## Technologies Used
   - Python
   - pandas, numpy
   - scikit-learn
   - TF-IDF
   - SHAP
   - LIME
   - Streamlit


## Future Improvements
   - Transformer-based models (BERT / MiniLM)
   - Cross-dataset generalization testing
   - Real-time news ingestion
   - Dockerized deployment
   - Cloud hosting


## Author

Sawani Weerathunga

📌 LinkedIn: https://github.com/SawaniD-Weerathunga 

