import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# =========================
# Load Dataset
# =========================

data = pd.read_csv("model/dataset_real.csv")

data.columns = [col.strip().lower() for col in data.columns]

print("Dataset columns:", data.columns.tolist())


# =========================
# Prepare Labels
# =========================

if "label" in data.columns:
    data["label"] = data["label"].astype(str).str.lower().str.strip()

    data["label"] = data["label"].map({
        "legitimate": 0,
        "phishing": 1,
        "0": 0,
        "1": 1
    })

else:
    raise ValueError("Label column not found.")


data = data.dropna(subset=["url", "label"])


X = data["url"].astype(str)
y = data["label"].astype(int)


# =========================
# Split Dataset
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================
# TF-IDF
# =========================

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=100000
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# =========================
# Train Model
# =========================

model = SGDClassifier(
    loss="log_loss",
    random_state=42,
    max_iter=1000,
    tol=1e-3
)

model.fit(X_train_vectorized, y_train)


# =========================
# Prediction
# =========================

y_pred = model.predict(X_test_vectorized)


# =========================
# Evaluation
# =========================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    )
)


# =========================
# Save Model
# =========================

joblib.dump(
    model,
    "model/phishing_model.pkl"
)

joblib.dump(
    vectorizer,
    "model/vectorizer.pkl"
)


print("\nModel and vectorizer saved successfully!")