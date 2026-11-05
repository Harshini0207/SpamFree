import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load cleaned dataset
df = pd.read_csv("dataset/cleaned_emails.csv")

X = df["email"].astype(str)
y = df["label"]

print("Dataset loaded:", df.shape)

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 3. Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2
    )),
    ("classifier", MultinomialNB())
])

# 4. Train model
print("\nTraining Multinomial Naive Bayes...")

model.fit(X_train, y_train)

print("Model training completed!")

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate model
print("\n========== MODEL EVALUATION ==========")

print("Accuracy:",
      accuracy_score(y_test, y_pred))

print("Precision:",
      precision_score(y_test, y_pred, zero_division=0))

print("Recall:",
      recall_score(y_test, y_pred, zero_division=0))

print("F1 Score:",
      f1_score(y_test, y_pred, zero_division=0))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham", "Spam"],
    zero_division=0
))

# 7. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)

plt.title("Spam Email Detection - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()
plt.savefig("dataset/confusion_matrix.png")
plt.show()

# 8. Save trained model
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/spam_detector.pkl")

print("\nModel saved successfully!")
print("Location: model/spam_detector.pkl")