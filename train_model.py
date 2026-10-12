import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Load dataset
df = pd.read_csv("tweets.csv")
df = df[["text", "target"]].dropna()

X = df["text"].astype(str)
y = df["target"].astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert text to numerical features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluate
predictions = model.predict(X_test_tfidf)
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification report:")
print(classification_report(y_test, predictions, target_names=["Non-disaster", "Disaster"]))

ConfusionMatrixDisplay.from_predictions(
    y_test, predictions, display_labels=["Non-disaster", "Disaster"], cmap="Blues"
)
plt.title("Disaster Tweet Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# Save model and vectorizer
joblib.dump(model, "disaster_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nSaved: disaster_model.pkl")
print("Saved: tfidf_vectorizer.pkl")
print("Saved: confusion_matrix.png")
