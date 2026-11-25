
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 1. Load dataset
data = pd.read_csv("data/enquiries.csv")

X = data["message"]
y_service = data["service"]
y_priority = data["priority"]


# 2. Split data into training and testing sets
X_train, X_test, y_service_train, y_service_test = train_test_split(
    X, y_service, test_size=0.25, random_state=42, stratify=y_service
)

_, _, y_priority_train, y_priority_test = train_test_split(
    X, y_priority, test_size=0.25, random_state=42, stratify=y_priority
)


# 3. Create service classification model
service_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

service_model.fit(X_train, y_service_train)

service_predictions = service_model.predict(X_test)


# 4. Create priority classification model
priority_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

priority_model.fit(X_train, y_priority_train)

priority_predictions = priority_model.predict(X_test)


# 5. Evaluate models
print("\nSERVICE CLASSIFICATION RESULTS")
print("Accuracy:", accuracy_score(y_service_test, service_predictions))
print(classification_report(y_service_test, service_predictions, zero_division=0))

print("\nPRIORITY CLASSIFICATION RESULTS")
print("Accuracy:", accuracy_score(y_priority_test, priority_predictions))
print(classification_report(y_priority_test, priority_predictions, zero_division=0))


# 6. Save trained models
joblib.dump(service_model, "models/service_model.pkl")
joblib.dump(priority_model, "models/priority_model.pkl")

print("\nBoth models trained and saved successfully!")
