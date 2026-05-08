import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true])

# Input and output
x = data["text"]
y = data["label"]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

x_vectorized = vectorizer.fit_transform(x)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x_vectorized, y, test_size=0.2, random_state=42
)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Accuracy
score = accuracy_score(y_test, y_pred)

print("Accuracy:", score)

# Save model
pickle.dump(model, open("news_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully!")