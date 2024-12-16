import pandas as pd

# Load the dataset
df = pd.read_csv("BBCNews.csv")

# Display the first few rows
print(df.head())

# Display column names
print("Column names:", df.columns)

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Display basic information
print(df.info())

####
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Drop the unnecessary column
df = df.drop(columns=['Unnamed: 0'])

# Encode the target variable (categories in 'tags')
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['tags'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['descr'],  # News article text
    df['category_encoded'],
    test_size=0.2,
    random_state=42
)

# Check the split
print("Training data size:", len(X_train))
print("Testing data size:", len(X_test))

####
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Convert text data to numerical features using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train a Logistic Regression model
clf = LogisticRegression(max_iter=200)
clf.fit(X_train_tfidf, y_train)

# Make predictions and evaluate
y_pred = clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

#####

import pickle

# Save the trained model and vectorizer
with open("bbc_tfidf_model.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# Save the label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Model, vectorizer, and encoder saved!")

####

from flask import Flask, request, jsonify
import pickle

# Load the model, vectorizer, and encoder
clf = pickle.load(open("bbc_tfidf_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input text
        input_text = request.json['text']

        # Transform input text using TF-IDF
        input_vectorized = tfidf.transform([input_text])

        # Predict category
        prediction = clf.predict(input_vectorized)
        predicted_label = label_encoder.inverse_transform(prediction)

        # Return the first category label (assuming it's one label)
        return jsonify({'category': predicted_label[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

    if not input_text.strip():
        return jsonify({'error': 'Input text cannot be empty'}), 400


if __name__ == '__main__':
    app.run(debug=True)
