from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords


# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl") 

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Tokenize
    text = [word for word in text if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(text)

@app.route("/")
def home():
    return render_template('index1.html')

@app.route("/predict", methods=["POST"])
def predict():
    return render_template("predict.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data["text"]

    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Convert text to vector (TF-IDF)
    text_vector = vectorizer.transform([processed_text])
    
    # Predict sentiment using the loaded model
    sentiment = model.predict(text_vector)[0]

    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)
