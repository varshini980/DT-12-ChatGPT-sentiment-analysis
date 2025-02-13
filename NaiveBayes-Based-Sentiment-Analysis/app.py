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
model = joblib.load("model.pickle")
vectorizer = joblib.load("vectorizer.pickle") 

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
    try:
        data = request.get_json()
        print("Received data:", data)  # Debugging

        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Preprocess text
        processed_text = preprocess_text(text)
        print("Processed text:", processed_text)  # Debugging

        # Convert text to vector (TF-IDF)
        text_vector = vectorizer.transform([processed_text])

        # Predict sentiment using the loaded model
        sentiment = int(model.predict(text_vector)[0])
        if sentiment == 1:
            sentiment = "Neutral"
        elif sentiment == 0:
            sentiment = "Positive"
        elif sentiment == 2:
            sentiment = "Negative"
        print("Predicted sentiment:", sentiment)  # Debugging

        return jsonify({"sentiment": sentiment})
    
    except Exception as e:
        print("Error:", str(e))  # Debugging
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
