from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoTokenizer,AutoConfig
import numpy as np
from scipy.special import softmax

app = Flask(__name__)


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

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

        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        result = config.id2label[ranking[0]]
        return jsonify({"sentiment": result})
    
    except Exception as e:
        print("Error:", str(e))  # Debugging
        return jsonify({"error": "Internal Server Error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
