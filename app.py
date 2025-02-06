from flask import Flask, render_template, request, jsonify
from textblob import TextBlob

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index1.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"sentiment": "Neutral 😐", "color": "neutral"})

    # Perform sentiment analysis
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        sentiment = "Positive 😊"
        color = "positive"
    elif analysis.sentiment.polarity < 0:
        sentiment = "Negative 😞"
        color = "negative"
    else:
        sentiment = "Neutral 😐"
        color = "neutral"

    return jsonify({"sentiment": sentiment, "color": color})

if __name__ == "__main__":
    app.run(debug=True)
