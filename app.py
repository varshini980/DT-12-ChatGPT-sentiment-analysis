from flask import Flask, render_template, request, jsonify
from textblob import TextBlob

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index1.html')

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")
    elif request.method == "POST":
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
    # Use a different port (5001) to avoid conflicts
    app.run(debug=True, host="0.0.0.0", port=5001)
