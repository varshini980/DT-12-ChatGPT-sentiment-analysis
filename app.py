from flask import Flask, render_template, request

app = Flask(__name__, template_folder="templates")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "")

    if not text.strip():
        sentiment = "Neutral 😐"
    else:
        from textblob import TextBlob
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            sentiment = "Positive 😊"
        elif analysis.sentiment.polarity < 0:
            sentiment = "Negative 😞"
        else:
            sentiment = "Neutral 😐"

    return render_template("predict.html", sentiment=sentiment, user_text=text)

if __name__ == "__main__":
    app.run(debug=True)
