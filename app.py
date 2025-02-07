from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
with open("sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index1.html')

@app.route("/predict", methods=["POST"])
def predict():
    return render_template("predict.html")
def analyze():
    data = request.get_json()
    text = data["text"]
    
    
    # Predict sentiment
    sentiment = model.predict(text)[0]

    return jsonify({"sentiment": sentiment})
if __name__ == "__main__":
    app.run(debug=True)
