from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("yorum_modeli_brf.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    yorum = data.get("content", "")
    X = vectorizer.transform([yorum])
    tahmin = int(model.predict(X)[0])
    return jsonify({"predicted_star": tahmin})

if __name__ == "__main__":
    app.run(port=5000)