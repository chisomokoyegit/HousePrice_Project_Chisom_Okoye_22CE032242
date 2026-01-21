from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model
model_path = os.path.join("model", "house_price_model.pkl")
with open(model_path, "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]

@app.route("/")
def home():
    return render_template("index.html", prediction="")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["OverallQual"]),
            float(request.form["GrLivArea"]),
            float(request.form["TotalBsmtSF"]),
            float(request.form["GarageCars"]),
            float(request.form["FullBath"]),
            float(request.form["YearBuilt"]),
        ]

        features_scaled = scaler.transform([features])
        price = model.predict(features_scaled)[0]

        return render_template(
            "index.html",
            prediction=f"Estimated House Price: ₦{price:,.2f}"
        )

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


