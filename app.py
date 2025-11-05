from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# ----------------------------
# Initialize Flask app
# ----------------------------
app = Flask(__name__)

# ✅ Allow both local and deployed frontends
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://localhost:8080/monitoring/esp001",
            "https://forest-fire-api2.onrender.com",   # backend itself
            "https://van-dash.vercel.app",
            "https://van-rakshak.netlify.app/"# (optional) replace if you deploy frontend
        ]
    }
}, supports_credentials=True)

# ----------------------------
# Load model and scaler
# ----------------------------
try:
    model = joblib.load("forest_fire_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Model and Scaler loaded successfully.")
except Exception as e:
    print("❌ Error loading model or scaler:", e)
    model, scaler = None, None

# ----------------------------
# Feature list (for validation)
# ----------------------------
USEFUL_FEATURES = [
    "temperature",
    "humidity",
    "smoke",
    "temp_max",
    "temp_min",
    "wind_speed",
    "wind_gust"
]

# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🌲 Vanrakshak Forest Fire Detection API is Live and Secure!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({"error": "Model or scaler not loaded"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # ✅ Convert JSON → DataFrame with exact column order
        input_df = pd.DataFrame([data], columns=USEFUL_FEATURES)

        # ✅ Scale data
        scaled = scaler.transform(input_df)

        # ✅ Predict class
        prediction = int(model.predict(scaled)[0])

        # ✅ Optional probabilities
        prob_dict = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(scaled)[0]
            prob_dict = {str(i): round(float(p), 4) for i, p in enumerate(proba)}

        # ✅ Interpret result
        risk_levels = {
            0: {"level": "Safe", "emoji": "✅", "message": "No fire risk detected."},
            1: {"level": "High Risk", "emoji": "🔥", "message": "Forest fire likely — take action!"},
            2: {"level": "Borderline", "emoji": "⚠️", "message": "Uncertain — monitor closely."}
        }
        result = risk_levels.get(prediction, {"level": "Unknown", "emoji": "❓", "message": "Invalid prediction output."})

        return jsonify({
            "prediction": prediction,
            "level": result["level"],
            "emoji": result["emoji"],
            "message": result["message"],
            "probabilities": prob_dict
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------
# Run app (local or Render)
# ----------------------------
if __name__ == "__main__":
    # On Render, Flask automatically uses PORT env variable
    app.run(host="0.0.0.0", port=5000, debug=True)



