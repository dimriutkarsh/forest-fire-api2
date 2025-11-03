from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# ✅ Allow all frontend origins (safe for dev)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Load trained model and scaler
try:
    model = joblib.load("forest_fire_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Model and Scaler loaded successfully.")
except Exception as e:
    print("❌ Error loading model or scaler:", e)

# ✅ Final selected features for Vanrakshak
USEFUL_FEATURES = [
    'temperature',
    'humidity',
    'smoke',
    'temp_max',
    'temp_min',
    'wind_speed',
    'wind_gust'
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🌲 Vanrakshak Forest Fire Detection API is Running Successfully!"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # ✅ Validate input
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # ✅ Convert JSON → DataFrame (keep feature names)
        input_df = pd.DataFrame([data], columns=USEFUL_FEATURES)

        # ✅ Scale input using trained scaler
        scaled = scaler.transform(input_df)

        # ✅ Predict fire risk (0, 1, 2)
        prediction = int(model.predict(scaled)[0])

        # ✅ Handle probability (if supported)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(scaled)[0]
            prob_dict = {str(i): round(float(p), 4) for i, p in enumerate(proba)}
        else:
            prob_dict = None

        # ✅ Interpret prediction meaning
        risk_levels = {
            0: {"level": "Safe", "emoji": "✅", "message": "No fire risk detected."},
            1: {"level": "High Risk", "emoji": "🔥", "message": "Forest fire likely — immediate action advised!"},
            2: {"level": "Borderline", "emoji": "⚠️", "message": "Uncertain condition — monitor closely."}
        }

        result = risk_levels.get(prediction, {"level": "Unknown", "emoji": "❓", "message": "Invalid prediction output."})

        # ✅ Return formatted response
        return jsonify({
            "prediction": prediction,
            "level": result["level"],
            "emoji": result["emoji"],
            "message": result["message"],
            "probabilities": prob_dict
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # ✅ Works locally or on Render
    app.run(host="0.0.0.0", port=5000, debug=True)
