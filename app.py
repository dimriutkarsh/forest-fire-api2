from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# ===========================================
# 🌲 VANRAKSHAK FOREST FIRE DETECTION API
# ===========================================

# Initialize Flask app
app = Flask(__name__)

# ✅ Allow all frontend origins (development mode)
# You can replace "*" with your frontend domain (e.g., "https://vanrakshak.vercel.app") when deployed
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ✅ Add manual headers (for browsers that ignore CORS lib)
@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    return response


# ===========================================
# 🎯 Model + Scaler Loading
# ===========================================
try:
    model = joblib.load("forest_fire_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Model and Scaler loaded successfully.")
except Exception as e:
    print("❌ Error loading model or scaler:", e)


# ✅ Final selected features for Vanrakshak AI model
USEFUL_FEATURES = [
    "temperature",
    "humidity",
    "smoke",
    "temp_max",
    "temp_min",
    "wind_speed",
    "wind_gust",
]


# ===========================================
# 🌐 ROUTES
# ===========================================

@app.route("/", methods=["GET"])
def home():
    """Home route to confirm server status"""
    return jsonify(
        {"message": "🌲 Vanrakshak Forest Fire Detection API is Running Successfully!"}
    )


@app.route("/predict", methods=["POST"])
def predict():
    """AI-based forest fire risk prediction endpoint"""
    try:
        data = request.get_json()

        # ✅ Validate input
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # ✅ Convert JSON → DataFrame with defined feature order
        input_df = pd.DataFrame([data], columns=USEFUL_FEATURES)

        # ✅ Scale input using trained scaler
        scaled = scaler.transform(input_df)

        # ✅ Predict fire risk level (model output: 0, 1, or 2)
        prediction = int(model.predict(scaled)[0])

        # ✅ Handle probability (if classifier supports it)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(scaled)[0]
            prob_dict = {str(i): round(float(p), 4) for i, p in enumerate(proba)}
        else:
            prob_dict = None

        # ✅ Interpret prediction results
        risk_levels = {
            0: {
                "level": "Safe",
                "emoji": "✅",
                "message": "No fire risk detected.",
            },
            1: {
                "level": "High Risk",
                "emoji": "🔥",
                "message": "Forest fire likely — immediate action advised!",
            },
            2: {
                "level": "Borderline",
                "emoji": "⚠️",
                "message": "Uncertain condition — monitor closely.",
            },
        }

        result = risk_levels.get(
            prediction,
            {"level": "Unknown", "emoji": "❓", "message": "Invalid prediction output."},
        )

        # ✅ Construct and return JSON response
        return jsonify(
            {
                "prediction": prediction,
                "level": result["level"],
                "emoji": result["emoji"],
                "message": result["message"],
                "probabilities": prob_dict,
            }
        )

    except Exception as e:
        print("❌ Error in /predict:", e)
        return jsonify({"error": str(e)}), 500


# ===========================================
# 🚀 RUN APP
# ===========================================
if __name__ == "__main__":
    # ✅ Render uses $PORT env variable automatically
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
