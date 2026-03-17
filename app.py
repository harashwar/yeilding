from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import json
import os

app = Flask(__name__)
CORS(app)

# -------------------------
# Paths (Render-safe)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "crop_yield_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model", "model_metadata.json")

model = None
metadata = None

# -------------------------
# Load Model + Metadata
# -------------------------
def load_resources():
    global model, metadata

    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("✅ Model loaded")
        else:
            print("❌ Model file not found")

        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH) as f:
                metadata = json.load(f)
            print("✅ Metadata loaded")
        else:
            print("⚠️ Metadata not found")

    except Exception as e:
        print("❌ Error loading resources:", e)

# Load at startup
load_resources()

# -------------------------
# Prediction Function
# -------------------------
def predict_yield(crop, rainfall, temperature):
    try:
        input_df = pd.DataFrame([[crop, rainfall, temperature]],
                                columns=['Crop', 'Rainfall', 'Temperature'])

        prediction = model.predict(input_df)[0]
        return float(prediction), None

    except Exception as e:
        return None, str(e)

# -------------------------
# Helper
# -------------------------
def get_yield_range(pred):
    if metadata and "rf_mae" in metadata:
        margin = metadata["rf_mae"]
    else:
        margin = pred * 0.1

    lower = max(0, pred - margin)
    upper = pred + margin

    return f"{lower:.2f} - {upper:.2f}"

# -------------------------
# Routes
# -------------------------

@app.route("/")
def home():
    return "🚀 Crop Yield API is running"

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()

    if not data:
        return jsonify({"error": "No input data"}), 400

    # Validate input
    try:
        crop = str(data.get("crop"))
        rainfall = float(data.get("rainfall"))
        temperature = float(data.get("temperature"))
    except:
        return jsonify({"error": "Invalid input format"}), 400

    # Predict
    prediction, error = predict_yield(crop, rainfall, temperature)

    if prediction is None:
        return jsonify({"error": error}), 400

    accuracy = None
    if metadata and "rf_r2" in metadata:
        accuracy = round(metadata["rf_r2"] * 100, 2)

    return jsonify({
        "predicted_yield": round(prediction, 2),
        "yield_range": get_yield_range(prediction),
        "unit": "hg/ha",
        "model_accuracy_percent": accuracy,
        "status": "success"
    })

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)