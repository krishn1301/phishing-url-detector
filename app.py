"""
app.py - Flask Backend for Phishing URL Detection
===================================================
Loads the trained Random Forest model and serves a web interface
for URL classification with risk scoring and safe preview.

Usage:
    python app.py
"""

import os
import logging

import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

from utils import extract_features, safe_preview, FEATURE_NAMES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "phishing_model.pkl")
RISK_THRESHOLD = 0.70  # 70 % — trigger safe preview above this

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Load the trained model at startup
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    app.logger.info("Model loaded from %s", MODEL_PATH)
else:
    model = None
    app.logger.warning(
        "Model file not found at %s. Run train_model.py first.", MODEL_PATH
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a URL, extract features, classify with the model, and
    optionally run the safe preview module.

    Expects JSON: { "url": "<target-url>" }

    Returns JSON:
        {
            "classification": "Phishing" | "Legitimate",
            "risk_percentage": float,
            "features": { name: value, ... },
            "preview": { ... } | null
        }
    """
    # --- Validate input ---
    data = request.get_json(silent=True)
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' field in request body."}), 400

    url = data["url"].strip()
    if not url:
        return jsonify({"error": "URL cannot be empty."}), 400

    # --- Check model availability ---
    if model is None:
        return jsonify({
            "error": "Model not loaded. Please run train_model.py first."
        }), 503

    try:
        # Step 1 — Feature extraction
        features = extract_features(url)
        feature_array = np.array(features).reshape(1, -1)

        # Step 2 — Prediction
        prediction = model.predict(feature_array)[0]
        probabilities = model.predict_proba(feature_array)[0]

        # Risk = probability of class 1 (phishing)
        risk = float(probabilities[1]) * 100  # percentage
        classification = "Phishing" if prediction == 1 else "Legitimate"

        # Build feature dict for the frontend
        feature_dict = {
            name: val for name, val in zip(FEATURE_NAMES, features)
        }

        # Step 3 — Safe preview (only for high-risk URLs)
        preview = None
        if risk >= RISK_THRESHOLD * 100:
            preview = safe_preview(url)

        return jsonify({
            "classification": classification,
            "risk_percentage": round(risk, 2),
            "features": feature_dict,
            "preview": preview,
        })

    except Exception as exc:
        app.logger.exception("Prediction failed for URL: %s", url)
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
