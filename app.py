from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
CORS(app)

print("🚀 Starting app...")
print("📁 Current directory:", os.getcwd())

# ----------------------------
# LOAD MODEL (SAFE)
# ----------------------------
try:
    model = tf.keras.models.load_model("ecg_model.h5")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    model = None

IMG_SIZE = 224
classes = ['AH', 'H_MI', 'MI', 'Non-ECG', 'Normal']

# ----------------------------
# HOME ROUTE
# ----------------------------
@app.route("/")
def home():
    return "✅ ECG API is running"

# ----------------------------
# HEALTH CHECK (IMPORTANT)
# ----------------------------
@app.route("/health")
def health():
    if model is None:
        return jsonify({"status": "model not loaded"})
    return jsonify({"status": "ok"})

# ----------------------------
# PREDICTION ROUTE
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded on server"})

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]

        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)
        pred_index = np.argmax(preds)
        confidence = float(np.max(preds))

        predicted_class = classes[pred_index]

        if predicted_class == "Non-ECG" or confidence < 0.6:
            return jsonify({
                "prediction": "Not an ECG image",
                "confidence": round(confidence * 100, 2)
            })

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"error": str(e)})

# ----------------------------
# RUN SERVER (RENDER FIX)
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)