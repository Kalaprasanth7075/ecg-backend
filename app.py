from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# ----------------------------
# LOAD MODEL
# ----------------------------
print("Current directory:", os.getcwd())

model = tf.keras.models.load_model("ecg_model.keras")

IMG_SIZE = 224

# Must match training order
classes = ['AH', 'H_MI', 'MI', 'Non-ECG', 'Normal']

# ----------------------------
# HOME ROUTE
# ----------------------------
@app.route("/")
def home():
    return "ECG Heart Disease Detection API Running"

# ----------------------------
# PREDICTION ROUTE
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]

        # Read image
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        # Preprocess
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        preds = model.predict(img)
        pred_index = np.argmax(preds)
        confidence = float(np.max(preds))

        predicted_class = classes[pred_index]

        # 🔥 REJECTION LOGIC
        if predicted_class == "Non-ECG" or confidence < 0.6:
            return jsonify({
                "prediction": "Not an ECG image",
                "confidence": round(confidence * 100, 2),
                "probabilities": {
                    classes[i]: float(preds[0][i] * 100)
                    for i in range(len(classes))
                }
            })

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                classes[i]: float(preds[0][i] * 100)
                for i in range(len(classes))
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ----------------------------
# RUN SERVER
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)