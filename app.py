import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- MEMORY OPTIMIZATION ---
# Disable GPU (Render free tier doesn't have one) and reduce logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import TensorFlow after setting environment variables
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Global variable for the model
model = None
MODEL_PATH = "ecg_model.h5"
IMG_SIZE = 224
classes = ['AH', 'H_MI', 'MI', 'Non-ECG', 'Normal']

def load_model_if_needed():
    """Loads the model only when the first request arrives to save startup RAM."""
    global model
    if model is None:
        print("🚀 Loading model...")
        # compile=False saves memory and prevents errors if custom metrics were used
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully")
    return model

@app.route("/")
def home():
    return "ECG Backend is Live"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Check if file exists
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        
        # 2. Lazy load the model
        current_model = load_model_if_needed()

        # 3. Process Image
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to float32 to save memory compared to float64
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 4. Run Prediction
        preds = current_model.predict(img_array)
        pred_index = np.argmax(preds)
        confidence = float(np.max(preds))
        predicted_class = classes[pred_index]

        # 5. Return Results
        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                classes[i]: float(preds[0][i] * 100)
                for i in range(len(classes))
            }
        })

    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        return jsonify({"error": "Failed to process image. Server might be out of memory."}), 500

if __name__ == "__main__":
    # Get port from environment for Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)