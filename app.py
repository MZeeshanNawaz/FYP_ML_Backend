import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging

# ================= APP SETUP =================
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# ================= LOAD CROP MODEL ONLY =================
logging.info("Loading crop identification model...")
crop_model = tf.keras.models.load_model(
    "models/crop_identifier.h5",
    compile=False
)

# ================= LAZY MODELS =================
wheat_model = None
rice_model = None
corn_model = None

# ================= CLASSES =================
CROP_CLASSES = ["Wheat", "Rice", "Corn"]

WHEAT_CLASSES = [
    "Aphid","Black Rust","Blast","Brown Rust","Common Root Rot",
    "Fusarium Head Blight","Healthy","Leaf Blight","Mildew",
    "Mite","Septoria","Smut","Stem Fly","Tan Spot","Yellow Rust"
]

RICE_CLASSES = ["Bacterial Blight", "Blast", "Brown Spot", "Tungro"]
CORN_CLASSES = ["Common Rust", "Gray Leaf Spot", "Blight", "Healthy"]

# ================= TREATMENT DB =================
# (UNCHANGED — your full database stays exactly as provided)
from treatment_db import TREATMENT_DB
# ⬆️ OPTIONAL: You may keep DB in same file if you prefer

# ================= IMAGE PREPROCESS =================
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# ================= PREDICTION =================
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "Image not provided"}), 400

    img_tensor = preprocess_image(request.files["image"].read())

    # -------- Crop Prediction --------
    crop_probs = crop_model.predict(img_tensor, verbose=0)[0]
    crop_idx = int(np.argmax(crop_probs))
    crop = CROP_CLASSES[crop_idx]

    logging.info(f"Predicted Crop: {crop}")

    global wheat_model, rice_model, corn_model

    # -------- Disease Model Selection --------
    if crop == "Wheat":
        if wheat_model is None:
            logging.info("Loading wheat disease model...")
            wheat_model = tf.keras.models.load_model(
                "models/wheat_disease_model.h5",
                compile=False
            )
        model = wheat_model
        classes = WHEAT_CLASSES

    elif crop == "Rice":
        if rice_model is None:
            logging.info("Loading rice disease model...")
            rice_model = tf.keras.models.load_model(
                "models/rice_disease_model.h5",
                compile=False
            )
        model = rice_model
        classes = RICE_CLASSES

    else:
        if corn_model is None:
            logging.info("Loading corn disease model...")
            corn_model = tf.keras.models.load_model(
                "models/corn_disease_model.h5",
                compile=False
            )
        model = corn_model
        classes = CORN_CLASSES

    # -------- Disease Prediction --------
    probs = model.predict(img_tensor, verbose=0)[0]
    top_idx = np.argsort(probs)[-3:][::-1]

    top3 = [
        {
            "disease": classes[i],
            "confidence": round(float(probs[i]) * 100, 2)
        }
        for i in top_idx
    ]

    best = top3[0]

    # -------- Confidence Fallback --------
    disease = best["disease"] if best["confidence"] >= 60 else "Healthy"

    logging.info(f"Disease: {disease} | Confidence: {best['confidence']}%")

    return jsonify({
        "crop": crop,
        "disease": disease,
        "top3": top3,
        "treatment": TREATMENT_DB.get(disease, TREATMENT_DB["Healthy"])
    })

# ================= HEALTH CHECK =================
@app.route("/")
def health():
    return {"status": "AgriTech ML Backend Running"}

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
