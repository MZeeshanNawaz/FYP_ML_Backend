from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

# ================= LOAD MODELS =================
crop_model = tf.keras.models.load_model("models/crop_identifier.h5", compile=False)
wheat_model = tf.keras.models.load_model("models/wheat_disease_model.h5", compile=False)
rice_model  = tf.keras.models.load_model("models/rice_disease_model.h5", compile=False)
corn_model  = tf.keras.models.load_model("models/corn_disease_model.h5", compile=False)

# ================= CLASSES =================
CROP_CLASSES = ["Wheat", "Rice", "Corn"]

WHEAT_CLASSES = [
    "Aphid","Black Rust","Blast","Brown Rust","Common Root Rot",
    "Fusarium Head Blight","Healthy","Leaf Blight","Mildew",
    "Mite","Septoria","Smut","Stem Fly","Tan Spot","Yellow Rust"
]

RICE_CLASSES = ["Bacterial Blight", "Blast", "Brown Spot", "Tungro"]

CORN_CLASSES = ["Common Rust", "Gray Leaf Spot", "Blight", "Healthy"]

# ================= TREATMENT DATA =================
TREATMENT_DB = {

    # ================= WHEAT =================
    "Aphid": {
        "spray": "Imidacloprid or Neem Oil",
        "fertilizer": "Potassium-rich fertilizer",
        "advice": (
            "Inspect crop weekly, spray neem oil at early stage, and avoid excess nitrogen fertilizer. "
            "(فصل کا ہفتہ وار معائنہ کریں، ابتدائی مرحلے پر نیم کا تیل سپرے کریں، "
            "اور نائٹروجن کھاد زیادہ استعمال نہ کریں)"
        )
    },

    "Black Rust": {
        "spray": "Propiconazole or Tebuconazole",
        "fertilizer": "Balanced NPK",
        "advice": (
            "Spray fungicide immediately after first symptoms and use rust-resistant wheat varieties next season. "
            "(علامات ظاہر ہوتے ہی سپرے کریں اور آئندہ موسم میں زنگ سے محفوظ اقسام کاشت کریں)"
        )
    },

    "Blast": {
        "spray": "Tricyclazole",
        "fertilizer": "Silicon-based fertilizer",
        "advice": (
            "Avoid over-irrigation, apply silicon fertilizer, and keep field dry during early growth. "
            "(زیادہ پانی نہ دیں، سلیکان کھاد استعمال کریں، اور ابتدائی نشوونما میں کھیت خشک رکھیں)"
        )
    },

    "Brown Rust": {
        "spray": "Mancozeb",
        "fertilizer": "Balanced NPK",
        "advice": (
            "Apply fungicide early, monitor leaf color, and avoid late nitrogen application. "
            "(ابتدائی مرحلے پر سپرے کریں، پتوں کی حالت دیکھتے رہیں، اور آخری وقت نائٹروجن نہ دیں)"
        )
    },

    "Common Root Rot": {
        "spray": "Carbendazim",
        "fertilizer": "Organic compost",
        "advice": (
            "Improve drainage, avoid waterlogging, and use organic compost for stronger roots. "
            "(پانی کے نکاس کو بہتر کریں، کھیت میں پانی کھڑا نہ ہونے دیں، اور نامیاتی کھاد استعمال کریں)"
        )
    },

    "Fusarium Head Blight": {
        "spray": "Tebuconazole",
        "fertilizer": "Potassium fertilizer",
        "advice": (
            "Avoid irrigation during flowering stage and spray fungicide before grain formation. "
            "(پھول آنے کے دوران پانی نہ دیں اور دانہ بننے سے پہلے سپرے کریں)"
        )
    },

    "Leaf Blight": {
        "spray": "Chlorothalonil",
        "fertilizer": "Phosphorus-rich fertilizer",
        "advice": (
            "Remove infected plant debris and spray fungicide at 7–10 day interval. "
            "(متاثرہ پودوں کی باقیات ہٹا دیں اور 7 سے 10 دن کے وقفے سے سپرے کریں)"
        )
    },

    "Mildew": {
        "spray": "Sulfur fungicide",
        "fertilizer": "Balanced nutrients",
        "advice": (
            "Ensure proper spacing between plants and improve air circulation in the field. "
            "(پودوں کے درمیان مناسب فاصلہ رکھیں اور ہوا کی آمدورفت بہتر بنائیں)"
        )
    },

    "Mite": {
        "spray": "Abamectin",
        "fertilizer": "Potassium fertilizer",
        "advice": (
            "Avoid drought stress, maintain soil moisture, and spray acaricide when infestation starts. "
            "(پانی کی کمی سے بچائیں، نمی برقرار رکھیں، اور حملہ شروع ہوتے ہی سپرے کریں)"
        )
    },

    "Septoria": {
        "spray": "Azoxystrobin",
        "fertilizer": "Moderate nitrogen",
        "advice": (
            "Rotate crops every season and avoid dense planting to reduce humidity. "
            "(ہر موسم میں فصل تبدیل کریں اور گھنی کاشت سے پرہیز کریں)"
        )
    },

    "Smut": {
        "spray": "Seed treatment fungicide",
        "fertilizer": "Organic compost",
        "advice": (
            "Always treat seeds before sowing and never use infected seeds. "
            "(بیج بونے سے پہلے ٹریٹمنٹ کریں اور متاثرہ بیج استعمال نہ کریں)"
        )
    },

    "Stem Fly": {
        "spray": "Lambda-cyhalothrin",
        "fertilizer": "Balanced nitrogen",
        "advice": (
            "Sow crop early and remove infected plants immediately from field. "
            "(فصل جلدی کاشت کریں اور متاثرہ پودے فوراً کھیت سے نکال دیں)"
        )
    },

    "Tan Spot": {
        "spray": "Mancozeb",
        "fertilizer": "Potassium-rich fertilizer",
        "advice": (
            "Plow crop residues deeply and avoid continuous wheat cultivation. "
            "(فصل کی باقیات کو گہرا ہل چلائیں اور مسلسل گندم کاشت نہ کریں)"
        )
    },

    "Yellow Rust": {
        "spray": "Propiconazole",
        "fertilizer": "Balanced NPK",
        "advice": (
            "Spray at early yellow patches and prefer resistant varieties in next crop. "
            "(پیلاہٹ ظاہر ہوتے ہی سپرے کریں اور آئندہ مزاحم اقسام لگائیں)"
        )
    },

    # ================= RICE =================
    "Bacterial Blight": {
        "spray": "Streptocycline + Copper",
        "fertilizer": "Potassium fertilizer",
        "advice": (
            "Avoid stagnant water, reduce nitrogen, and spray antibiotics early. "
            "(کھیت میں پانی کھڑا نہ ہونے دیں، نائٹروجن کم کریں، اور بروقت سپرے کریں)"
        )
    },

    "Brown Spot": {
        "spray": "Mancozeb",
        "fertilizer": "Balanced nitrogen",
        "advice": (
            "Correct soil nutrient deficiency and maintain proper plant nutrition. "
            "(مٹی کی غذائی کمی دور کریں اور مناسب خوراک فراہم کریں)"
        )
    },

    "Tungro": {
        "spray": "Imidacloprid",
        "fertilizer": "Balanced nutrients",
        "advice": (
            "Control leafhopper insects and remove infected plants immediately. "
            "(لیفسوپر کیڑوں پر قابو رکھیں اور متاثرہ پودے فوراً نکال دیں)"
        )
    },

    # ================= CORN =================
    "Common Rust": {
        "spray": "Propiconazole",
        "fertilizer": "Potassium-rich fertilizer",
        "advice": (
            "Spray fungicide early and avoid dense planting. "
            "(ابتدائی مرحلے پر سپرے کریں اور گھنی کاشت سے بچیں)"
        )
    },

    "Gray Leaf Spot": {
        "spray": "Azoxystrobin",
        "fertilizer": "Balanced nitrogen",
        "advice": (
            "Rotate crops and plow infected residues after harvest. "
            "(فصل تبدیل کریں اور کٹائی کے بعد باقیات کو ہل چلائیں)"
        )
    },

    "Blight": {
        "spray": "Mancozeb",
        "fertilizer": "Phosphorus-rich fertilizer",
        "advice": (
            "Avoid overhead irrigation and spray fungicide at early stage. "
            "(اوپر سے پانی نہ دیں اور ابتدائی مرحلے پر سپرے کریں)"
        )
    },

    # ================= HEALTHY =================
    "Healthy": {
        "spray": "Not required",
        "fertilizer": "As per soil test",
        "advice": (
            "Continue regular monitoring, balanced fertilization, and timely irrigation. "
            "(باقاعدہ نگرانی جاری رکھیں، متوازن کھاد دیں، اور وقت پر پانی لگائیں)"
        )
    }
}



# ================= IMAGE PREPROCESS =================
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    img_tensor = preprocess_image(request.files["image"].read())

    crop_probs = crop_model.predict(img_tensor)[0]
    crop_idx = np.argmax(crop_probs)
    crop = CROP_CLASSES[crop_idx]

    model, classes = (
        (wheat_model, WHEAT_CLASSES) if crop == "Wheat" else
        (rice_model, RICE_CLASSES) if crop == "Rice" else
        (corn_model, CORN_CLASSES)
    )

    probs = model.predict(img_tensor)[0]
    top_idx = np.argsort(probs)[-3:][::-1]

    top3 = [{
        "disease": classes[i],
        "confidence": round(float(probs[i]) * 100, 2)
    } for i in top_idx]

    best = top3[0]
    disease = best["disease"] if best["confidence"] >= 60 else "Healthy"

    return jsonify({
        "crop": crop,
        "disease": disease,
        "top3": top3,
        "treatment": TREATMENT_DB.get(disease)
    })

if __name__ == "__main__":
    app.run(debug=True)
