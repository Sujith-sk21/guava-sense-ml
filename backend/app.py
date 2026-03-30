"""
Guava Plant Disease Prediction Backend API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import pickle
import os
from skimage import feature
from skimage.color import rgb2gray
from scipy import stats
import base64
from io import BytesIO
from PIL import Image
import traceback

# ✅ CREATE APP FIRST
app = Flask(__name__)
CORS(app)


# ============================================
# Feature Extractor (same as yours)
# ============================================
class MultiDomainFeatureExtractor:
    def __init__(self, img_size=128):
        self.img_size = img_size

    def extract_all_features(self, image):
        return np.concatenate([
            self.extract_rgb_features(image),
            self.extract_hsv_features(image),
            self.extract_texture_features(image),
            self.extract_edge_features(image),
            self.extract_histogram_features(image)
        ])

    def extract_rgb_features(self, image):
        features = []
        for c in range(3):
            ch = image[:, :, c]
            features.extend([
                np.mean(ch), np.std(ch), np.median(ch),
                np.percentile(ch, 25), np.percentile(ch, 75),
                np.min(ch), np.max(ch), np.ptp(ch),
                stats.skew(ch.flatten()), stats.kurtosis(ch.flatten())
            ])
        return np.array(features)

    def extract_hsv_features(self, image):
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        features = []
        for c in range(3):
            ch = hsv[:, :, c]
            features.extend([
                np.mean(ch), np.std(ch), np.median(ch)
            ])
        return np.array(features)

    def extract_texture_features(self, image):
        gray = rgb2gray(image)
        return np.array([np.mean(gray), np.std(gray)])

    def extract_edge_features(self, image):
        gray = rgb2gray(image)
        g8 = (gray * 255).astype(np.uint8)
        edges = cv2.Canny(g8, 50, 150)
        return np.array([np.sum(edges > 0)])

    def extract_histogram_features(self, image):
        gray = rgb2gray(image)
        hist, _ = np.histogram(gray, bins=32)
        return hist


# ============================================
# Load Model
# ============================================
MODEL_PATH = 'guava_disease_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        selected_features = model_data['selected_features']
        label_map = model_data['label_map']
    print("✅ Model loaded")
except:
    model = None
    selected_features = None
    label_map = None

extractor = MultiDomainFeatureExtractor()


# ============================================
# Helper Functions
# ============================================
def preprocess_image(image_data):
    if isinstance(image_data, str):
        image_data = image_data.split('base64,')[-1]
        img = Image.open(BytesIO(base64.b64decode(image_data)))
    else:
        img = Image.open(image_data)

    img = img.convert('RGB')
    img = np.array(img)
    img = cv2.resize(img, (128, 128)) / 255.0
    return img


def extract_features(image):
    all_features = extractor.extract_all_features(image)
    selected = all_features[selected_features]
    return selected.reshape(1, -1)


# ============================================
# Routes
# ============================================
@app.route('/')
def home():
    return jsonify({"status": "online"})


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        if 'file' in request.files:
            image_data = request.files['file']
        else:
            image_data = request.get_json().get('image')

        img = preprocess_image(image_data)
        features = extract_features(img)

        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        return jsonify({
            "class": label_map[pred],
            "confidence": float(prob[pred])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# RUN SERVER (FINAL FIX)
# ============================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render uses dynamic port
    app.run(host="0.0.0.0", port=port)
