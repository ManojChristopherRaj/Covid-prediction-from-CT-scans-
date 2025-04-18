import os
import sys
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, session
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# ✅ Import updated Grad-CAM with severity function
from classifier.grad_cam import get_gradcam_and_severity

app = Flask(__name__)
app.secret_key = "your_secret_key_here"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "classifier", "random_forest.pkl")
print(f"Model Path: {MODEL_PATH}")
rf_model = joblib.load(MODEL_PATH)
print("Random Forest model loaded successfully!")

resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
print("ResNet50 model loaded for feature extraction!")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print(f"Upload folder: {UPLOAD_FOLDER}")

def extract_features(img_path):
    try:
        print(f"Extracting features from image: {img_path}")
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = resnet_model.predict(img_array)
        print("Features extracted successfully!")
        return features.reshape(1, -1)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Update the route to include severity score calculation
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        print("File upload request received!")

        if "file" not in request.files:
            print("No file uploaded!")
            return render_template("index.html", error="No file uploaded!")

        file = request.files["file"]
        if file.filename == "":
            print("No file selected!")
            return render_template("index.html", error="No file selected!")

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        print(f"Image saved at: {filepath}")

        features = extract_features(filepath)
        if features is None:
            print("Error extracting features!")
            return render_template("index.html", error="Error processing image!")

        # Get prediction and confidence score
        probabilities = rf_model.predict_proba(features)
        prediction = rf_model.predict(features)[0]
        confidence = max(probabilities[0]) * 100  
        print(f"Prediction: {prediction} | Confidence: {confidence:.2f}%")

        # Use severity from Grad-CAM instead of 100 - confidence
        # Use severity from Grad-CAM instead of 100 - confidence
        heatmap_path, severity = get_gradcam_and_severity(filepath)
        severity_score = float(severity.replace('%', ''))  # Remove the '%' before converting to float
        print(f"Severity score (from Grad-CAM): {severity_score:.2f}%")

        label_map = {1: "COVID", 0: "Non-COVID"}
        predicted_label = label_map[prediction]

        print("Generating Grad-CAM heatmap...")
        heatmap_filename = os.path.basename(heatmap_path)  # No need to call it again
        print(f"Heatmap saved at: {heatmap_path}")

        # Save session variables
        session["image"] = file.filename
        session["prediction"] = predicted_label
        session["confidence"] = f"{confidence:.2f}%"
        session["heatmap"] = heatmap_filename
        session["severity_score"] = f"{severity_score:.2f}%"  # Save the calculated severity score

        return redirect(url_for("index"))

    return render_template("index.html", 
                           image=session.get("image"), 
                           prediction=session.get("prediction"), 
                           confidence=session.get("confidence"), 
                           heatmap=session.get("heatmap"),
                           severity_score=session.get("severity_score"))  # Pass severity score to template

if __name__ == "__main__":
    print("Starting Flask App...")
    app.run(debug=True)
