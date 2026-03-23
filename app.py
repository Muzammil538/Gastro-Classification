import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
import uuid
import json
from datetime import datetime
from model_pipeline import HybridModel

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
HISTORY_FILE = "history.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = HybridModel()

# Load history
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
else:
    history = []


def save_history():
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            filename = str(uuid.uuid4()) + ".jpg"
            path = os.path.join(UPLOAD_FOLDER, filename)

            file.save(path)

            img = cv2.imread(path)

            prediction, confidence = model.predict(img)

            image_path = path

            # Save history
            history.insert(0, {
                "image": path,
                "prediction": prediction,
                "confidence": round(confidence * 100, 2),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M")
            })

            # limit history
            if len(history) > 20:
                history.pop()

            save_history()

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           image_path=image_path)


@app.route("/history")
def view_history():
    return render_template("history.html", history=history)


if __name__ == "__main__":
    app.run(debug=True)