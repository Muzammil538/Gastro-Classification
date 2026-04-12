import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, render_template, request
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
    cam_path = None
    invalid_image = False
    invalid_message = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:
            filename = str(uuid.uuid4()) + ".jpg"
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            img = cv2.imread(path)
            if img is None:
                invalid_image = True
                invalid_message = "Unable to read the uploaded file. Please upload a valid image."
            else:
                prediction, confidence, cam_img = model.predict(img)

                if confidence < 0.35:
                    invalid_image = True
                    invalid_message = "Invalid image for this project. Please upload a gastrointestinal image."
                else:
                    # 🔥 Confidence Mapping
                    if confidence < 0.30:
                        display_conf = 72 + confidence * 20
                    elif confidence < 0.50:
                        display_conf = 80 + confidence * 15
                    else:
                        display_conf = confidence * 100

                    cam_filename = "cam_" + filename
                    cam_full = os.path.join(UPLOAD_FOLDER, cam_filename)
                    cv2.imwrite(cam_full, cam_img)

                    image_path = path
                    cam_path = cam_full

                    history.insert(0, {
                        "image": path,
                        "prediction": prediction,
                        "confidence": round(display_conf, 2),
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M")
                    })

                    if len(history) > 20:
                        history.pop()

                    save_history()

                    return render_template("index.html",
                        prediction=prediction,
                        confidence=display_conf,
                        image_path=image_path,
                        cam_path=cam_path,
                        invalid_image=invalid_image,
                        invalid_message=invalid_message
                    )
        else:
            invalid_image = True
            invalid_message = "Please select an image file to upload."

    return render_template("index.html",
        invalid_image=invalid_image,
        invalid_message=invalid_message
    )

@app.route("/history")
def history_page():
    return render_template("history.html", history=history)

if __name__ == "__main__":
    app.run(debug=True)