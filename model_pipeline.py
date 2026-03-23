import numpy as np
import cv2
import joblib
import torch
import timm

from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre

class HybridModel:

    def __init__(self):
        self.resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.efficient = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.eval()

        self.model = joblib.load("models/final_model.pkl")
        self.labels = joblib.load("models/labels.pkl")
        self.scaler = joblib.load("models/scaler.pkl")
        self.pca = joblib.load("models/pca.pkl")

    def extract_features(self, img):

        img_r = cv2.resize(img, (224,224))
        img_e = cv2.resize(img, (224,224))

        r = res_pre(img_r)
        e = eff_pre(img_e)

        r = np.expand_dims(r, axis=0)
        e = np.expand_dims(e, axis=0)

        f1 = self.resnet.predict(r, verbose=0)
        f2 = self.efficient.predict(e, verbose=0)

        img_v = cv2.resize(img, (224,224))
        img_v = img_v.astype(np.float32) / 255.0
        img_v = np.transpose(img_v, (2,0,1))
        img_v = torch.tensor(img_v).unsqueeze(0)

        with torch.no_grad():
            f3 = self.vit.forward_features(img_v)
            f3 = f3.mean(dim=1).numpy()

        fused = np.concatenate([f1, f2, f3], axis=1)
        return fused.flatten()

    def predict(self, img):

        features = self.extract_features(img)

        features = self.scaler.transform([features])
        features = self.pca.transform(features)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
            confidence = float(np.max(probs))
            pred_class = int(np.argmax(probs))
        else:
            pred_class = self.model.predict(features)[0]
            confidence = 1.0

        if confidence < 0.25:
            return "Invalid / Not Endoscopy Image", confidence
        elif confidence < 0.40:
            return f"Low Confidence: {self.labels[pred_class]}", confidence
        else:
            return self.labels[pred_class], confidence