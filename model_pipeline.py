import numpy as np
import cv2
import joblib
import torch
import timm
import tensorflow as tf

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
        img_v = img_v.astype(np.float32)/255.0
        img_v = np.transpose(img_v, (2,0,1))
        img_v = torch.tensor(img_v).unsqueeze(0)

        with torch.no_grad():
            f3 = self.vit.forward_features(img_v)
            f3 = f3.mean(dim=1).numpy()

        fused = np.concatenate([f1, f2, f3], axis=1)
        return fused.flatten()

    def generate_gradcam(self, img):

        img_resized = cv2.resize(img, (224,224))
        img_input = res_pre(img_resized)
        img_input = np.expand_dims(img_input, axis=0)

        grad_model = tf.keras.models.Model(
            [self.resnet.inputs],
            [self.resnet.get_layer("conv5_block3_out").output, self.resnet.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_input)
            loss = tf.reduce_max(predictions)

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)

        return heatmap.numpy()

    def overlay_heatmap(self, heatmap, img):

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    def predict(self, img):

        features = self.extract_features(img)
        features = self.scaler.transform([features])
        features = self.pca.transform(features)

        probs = self.model.predict_proba(features)[0]
        confidence = float(np.max(probs))
        pred_class = int(np.argmax(probs))

        heatmap = self.generate_gradcam(img)
        cam_img = self.overlay_heatmap(heatmap, img)

        return self.labels[pred_class], confidence, cam_img