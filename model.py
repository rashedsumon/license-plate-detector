import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from data_loader import download_yolo_weights

class PlateDetector:
    def __init__(self):
        # 1. Automatically download/get path to weights
        self.weights_path = download_yolo_weights()
        # 2. Load the YOLO model with the weights
        self.model = YOLO(self.weights_path)

    def predict(self, pil_image):
        """
        Takes a PIL Image, runs it through the YOLO model,
        and returns the annotated image (with bounding boxes) and raw results.
        """
        # Convert PIL image to OpenCV BGR format for YOLO
        opencv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Run inference (prediction)
        results = self.model(opencv_img)
        
        # Draw bounding boxes onto the image
        # plot() returns a BGR numpy array
        annotated_bgr = results[0].plot()
        
        # Convert back to RGB PIL Image for Streamlit display
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        final_image = Image.fromarray(annotated_rgb)
        
        return final_image, results[0]