import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from data_loader import download_yolo_weights

class PlateDetector:
    def __init__(self):
        # 1. Get the downloaded Darknet file paths 
        self.weights_path, self.cfg_path = download_yolo_weights()
        
        # 2. Instantiate YOLO using the config architecture, then load the weights
        # This converts legacy formats cleanly for evaluation
        self.model = YOLO(self.cfg_path)  
        self.model.load(self.weights_path)

    def predict(self, pil_image):
        """
        Takes a PIL Image, runs it through the model,
        and returns the annotated image and raw results.
        """
        opencv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Run inference
        results = self.model(opencv_img)
        
        # Draw bounding boxes onto the image
        annotated_bgr = results[0].plot()
        
        # Convert back to PIL Image
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        final_image = Image.fromarray(annotated_rgb)
        
        return final_image, results[0]