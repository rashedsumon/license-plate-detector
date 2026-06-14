import cv2
import numpy as np
from PIL import Image
from data_loader import download_yolo_weights

class PlateDetector:
    def __init__(self):
        # 1. Fetch downloaded Darknet paths
        self.weights_path, self.cfg_path = download_yolo_weights()
        
        # 2. Load the network using OpenCV's DNN module
        self.net = cv2.dnn.readNetFromDarknet(self.cfg_path, self.weights_path)
        
        # Enable CPU optimizations for cloud serving environment
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Get the output layer names needed to pull predictions
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def predict(self, pil_image):
        """
        Processes an image, matches shapes, finds bounding boxes via OpenCV DNN,
        and returns the annotated image and dummy raw results list.
        """
        # Convert PIL to BGR matrix
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        height, width, _ = frame.shape
        
        # Prepare the image format for YOLOv3 (resize to 416x416, scale pixels)
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Run forward pass
        layer_outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        
        # Parse through raw outputs
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter for high confidence items
                if confidence > 0.4:
                    # Scale bounding box back to image sizing dimensions
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    
        # Apply Non-Maximum Suppression to clear overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.4, nms_threshold=0.3)
        
        # Class tracking mock to avoid changing your UI layout
        class MockBox:
            def __init__(self, conf, xyxy):
                self.conf = [conf]
                self.xyxy = [xyxy]
                
        class MockResults:
            def __init__(self, boxes_list):
                self.boxes = boxes_list

        detected_boxes_ui = []

        # Draw the target boxes on image array
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                conf = confidences[i]
                
                # Draw standard rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                # Label positioning
                label = f"Plate: {conf:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Build mock structure expected by downstream UI
                detected_boxes_ui.append(MockBox(conf, [x, y, x+w, y+h]))

        # Format output mapping back to standard RGB PIL object
        annotated_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated_rgb), MockResults(detected_boxes_ui)