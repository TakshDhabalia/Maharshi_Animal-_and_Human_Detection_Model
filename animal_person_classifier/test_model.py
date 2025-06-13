import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import argparse
import time
from ultralytics import YOLO
from collections import defaultdict

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['Cat', 'Dog', 'Elephant', 'Lion', 'Person']
MODEL_PATH = 'animal_person_classifier.h5'
YOLO_CONFIDENCE = 0.5

class YOLODetector:
    def __init__(self, model_name='yolov8x.pt'):  # Using larger model for better accuracy
        """Initialize YOLO detector with a pretrained model."""
        self.model = YOLO(model_name)
        self.class_names = self.model.names
        # Map YOLO class names to our class names
        self.class_mapping = {
            'person': 'Person',
            'cat': 'Cat',
            'dog': 'Dog',
            'elephant': 'Elephant',
            'bear': 'Lion',  # Map bear to Lion as they have similar features
            'zebra': 'Elephant',  # Map zebra to Elephant as they're both large animals
            'giraffe': 'Elephant'  # Map giraffe to Elephant
        }
    
    def detect_objects(self, image):
        """Detect objects in the image using YOLO with enhanced detection."""
        results = self.model(image, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = self.class_names[cls_id]
            
            if conf >= YOLO_CONFIDENCE:
                mapped_class = self.class_mapping.get(class_name, class_name)
                
                detections.append({
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(conf),
                    'class_id': cls_id,
                    'class_name': class_name,
                    'mapped_class': mapped_class
                })
        
        return detections

def get_best_detection(detections, image_shape, target_classes=None):
    if target_classes is None:
        target_classes = ['person', 'cat', 'dog', 'elephant', 'bear', 'zebra', 'giraffe']
    
    # Priority order for our classes
    priority_order = {
        'Person': 0,
        'Lion': 1,
        'Elephant': 2,
        'Dog': 3,
        'Cat': 4
    }
    
    valid_detections = []
    for det in detections:
        if det['class_name'] in target_classes:
            if det['mapped_class'] in priority_order:
                valid_detections.append(det)
    
    if not valid_detections:
        # If no valid detections, return center crop
        height, width = image_shape[:2]
        size = min(height, width) // 2
        x = (width - size) // 2
        y = (height - size) // 2
        return (x, y, x + size, y + size), None
    
    # Sort detections by our priority order and then by confidence
    valid_detections.sort(
        key=lambda x: (priority_order.get(x['mapped_class'], float('inf')), -x['confidence'])
    )
    
    best_det = valid_detections[0]
    return best_det['box'], best_det['mapped_class']

def preprocess_with_yolo(image, yolo_detector, target_size=IMAGE_SIZE):
    """Preprocess image using YOLO for object detection with enhanced logic."""
    detections = yolo_detector.detect_objects(image)
    box, predicted_class = get_best_detection(detections, image.shape)
    
    if predicted_class:
        class_confidence = 0.9 
    else:
        class_confidence = 0.5  
    
    x1, y1, x2, y2 = box
    padding_x = int((x2 - x1) * 0.25)
    padding_y = int((y2 - y1) * 0.25)
    
    # Apply padding with image boundaries check
    h, w = image.shape[:2]
    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(w, x2 + padding_x)
    y2 = min(h, y2 + padding_y)
    
    # Crop and resize
    cropped = image[int(y1):int(y2), int(x1):int(x2)]
    if cropped.size == 0:
        size = min(h, w) // 2
        x = (w - size) // 2
        y = (h - size) // 2
        cropped = image[y:y+size, x:x+size]
    
    resized = cv2.resize(cropped, target_size)
    processed = resized.astype(np.float32) / 255.0
    
    return processed, box, predicted_class, class_confidence

def test_single_image(image_path, model, yolo_detector):
    """Test the model on a single image with enhanced YOLO detection."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preprocess with YOLO
    start_time = time.time()
    processed_img, box, yolo_class, yolo_confidence = preprocess_with_yolo(img_rgb, yolo_detector)
    preprocess_time = time.time() - start_time
    
    img_array = np.expand_dims(processed_img, axis=0)
    start_time = time.time()
    predictions = model.predict(img_array, verbose=0)[0]
    inference_time = time.time() - start_time
    
    if yolo_confidence > 0.8 and yolo_class in CLASS_NAMES:
        class_idx = CLASS_NAMES.index(yolo_class)
        predictions[class_idx] = max(predictions[class_idx], yolo_confidence)
        predictions = predictions / np.sum(predictions)
    
    predicted_class_idx = np.argmax(predictions)
    confidence = np.max(predictions)
    
    # Draw bounding box on original image
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    if yolo_class and yolo_class in CLASS_NAMES:
        label = f"Model: {CLASS_NAMES[predicted_class_idx]}"
    else:
        label = f"Model: {CLASS_NAMES[predicted_class_idx]}"
    
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Detection & Classification')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(CLASS_NAMES))
    plt.barh(y_pos, predictions, align='center')
    plt.yticks(y_pos, CLASS_NAMES)
    plt.xlabel('Probability')
    plt.title('Class Probabilities')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Preprocessing time: {preprocess_time*1000:.1f}ms")
    print(f"Inference time: {inference_time*1000:.1f}ms")
    print(f"Predicted: {CLASS_NAMES[predicted_class_idx]} with confidence {confidence:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced animal and person classifier with YOLO detection')
    parser.add_argument('--image', type=str, required=True, help='Path to an image file for testing')
    parser.add_argument('--yolo-model', type=str, default='yolov8x.pt', 
                       help='Path to YOLO model file (default: yolov8x.pt)')
    args = parser.parse_args()
    
    print("Loading models...")
    try:
        model = load_model(MODEL_PATH)
        print(f"Classification model loaded from {MODEL_PATH}")
        
        yolo_detector = YOLODetector(args.yolo_model)
        print(f"YOLO detector initialized with {args.yolo_model}")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    test_single_image(args.image, model, yolo_detector)

if __name__ == "__main__":
    main()
