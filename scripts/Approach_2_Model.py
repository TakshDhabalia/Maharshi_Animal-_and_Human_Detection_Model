import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import argparse
import time
from ultralytics import YOLO
import uuid
from pathlib import Path

# Constants
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['Cat', 'Dog', 'Elephant', 'Lion', 'Person']
MODEL_PATH = 'animal_person_classifier.h5'
YOLO_CONFIDENCE = 0.5
INTERMEDIATE_DIR = Path("intermediate/cropped")
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)


class YOLODetector:
    def __init__(self, model_name='yolov8x.pt'):
        print(f"[INFO] Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        self.class_names = self.model.names
        self.class_mapping = {
            'person': 'Person',
            'cat': 'Cat',
            'dog': 'Dog',
            'elephant': 'Elephant',
            'bear': 'Lion',
            'zebra': 'Elephant',
            'giraffe': 'Elephant'
        }
        print("[INFO] YOLO model loaded.")

    def detect_objects(self, image):
        print("[INFO] Running YOLO object detection...")
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
                print(f"[DETECTION] {class_name} ({mapped_class}) - Conf: {conf:.2f}, Box: ({x1}, {y1}, {x2}, {y2})")
        return detections


def get_best_detection(detections, image_shape):
    priority_order = {
        'Person': 0,
        'Lion': 1,
        'Elephant': 2,
        'Dog': 3,
        'Cat': 4
    }
    valid_detections = [d for d in detections if d['mapped_class'] in priority_order]
    if not valid_detections:
        print("[WARN] No valid detections found. Falling back to center crop.")
        h, w = image_shape[:2]
        size = min(h, w) // 2
        x = (w - size) // 2
        y = (h - size) // 2
        return (x, y, x + size, y + size), None

    valid_detections.sort(key=lambda d: (priority_order[d['mapped_class']], -d['confidence']))
    print(f"[INFO] Selected detection for classification: {valid_detections[0]['mapped_class']}")
    return valid_detections[0]['box'], valid_detections[0]['mapped_class']


def preprocess_with_yolo(image, yolo_detector, target_size=IMAGE_SIZE, save_prefix="frame"):
    detections = yolo_detector.detect_objects(image)
    box, predicted_class = get_best_detection(detections, image.shape)

    x1, y1, x2, y2 = box
    padding_x = int((x2 - x1) * 0.25)
    padding_y = int((y2 - y1) * 0.25)
    h, w = image.shape[:2]
    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(w, x2 + padding_x)
    y2 = min(h, y2 + padding_y)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        print("[WARN] Cropped region is empty. Using fallback center crop.")
        size = min(h, w) // 2
        x = (w - size) // 2
        y = (h - size) // 2
        cropped = image[y:y+size, x:x+size]

    filename = f"{save_prefix}_{uuid.uuid4().hex[:8]}.jpg"
    save_path = INTERMEDIATE_DIR / filename
    cv2.imwrite(str(save_path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    print(f"[SAVED] Cropped image saved to {save_path}")

    resized = cv2.resize(cropped, target_size)
    processed = resized.astype(np.float32) / 255.0
    return processed, box, predicted_class


def classify_image(img_rgb, model, yolo_detector, save_prefix="frame"):
    print("[INFO] Preprocessing image for classification...")
    processed_img, box, yolo_class = preprocess_with_yolo(img_rgb, yolo_detector, save_prefix=save_prefix)
    img_array = np.expand_dims(processed_img, axis=0)
    print("[INFO] Running classification...")
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class_idx = np.argmax(predictions)
    confidence = np.max(predictions)
    print(f"[RESULT] Prediction: {CLASS_NAMES[predicted_class_idx]}, Confidence: {confidence:.2f}  \n\n")
    return box, CLASS_NAMES[predicted_class_idx], confidence, predictions


def process_image(image_path, model, yolo_detector):
    print(f"[INFO] Reading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    box, predicted_class, confidence, predictions = classify_image(
        img_rgb, model, yolo_detector, save_prefix="image"
    )
    x1, y1, x2, y2 = box

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{predicted_class}: {confidence:.2f}"
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    plt.figure(figsize=(10, 5))
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


def process_video(video_path, model, yolo_detector):
    print(f"[INFO] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps
    print(f"[INFO] Video FPS: {fps}, Processing every {frame_interval} frames.")

    frame_count = 0
    processed = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break
        if frame_count % frame_interval == 0:
            print(f"[FRAME] Processing frame #{frame_count}")
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            box, predicted_class, confidence, _ = classify_image(
                img_rgb, model, yolo_detector, save_prefix=f"frame{frame_count}"
            )
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{predicted_class}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            processed += 1
        frame_count += 1

    cap.release()
    print(f"[DONE] Processed {processed} frames.")


def main():
    parser = argparse.ArgumentParser(description='Classify image or video using YOLO + custom classifier.')
    parser.add_argument('--image', type=str, help='Path to image file.')
    parser.add_argument('--video', type=str, help='Path to video file.')
    parser.add_argument('--yolo-model', type=str, default='yolov8x.pt', help='Path to YOLO model.')
    args = parser.parse_args()

    print("[START] Loading classification model... \n")
    model = load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully. \n")

    print("[START] Initializing YOLO detector... \n")
    yolo_detector = YOLODetector(args.yolo_model)

    if args.image:
        print(f"[RUN] Processing single image: {args.image}")
        process_image(args.image, model, yolo_detector)
    elif args.video:
        print(f"[RUN] Processing video: {args.video}")
        process_video(args.video, model, yolo_detector)
    else:
        print("[ERROR] Please provide either --image or --video as input. \n")


if __name__ == "__main__":
    main()
