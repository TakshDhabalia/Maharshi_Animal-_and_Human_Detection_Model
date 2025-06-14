import os
import cv2
import time
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from transformers import pipeline

# ============================
# Section 1: Configuration
# ============================
VIDEO_INPUT_DIR = r'D:\Internship_Tasks\Maharshi_Animal-_and_Human_Detection_Model\Maharshi_Animal-_and_Human_Detection_Model\test_videos'
INTERMEDIATE_OUTPUT_DIR = './intermediate/'
ANNOTATED_FRAMES_DIR = os.path.join(INTERMEDIATE_OUTPUT_DIR, 'annotated_frames')
CROPS_DIR = os.path.join(INTERMEDIATE_OUTPUT_DIR, 'crops')
CLASSIFIED_CROPS_DIR = os.path.join(INTERMEDIATE_OUTPUT_DIR, 'classified_crops')
RESULT_JSON = os.path.join(INTERMEDIATE_OUTPUT_DIR, 'classification_results.json')

os.makedirs(VIDEO_INPUT_DIR, exist_ok=True)
os.makedirs(ANNOTATED_FRAMES_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(CLASSIFIED_CROPS_DIR, exist_ok=True)

# Load Models
print("[INFO] Loading YOLOv8 model...")
detector = YOLO('yolov8s.pt')
print("[INFO] YOLOv8 model loaded.")

print("[INFO] Loading HuggingFace classification pipeline...")
classifier = pipeline("image-classification", model="dima806/animal_151_types_image_detection")
print("[INFO] Classification pipeline ready.")

# ============================
# Section 2: Frame Processing
# ============================
def process_video(video_path, interval=1):
    print(f"\n[PROCESS] Starting processing on video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps)
    video_name = Path(video_path).stem
    frame_results = []

    for second in range(0, duration, interval):
        print(f"\n[TIME] Extracting frame at {second}s...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame, skipping...")
            continue

        print("[INFO] Running YOLOv8 detection...")
        results = detector(frame)[0]
        boxes = results.boxes
        print(f"[INFO] Detected {len(boxes)} objects at {second}s")

        annotated_frame = frame.copy()

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            print(f"  [DETECT] Object {i}: Class={detector.names[cls_id]} Confidence={conf:.2f}")
            crop = frame[y1:y2, x1:x2]
            crop_fname = f"{video_name}_sec{second}_det{i}_{detector.names[cls_id]}.jpg"
            crop_path = os.path.join(CROPS_DIR, crop_fname)
            cv2.imwrite(crop_path, crop)
            print(f"    [SAVE] Crop saved: {crop_path}")

            # Classification
            try:
                img_pil = Image.open(crop_path).convert("RGB")
                preds = classifier(img_pil)
                top_pred = preds[0]
                label = f"{top_pred['label']} ({top_pred['score']:.2f})"
                print(f"    [CLASSIFY] {label}")
            except Exception as e:
                print(f"    [ERROR] Classification failed: {e}")
                label = "Unknown"
                top_pred = {"label": "unknown", "score": 0.0}

            # Draw box + label on frame
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Annotate and save classified crop
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.load_default()
            draw.text((5, 5), label, fill="red", font=font)
            classified_crop_path = os.path.join(CLASSIFIED_CROPS_DIR, crop_fname)
            img_pil.save(classified_crop_path)
            print(f"    [SAVE] Classified crop saved: {classified_crop_path}")

            frame_results.append({
                "video": video_name,
                "frame_time": second,
                "box_id": i,
                "bbox": [x1, y1, x2, y2],
                "label": top_pred["label"],
                "confidence": top_pred["score"],
                "crop_file": crop_fname
            })

        annotated_name = f"{video_name}_sec{second}_annotated.jpg"
        annotated_path = os.path.join(ANNOTATED_FRAMES_DIR, annotated_name)
        cv2.imwrite(annotated_path, annotated_frame)
        print(f"[SAVE] Annotated frame saved: {annotated_path}")

    cap.release()
    print(f"[DONE] Completed processing of video: {video_path}")
    return frame_results

# ============================
# Section 3: Monitor Directory
# ============================
def monitor_directory():
    print("[INFO] Monitoring directory for new videos... Press Ctrl+C to stop.")
    processed_videos = set()
    all_results = []

    while True:
        video_files = [f for f in os.listdir(VIDEO_INPUT_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
        for video in video_files:
            if video not in processed_videos:
                video_path = os.path.join(VIDEO_INPUT_DIR, video)
                print(f"[NEW] New video detected: {video}")
                results = process_video(video_path)
                all_results.extend(results)

                with open(RESULT_JSON, 'w') as f:
                    json.dump(all_results, f, indent=4)
                    print(f"[SAVE] Results written to {RESULT_JSON}")

                processed_videos.add(video)
        time.sleep(5)

# ============================
# Section 4: Run Script
# ============================
if __name__ == '__main__':
    monitor_directory()
