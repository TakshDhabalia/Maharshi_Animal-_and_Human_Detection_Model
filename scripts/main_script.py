# animal_human_detection.py

"""
This script handles the object detection part of the Animal and Human Detection Task.
It continuously monitors the ./test_videos/ directory for new video files, processes them using a YOLOv8 model (from Ultralytics),
and outputs intermediate results (cropped detected object images and bounding boxes) into a temporary directory for use by the classification model.
"""

# ============================
# Section 1: Import Libraries
# ============================
import os
import cv2
import time
from pathlib import Path
from ultralytics import YOLO

# ============================
# Section 2: Configuration
# ============================
VIDEO_INPUT_DIR = 'Maharshi_Animal-_and_Human_Detection_Model/test_videos/'
INTERMEDIATE_OUTPUT_DIR = 'Maharshi_Animal-_and_Human_Detection_Model/intermediate/'
ANNOTATED_FRAMES_DIR = os.path.join(INTERMEDIATE_OUTPUT_DIR, 'annotated_frames')
CROPS_DIR = os.path.join(INTERMEDIATE_OUTPUT_DIR, 'crops')

os.makedirs(VIDEO_INPUT_DIR, exist_ok=True)
os.makedirs(ANNOTATED_FRAMES_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)

# Load the YOLOv8 detection model
model = YOLO('Maharshi_Animal-_and_Human_Detection_Model/yolov8s.pt')

# ============================
# Section 3: Frame Extraction and Detection
# ============================
def extract_frames_and_detect(video_path, output_dir, interval=1):
    """
    Extracts one frame every `interval` seconds from the video,
    runs detection, saves annotated frames and cropped detections.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps)

    video_name = Path(video_path).stem

    for second in range(0, duration, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
        ret, frame = cap.read()
        if not ret:
            continue

        # Run detection
        results = model(frame)[0]
        annotated_frame = frame.copy()

        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw box + label
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[cls_id]}: {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Crop and save detected region
            crop = frame[y1:y2, x1:x2]
            crop_filename = f"{video_name}_sec{second}_det{i}_{model.names[cls_id]}.jpg"
            cv2.imwrite(os.path.join(CROPS_DIR, crop_filename), crop)

        # Save annotated frame
        annotated_name = f"{video_name}_sec{second}_annotated.jpg"
        cv2.imwrite(os.path.join(ANNOTATED_FRAMES_DIR, annotated_name), annotated_frame)

    cap.release()

# ============================
# Section 4: Monitor Folder
# ============================
def monitor_directory():
    """
    Monitors VIDEO_INPUT_DIR for new videos and processes them when found.
    """
    print("[INFO] Monitoring directory for new videos...")
    processed_videos = set()

    while True:
        video_files = [f for f in os.listdir(VIDEO_INPUT_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
        for video in video_files:
            if video not in processed_videos:
                video_path = os.path.join(VIDEO_INPUT_DIR, video)
                print(f"[INFO] Processing video: {video}")
                extract_frames_and_detect(video_path, INTERMEDIATE_OUTPUT_DIR)
                print(f"[INFO] Finished processing: {video}")
                processed_videos.add(video)
        time.sleep(5)

# ============================
# Section 5: Run the Program
# ============================
if __name__ == '__main__':
    monitor_directory()
