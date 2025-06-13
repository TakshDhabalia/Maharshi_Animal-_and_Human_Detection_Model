# animal_human_detection.py

"""
This script handles the object detection part of the Animal and Human Detection Task.
It continuously monitors the ./test_videos/ directory for new video files, processes them using a YOLOv8 model (from Ultralytics), 
and outputs intermediate results (cropped detected object images and bounding boxes) into a temporary directory for use by the classification model.

# ============================
# Section 1: Import Libraries
# ============================
"""
import os
import cv2
import time
import torch
from pathlib import Path
from ultralytics import YOLO

# ============================
# Section 2: Configuration
# ============================
# Set paths
VIDEO_INPUT_DIR = 'Maharshi_Animal-_and_Human_Detection_Model/test_videos/'
INTERMEDIATE_OUTPUT_DIR = 'Maharshi_Animal-_and_Human_Detection_Model/intermediate/'

# Create necessary directories if not exist
os.makedirs(VIDEO_INPUT_DIR, exist_ok=True)
os.makedirs(INTERMEDIATE_OUTPUT_DIR, exist_ok=True)

# Load the pretrained YOLOv8 detection model from Ultralytics (e.g., YOLOv8s)
model = YOLO('Maharshi_Animal-_and_Human_Detection_Model/yolov8s.pt')

# ============================
# Section 3: Utility Functions
# ============================
def extract_frames_and_detect(video_path, output_dir):
    """
    Extracts 1 frame per second from a video and runs YOLOv8 detection model on each frame.
    Detected regions are cropped and saved for classification.
    Also saves metadata (frame number, bounding boxes).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    interval = int(fps)  # Skip to 1 frame per second

    video_name = Path(video_path).stem
    frame_idx = 0
    saved_frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            results = model(frame)[0]
            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Crop and save detected object
                crop = frame[y1:y2, x1:x2]
                crop_filename = f"{video_name}_frame{saved_frame_idx}_box{i}.jpg"
                crop_path = os.path.join(output_dir, crop_filename)
                cv2.imwrite(crop_path, crop)

                # Save metadata
                with open(os.path.join(output_dir, f"{video_name}_metadata.txt"), 'a') as f:
                    f.write(f"{crop_filename},{frame_idx},{x1},{y1},{x2},{y2},{cls_id},{conf:.2f}\n")

            saved_frame_idx += 1

        frame_idx += 1

    cap.release()

# ============================
# Section 4: Directory Monitoring Loop
# ============================
def monitor_directory():
    """
    Continuously monitor the test_videos directory for new videos.
    Once a video is detected, run detection and save intermediate output.
    """
    print("[INFO] Monitoring directory for new videos...")
    processed_videos = set()

    while True:
        video_files = [f for f in os.listdir(VIDEO_INPUT_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
        for video in video_files:
            video_path = os.path.join(VIDEO_INPUT_DIR, video)
            if video not in processed_videos:
                print(f"[INFO] Processing video: {video}")
                extract_frames_and_detect(video_path, INTERMEDIATE_OUTPUT_DIR)
                print(f"[INFO] Finished processing: {video}")
                processed_videos.add(video)
        time.sleep(5)  # Polling interval

# ============================
# Section 5: Main Entrypoint
# ============================
if __name__ == '__main__':
    monitor_directory()

"""
What this script does:
- Monitors `./test_videos/` for new videos.
- For each video, it:
  - Extracts frames.
  - Runs YOLOv8 on each frame.
  - Crops each detected object and saves it to `./intermediate/`.
  - Logs bounding box and detection metadata for later use.

Next Steps:
- The classification model will use these crops from the `intermediate/` directory to classify each object as human or animal.
"""
