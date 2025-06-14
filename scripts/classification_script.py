import os
import json
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline

# === CONFIG ===
CROP_DIR = "intermediate/crops"
RESULTS_DIR = "intermediate/results"
ANNOTATED_DIR = os.path.join(RESULTS_DIR, "annotated_images")
RESULT_JSON = os.path.join(RESULTS_DIR, "classification_results.json")

os.makedirs(ANNOTATED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

classifier = pipeline("image-classification", model="dima806/animal_151_types_image_detection")

results = []

# === Parse all crops ===
for fname in sorted(os.listdir(CROP_DIR)):
    if not fname.endswith(".jpg"):
        continue

    img_path = os.path.join(CROP_DIR, fname)
    crop_img = Image.open(img_path).convert("RGB")

    preds = classifier(crop_img)
    top_pred = preds[0]

    # Draw label on the crop image
    draw = ImageDraw.Draw(crop_img)
    font = ImageFont.load_default()
    label = f"{top_pred['label']} ({top_pred['score']:.2f})"
    draw.text((5, 5), label, fill="red", font=font)

    # Save annotated image
    out_path = os.path.join(ANNOTATED_DIR, fname)
    crop_img.save(out_path)

    results.append({
        "file": fname,
        "label": top_pred["label"],
        "confidence": top_pred["score"]
    })

# === Save JSON ===
with open(RESULT_JSON, "w") as f:
    json.dump(results, f, indent=4)

print(f"[INFO] Saved classified crops in {ANNOTATED_DIR}")
print(f"[INFO] Results saved to {RESULT_JSON}")
