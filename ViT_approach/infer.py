import torch
from torch import nn
from torchvision import transforms
from transformers import ViTModel, ViTImageProcessor
from PIL import Image

# =======================
# 1. Define the model
# =======================
class ViTBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.classifier = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits


# =======================
# 2. Load model weights
# =======================
model = ViTBinaryClassifier()
model.load_state_dict(torch.load("D:/Internship_Tasks/Maharshi_Animal-_and_Human_Detection_Model/ViT_approach/models/vit_animal_binary.pth", map_location="cpu"))
model.eval()

# =======================
# 3. Prepare image
# =======================
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# Replace this with your test image path
image_path = r"D:\Internship_Tasks\Maharshi_Animal-_and_Human_Detection_Model\ViT_approach\rrc.jpg"
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

# =======================
# 4. Run inference
# =======================
with torch.no_grad():
    prob = model(image_tensor)
    prediction = int(prob > 0.5)


# =======================
# 5. Show result
# =======================
print(f"Prediction: {'Animal Present' if prediction else 'No Animal'} (Confidence: {prob.item():.4f})")


"""
ViT needs a lot more data so this is very bad of an attempt for this approach of ensembling
"""