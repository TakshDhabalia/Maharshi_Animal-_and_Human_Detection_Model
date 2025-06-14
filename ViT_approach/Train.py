import os
import json
import wandb
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import ViTModel, ViTImageProcessor
from torch.optim import AdamW

# ====================
# 1. Initialize wandb
# ====================
wandb.init(project="ViT-Animal-Detection", config={
    "model": "vit-base-patch16-224",
    "epochs": 5,
    "batch_size": 16,
    "lr": 2e-5,
    "img_size": 224,
})

config = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================
# 2. Convert COCO annotations to binary
# =====================================
def parse_coco_labels(coco_json_path, animal_ids=set(range(1, 11))):
    with open(coco_json_path) as f:
        data = json.load(f)

    image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    labels = {filename: 0 for filename in image_id_to_filename.values()}

    for ann in data['annotations']:
        if ann['category_id'] in animal_ids:
            labels[image_id_to_filename[ann['image_id']]] = 1

    return labels

# =======================
# 3. Custom Dataset Class
# =======================
class AnimalPresenceDataset(Dataset):
    def __init__(self, image_dir, annotation_path, transform=None):
        self.image_dir = image_dir
        self.labels = parse_coco_labels(annotation_path)
        self.image_files = list(self.labels.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels[img_name], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label.unsqueeze(0)  # shape: [1]

# ========================
# 4. Custom ViT Classifier
# ========================
class ViTBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.classifier = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 1)
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        logits = self.classifier(cls_token)
        return logits

# ====================
# 5. Transforms
# ====================
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# ====================
# 6. Dataset + Dataloader
# ====================
train_dataset = AnimalPresenceDataset(
    "dst/animals-2/train", "dst/animals-2/train/_annotations.coco.json", transform)
val_dataset = AnimalPresenceDataset(
    "dst/animals-2/valid", "dst/animals-2/valid/_annotations.coco.json", transform)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

# ====================
# 7. Model, Loss, Optim
# ====================
model = ViTBinaryClassifier().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=config.lr)

# ====================
# 8. Training Loop
# ====================
from tqdm import tqdm

# ====================
# 8. Training Loop (Verbose)
# ====================
for epoch in range(config.epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    print(f"\nEpoch [{epoch + 1}/{config.epochs}]")

    train_bar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (imgs, labels) in enumerate(train_bar):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Update tqdm bar
        train_bar.set_postfix({
            "Batch Loss": loss.item(),
            "Running Acc": f"{(correct/total)*100:.2f}%"
        })

        # Optional: print every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"  [Batch {batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    train_acc = correct / total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc="Validation", leave=False)
        for imgs, labels in val_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            val_bar.set_postfix({
                "Batch Loss": loss.item()
            })

    val_acc = val_correct / val_total

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": total_loss / total,
        "train_acc": train_acc,
        "val_loss": val_loss / val_total,
        "val_acc": val_acc
    })

    print(f"✅ Epoch {epoch+1}: Train Loss={total_loss/total:.4f}, Train Acc={train_acc:.4f}, "
          f"Val Loss={val_loss/val_total:.4f}, Val Acc={val_acc:.4f}")

# ====================
# 9. Save model
# ====================
torch.save(model.state_dict(), "vit_animal_binary.pth")
wandb.save("vit_animal_binary.pth")
