import os
import json
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# Paths
data_dir = "ds"
img_dir = os.path.join(data_dir, "img")
ann_dir = os.path.join(data_dir, "ann")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Custom Dataset class


class HAM10000Dataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None, max_images=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        print("Loading and parsing annotations...")
        class_titles = set()

        for ann_file in tqdm(sorted(os.listdir(ann_dir))):
            if max_images is not None and len(self.samples) >= max_images:
                break
            ann_path = os.path.join(ann_dir, ann_file)
            with open(ann_path, 'r') as f:
                data = json.load(f)

            objects = data.get("objects", [])
            if not objects:
                continue
            label = objects[0].get("classTitle")
            if not label:
                continue

            image_filename = ann_file.replace(".json", "")
            image_path = os.path.join(img_dir, image_filename)
            if not os.path.exists(image_path):
                continue

            class_titles.add(label)
            self.samples.append((image_path, label))

        self.class_to_idx = {label: idx for idx,
                             label in enumerate(sorted(class_titles))}
        self.idx_to_class = {idx: label for label,
                             idx in self.class_to_idx.items()}
        print(
            f"Found {len(self.samples)} samples across {len(self.class_to_idx)} classes.")
        print("Class to Index Mapping:")
        for label, idx in self.class_to_idx.items():
            print(f"{idx} : {label}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_idx = self.class_to_idx[label]
        return image, label_idx


# Step 2: Model Definition
class SkinCancerClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        self.base_model = resnet50(weights=weights)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


# Step 3: Main Training Script
def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = HAM10000Dataset(
        img_dir, ann_dir, transform=transform, max_images=None)

    dataloader = DataLoader(dataset, batch_size=32,
                            shuffle=True, num_workers=6)

    model = SkinCancerClassifier(
        num_classes=len(dataset.class_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    os.makedirs("checkpoints", exist_ok=True)

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(dataset)
        epoch_acc = correct / total
        print(
            f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_to_idx': dataset.class_to_idx
        }, f"checkpoints/ham10000_epoch_{epoch+1}.pth")

    print("Training completed.")


if __name__ == "__main__":
    main()
