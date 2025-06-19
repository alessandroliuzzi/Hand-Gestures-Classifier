import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models.video as models
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# === Config ===
CSV_PATH = '/kaggle/input/20bn-jester/Train.csv'  # adapt it according to your Train.csv directory
ROOT_DIR = '/kaggle/input/20bn-jester/Train'      # adapt it according to your Train folder directory
BEST_MODEL_PATH = '/kaggle/input/cv-latest-full/checkpoints/best_model.pth' #adapt it according to your best_model.pth directory

FRAMES_PER_CLIP = 16
BATCH_SIZE = 8
DATASET_PORTION = 0.4

# === Dataset ===
class VideoDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.root_dir = root_dir
        for _, row in df.iterrows():
            video_id = str(row['video_id'])
            label = row['label_id']
            video_dir = os.path.join(self.root_dir, video_id)
            if not os.path.isdir(video_dir):
                continue
            frame_files = sorted(os.listdir(video_dir), key=lambda x: int(x.split('.')[0]))
            if len(frame_files) >= 2 * FRAMES_PER_CLIP:
                clip_paths = [os.path.join(video_dir, frame_files[i]) for i in range(0, 2 * FRAMES_PER_CLIP, 2)]
                self.samples.append((clip_paths, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_paths, label = self.samples[idx]
        clip = []
        for path in clip_paths:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            clip.append(img)
        clip = torch.stack(clip).permute(1, 0, 2, 3)  # C, T, H, W
        return clip, label

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Prepare test split ===
df = pd.read_csv(CSV_PATH)

df_small, _ = train_test_split(
    df,
    test_size=1 - DATASET_PORTION,
    stratify=df["label_id"],
    random_state=42
)

_, val_test_df = train_test_split(
    df_small,
    test_size=0.2,
    stratify=df_small["label_id"],
    random_state=42
)
_, test_df = train_test_split(
    val_test_df,
    test_size=0.5,
    stratify=val_test_df["label_id"],
    random_state=42
)

# === Dataloader ===
test_dataset = VideoDataset(test_df, ROOT_DIR, transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Model ===
model = models.r3d_18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 27)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# === Evaluation ===
criterion = nn.CrossEntropyLoss()
test_loss = 0.0
correct, total = 0, 0

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"\n=== TEST RESULTS ===")
print(f"Test Loss: {test_loss / len(test_loader):.4f}")
print(f"Test Accuracy: {correct / total:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# === Optional: Show sample predictions ===
print("\nSample predictions:")
for i in range(10):
    print(f"Sample {i+1}: Predicted = {all_preds[i]}, Actual = {all_labels[i]}")
