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
CSV_PATH = '/kaggle/input/20bn-jester/Train.csv'  #to adapt according to the environment
ROOT_DIR = '/kaggle/input/20bn-jester/Train'      #to adapt according to the environment
CHECKPOINT_DIR = './checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

FRAMES_PER_CLIP = 16
BATCH_SIZE = 8
EPOCHS = 10 

DATASET_PORTION = 0.4  # 40% of Kaggle version of Jester dataset

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
            frame_files = sorted(os.listdir(video_dir), key=lambda x: int(x.split('.')[0]))
            # Sampling every other frame
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

# === Load DataFrame and sample with stratification ===
df = pd.read_csv(CSV_PATH)

# Keep 1/5 of total dataset
df_small, _ = train_test_split(
    df,
    test_size=1 - DATASET_PORTION,
    stratify=df["label_id"],
    random_state=42
)

# Split 80-10-10 on that 20%
train_df, val_test_df = train_test_split(
    df_small,
    test_size=0.2,
    stratify=df_small["label_id"],
    random_state=42
)
val_df, test_df = train_test_split(
    val_test_df,
    test_size=0.5,
    stratify=val_test_df["label_id"],
    random_state=42
)

# === Create Dataset ===
train_dataset = VideoDataset(train_df, ROOT_DIR, transform)
val_dataset = VideoDataset(val_df, ROOT_DIR, transform)
test_dataset = VideoDataset(test_df, ROOT_DIR, transform)

# === Dataloader ===
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Model ===
model = models.r3d_18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 27)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === Resume from checkpoint, if existing ===
start_epoch = 0
checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint.pth')
if os.path.isfile(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch']+1
    print(f"Resuming from epoch {start_epoch+1}")

# === Training Loop ===
best_val_loss = float('inf')

for epoch in range(start_epoch, EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f"[Epoch {epoch+1}/{EPOCHS}] Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        # Checkpoint 
        
    torch.save({
                'epoch': epoch,
                'batch': i + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }, checkpoint_path)

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed - Average Training Loss: {epoch_loss:.4f}")

    # === Validation ===
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = correct / total
    print(f"Validation Loss: {val_loss:.4f} - Accuracy: {val_accuracy:.4f}")

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
        print(f"Best model saved at epoch {epoch+1}")

# === Testing ===
model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth')))
model.eval()
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

print(f"Test Loss: {test_loss / len(test_loader):.4f} - Accuracy: {correct / total:.4f}")

# === Classification Report and Confusion Matrix ===
print("Classification Report:")
print(classification_report(all_labels, all_preds))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
