import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Configure GPU device for training, fallback to CPU if CUDA is unavailable, display device name and VRAM size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Check if NVIDIA GPU (via CUDA) is supported.

print("=" * 60) 
print(f"DEVICE: {device}")
if torch.cuda.is_available():
    print(f"GPU NAME: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB") # in dung lượng VRAM của card đồ hoạ
print("=" * 60)



# Training hyperparameter configuration (fixed style)
"""
Determine the number of predicted labels and the model output directory, and check whether the directory exists.
determine batch size, learning rate


"""
BATCH_SIZE = 64 # Optimal power-of-two for GPU memory alignment & gradient stability
LR = 0.001 #Default Adam Learning rate
EPOCHS = 50 
NUM_CLASSES = 4 
CHECKPOINT_DIR = "checkpoints_pytorch" 
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class DAISEEDataset(Dataset): # Declare Dataset class for data storage and access
    def __init__(self, dataset_dir, label_csv, transform=None, max_frames=3):
        self.transform = transform
        self.samples = [] # (image_path, label_tensor)
        
        # Read CSV labels
        df = pd.read_csv(label_csv)
        # Create ClipID -> label dictionary for O(1) ultra-fast lookup
        # ClipIDs typically look like: '1100011002.avi' or '1100011002.mp4'
        label_dict = {}
        for _, row in df.iterrows():
            clip_id = str(row['ClipID']).strip()
            clip_base = os.path.splitext(clip_id)[0]
            # Labels: [Boredom, Engagement, Confusion, Frustration]
            boredom = 1.0 if row.get('Boredom', 0) >= 1 else 0.0
            engagement = 1.0 if row.get('Engagement', 0) >= 1 else 0.0
            confusion = 1.0 if row.get('Confusion', 0) >= 1 else 0.0
            frustration = 1.0 if (row.get('Frustration ', 0) if 'Frustration ' in row else row.get('Frustration', 0)) >= 1 else 0.0
            label_vec = [boredom, engagement, confusion, frustration]
            label_dict[clip_base] = label_vec

        # Scan directories for image frames
        humans = sorted(os.listdir(dataset_dir))
        for human in humans:
            h_dir = os.path.join(dataset_dir, human)
            if not os.path.isdir(h_dir):
                continue
            videos = sorted(os.listdir(h_dir))
            for vid in videos:
                v_dir = os.path.join(h_dir, vid)
                if not os.path.isdir(v_dir):
                    continue
                if vid in label_dict:
                    label = label_dict[vid]
                    jpgs = [os.path.join(v_dir, f) for f in os.listdir(v_dir) if f.lower().endswith(".jpg")]
                    if len(jpgs) > max_frames:
                        chosen = random.sample(jpgs, max_frames)
                    else:
                        chosen = jpgs
                    for p in chosen:
                        self.samples.append((p, label))

        print(f"Loaded {len(self.samples)} images from {dataset_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


# Define a simple custom CNN architecture

class EngagementCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(EngagementCNN, self).__init__()
        # Conv block 1: 224 -> 112 -> 56
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Conv block 2: 56 -> 28 -> 14
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Conv block 3: 14 -> 7 -> 3
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Fully connected: 512 * 3 * 3 = 4608
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        logits = self.fc(x)
        return logits


def compute_macro_f1(y_true, y_pred_prob, thresh=0.5):
    y_pred = (y_pred_prob > thresh).float()
    tp = (y_pred * y_true).sum(dim=0)
    fp = (y_pred * (1.0 - y_true)).sum(dim=0)
    fn = ((1.0 - y_pred) * y_true).sum(dim=0)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    return f1.mean().item()


def train():
    # Image transformation pipeline & Data Augmentation on GPU/CPU
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize Datasets
    print("\nĐang tải dữ liệu...")
    train_dataset = DAISEEDataset("dataset/DAiSEE/DataSet/Train", "dataset/DAiSEE/Labels/TrainLabels.csv", transform=train_transform, max_frames=3)
    val_dataset = DAISEEDataset("dataset/DAiSEE/DataSet/Validation", "dataset/DAiSEE/Labels/ValidationLabels.csv", transform=val_transform, max_frames=3)
    test_dataset = DAISEEDataset("dataset/DAiSEE/DataSet/Test", "dataset/DAiSEE/Labels/TestLabels.csv", transform=val_transform, max_frames=3)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)

    model = EngagementCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # LR Scheduler: Automatically reduce LR by 50% if Val F1 plateaus for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)

    best_val_f1 = 0.0

    print(f"\nStarting training on {device} ({EPOCHS} Epochs)...")
    for epoch in range(1, EPOCHS + 1):
        # 1. Training loop
        model.train()
        train_loss = 0.0
        train_preds, train_trues = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits)
            train_preds.append(probs.detach().cpu())
            train_trues.append(labels.detach().cpu())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_train_loss = train_loss / len(train_dataset)
        train_f1 = compute_macro_f1(torch.cat(train_trues), torch.cat(train_preds))

        # 2. Validation loop
        model.eval()
        val_loss = 0.0
        val_preds, val_trues = [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(logits)
                val_preds.append(probs.cpu())
                val_trues.append(labels.cpu())

        epoch_val_loss = val_loss / len(val_dataset)
        val_f1 = compute_macro_f1(torch.cat(val_trues), torch.cat(val_preds))

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n>>> [Epoch {epoch}/{EPOCHS}] LR: {current_lr:.6f} | Train Loss: {epoch_train_loss:.4f} | Train F1: {train_f1:.4f} | Val Loss: {epoch_val_loss:.4f} | Val F1: {val_f1:.4f}")

        # Update LR Scheduler based on Val F1
        scheduler.step(val_f1)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            print(f" 📉 [LR Scheduler] Val F1 plateaued after 3 epochs -> Reducing LR: {current_lr:.6f} -> {new_lr:.6f}")

        # Save checkpoint after each Epoch
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"Epoch_{epoch}_model.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # Save the best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f" ⭐ Saved Best Model with Val F1 = {val_f1:.4f}!")

    # Test evaluation after training completion
    print("\n--- EVALUATION ON TEST SET ---")
    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            test_preds.append(probs.cpu())
            test_trues.append(labels.cpu())

    test_f1 = compute_macro_f1(torch.cat(test_trues), torch.cat(test_preds))
    print(f"\n==========================================")
    print(f"FINAL RESULT ON TEST SET: Macro F1 = {test_f1:.4f}")
    print(f"==========================================")

if __name__ == '__main__':
    train()
