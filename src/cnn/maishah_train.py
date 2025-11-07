"""
Training Pipeline using Pre-prepared Datasets
"""

import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (f1_score, precision_score, recall_score, hamming_loss)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

class Config:
    LABELED_DATA = 'prepared_labeled_data.csv'
    UNLABELED_DATA = 'prepared_unlabelled_data.csv'  # not used in training
    OUTPUT_DIR = 'results/'
    PLOTS_DIR = 'results/plots/'
    METRICS_DIR = 'results/metrics/'

    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    IMG_SIZE = 160
    NUM_WORKERS = 0
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    CLASSES = ['FR I', 'FR II', 'Point Source', 'Bent', 'Exotic',
               'Should be discarded', 'X-Shaped', 'S/Z shaped']

config = Config()

# Create output directories
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.PLOTS_DIR, exist_ok=True)
os.makedirs(config.METRICS_DIR, exist_ok=True)

# ==================== TIMER ====================

class Timer:
    def __init__(self, name="Operation"):
        self.name = name
    def __enter__(self):
        self.start_time = time.time()
        return self
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        print(f"⏱️  {self.name}: {elapsed:.2f}s ({elapsed/60:.2f}m)")

# ==================== DATASET ====================

class RadioSourceDataset(Dataset):
    def __init__(self, df, mlb, transform=None):
        self.df = df.reset_index(drop=True)
        self.mlb = mlb
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['path']
        labels = eval(self.df.iloc[idx]['labels']) if isinstance(self.df.iloc[idx]['labels'], str) else self.df.iloc[idx]['labels']

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        label_encoded = self.mlb.transform([labels])[0]
        return image, torch.FloatTensor(label_encoded)

# ==================== TRANSFORMS ====================

def get_transforms(is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

# ==================== MODEL ====================

class RadioSourceClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ==================== TRAINING + VALIDATION ====================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > threshold).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    return total_loss / len(dataloader), np.vstack(all_preds), np.vstack(all_labels)

# ==================== MAIN ====================

def main():
    print("\n" + "="*70)
    print(" RADIO SOURCE TRAINING - USING PREPARED DATA")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {config.DEVICE}\n")

    # Load prepared data
    with Timer("Loading Prepared Data"):
        df = pd.read_csv(config.LABELED_DATA)
        print(f" Loaded {len(df)} labeled samples from prepared_labeled_data.csv")

    # Encode labels
    mlb = MultiLabelBinarizer(classes=config.CLASSES)
    mlb.fit([config.CLASSES])

    # Split data
    with Timer("Data Splitting"):
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Create datasets and loaders
    train_dataset = RadioSourceDataset(train_df, mlb, transform=get_transforms(True))
    val_dataset = RadioSourceDataset(val_df, mlb, transform=get_transforms(False))

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Initialize model
    model = RadioSourceClassifier(num_classes=len(config.CLASSES)).to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(config.EPOCHS):
        with Timer(f"Epoch {epoch+1}/{config.EPOCHS}"):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
            val_loss, preds, labels = validate(model, val_loader, criterion, config.DEVICE)
            scheduler.step(val_loss)

            f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
            f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | F1-micro: {f1_micro:.4f} | F1-macro: {f1_macro:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, 'best_model_prepared.pth'))
                print("   Saved best model")

    print("\n Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved as: {os.path.join(config.OUTPUT_DIR, 'best_model_prepared.pth')}")

if __name__ == "__main__":
    main()
