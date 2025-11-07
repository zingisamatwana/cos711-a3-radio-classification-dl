import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations import (
    Compose, Normalize, HorizontalFlip, VerticalFlip, RandomRotate90,
    ShiftScaleRotate, GaussianBlur, GaussNoise, CoarseDropout, Resize, CenterCrop, Affine,
)
from albumentations.pytorch import ToTensorV2
from typing import List
from .config import IMG_SIZE, CROP_SIZE, MEAN, STD

def build_transforms(train: bool):
    if train:
        return Compose([
            Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_CUBIC),
            # ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=15, p=0.6, border_mode=cv2.BORDER_REFLECT101),
            Affine(scale=(0.95, 1.05), rotate=(-15, 15), translate_percent=(-0.03, 0.03), p=0.6),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomRotate90(p=0.2),
            GaussNoise(p=0.2),
            GaussianBlur(blur_limit=(3,5), p=0.15),  
            CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(1, IMG_SIZE // 12),
            hole_width_range=(1, IMG_SIZE // 12),
            p=0.3,),
            CenterCrop(CROP_SIZE, CROP_SIZE),
            Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ])
    else:
        return Compose([
            Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_CUBIC),
            CenterCrop(CROP_SIZE, CROP_SIZE),
            Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ])

class GalaxySingleHead(Dataset):
    """
    train/val csv must have: filepath,label
    """
    def __init__(self, csv_path: str, class_names: List[str], train: bool):
        self.df = pd.read_csv(csv_path)
        self.class_names = class_names
        self.cls2idx = {c:i for i,c in enumerate(class_names)}
        self.train = train
        self.tfms = build_transforms(train)

    def __len__(self): return len(self.df) 

    def _load_gray_as_rgb(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) # keeps channels as-is
        if img is None:
            raise FileNotFoundError(path)
        
        if img.ndim == 2: # grayscale -> 3ch
            img = np.repeat(img[..., None], 3, axis=2)
        elif img.ndim == 3:
            if img.shape[2] == 1: # single channel -> 3ch
                img = np.repeat(img, 3, axis=2)
            elif img.shape[2] == 4: # drop alpha
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_gray_as_rgb(row["filepath"])
        x = self.tfms(image=img)["image"]
        y = self.cls2idx[row["label"]]
        return x, torch.tensor(y, dtype=torch.long)
