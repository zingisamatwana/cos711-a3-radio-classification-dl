"""
PREDICTION GENERATOR FOR RADIO SOURCE CLASSIFICATION
Creates test_labels.csv and generated_labels.csv
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================

class Config:
    MODEL_PATH = 'results/best_model_prepared.pth'
    IMG_SIZE = 160
    TYP_DIR = 'typ_PNG/'
    EXO_DIR = 'exo_PNG/'
    UNL_DIR = 'unl_PNG/'
    TEST_FILE = 'test.csv'
    BATCH_SIZE = 64
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CONFIDENCE_THRESHOLD = 0.5
    CLASSES = ['FR I', 'FR II', 'Point Source', 'Bent', 'Exotic',
               'Should be discarded', 'X-Shaped', 'S/Z shaped']
    COORD_THRESHOLD = 0.1

config = Config()

print("=" * 70)
print("RADIO SOURCE PREDICTION GENERATOR")
print("=" * 70)
print(f"Device: {config.DEVICE}")
print(f"Model: {config.MODEL_PATH}")
print(f"Image size: {config.IMG_SIZE}x{config.IMG_SIZE}")
print(f"Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
print("=" * 70)

# ==================== MODEL ====================

class RadioSourceClassifier(nn.Module):
    def __init__(self, num_classes):
        super(RadioSourceClassifier, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
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

# ==================== UTILITIES ====================

def parse_filename_coordinates(filename):
    """Parse RA and DEC from filename"""
    try:
        coord_part = filename.split('_')[0]
        coords = coord_part.split()
        if len(coords) == 2:
            return float(coords[0]), float(coords[1])
    except (ValueError, IndexError):
        pass
    return None, None

def find_closest_image(target_ra, target_dec, image_dict, threshold=0.1):
    """Find closest image to target coordinates"""
    if not image_dict:
        return None
    
    target_coords = np.array([[target_ra, target_dec]])
    coords = np.array(list(image_dict.keys()))
    files = list(image_dict.values())
    
    distances = cdist(target_coords, coords, metric='euclidean')[0]
    min_idx = np.argmin(distances)
    
    if distances[min_idx] < threshold:
        return files[min_idx]
    return None

# ==================== DATASET ====================

class PredictionDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (config.IMG_SIZE, config.IMG_SIZE), color='black')
        
        if self.transform:
            image = self.transform(image)
        return image

# ==================== MODEL LOADING ====================

def load_model():
    """Load trained model"""
    print("\n Loading model...")
    
    if not os.path.exists(config.MODEL_PATH):
        print(f"WARNING: Model not found: {config.MODEL_PATH}")
        return None
    
    model = RadioSourceClassifier(num_classes=len(config.CLASSES))
    
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
        model = model.to(config.DEVICE)
        model.eval()
        print(f" Model loaded successfully")
        return model
    except Exception as e:
        print(f" Error loading model: {e}")
        return None

# ==================== PREDICTION ====================

def predict_batch(model, dataloader, mlb):
    """Make predictions for batch of images"""
    model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images in dataloader:
            images = images.to(config.DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > config.CONFIDENCE_THRESHOLD).cpu().numpy()
            
            all_preds.append(preds)
            all_probs.append(probs.cpu().numpy())
     
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
     
    results = []
    for i in range(len(all_preds)):
        pred = all_preds[i]
        prob = all_probs[i] 
        labels = mlb.inverse_transform(pred.reshape(1, -1))[0]
        
        if len(labels) == 0:
            labels = ('Unknown',)
        
        confidences = {mlb.classes_[j]: float(prob[j])
                      for j in range(len(mlb.classes_)) if pred[j]}
        
        results.append({
            'labels': list(labels),
            'confidences': confidences
        })
    
    return results

# ==================== TEST SET ====================

def predict_test_set(model, mlb):
    """Predict labels for test.csv"""
    print("\n" + "=" * 70)
    print("PREDICTING TEST SET")
    print("=" * 70)
    
    if not os.path.exists(config.TEST_FILE):
        print(f" {config.TEST_FILE} not found!")
        return None
    
    test_df = pd.read_csv(config.TEST_FILE, header=None, names=['ra', 'dec'])
    print(f" Loaded {len(test_df)} test coordinates")
     
    print(" Building image index...")
    image_dict = {}
    
    for directory in [config.TYP_DIR, config.EXO_DIR]:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.endswith('.png') and not filename.startswith('.'):
                    ra, dec = parse_filename_coordinates(filename)
                    if ra is not None:
                        image_dict[(ra, dec)] = os.path.join(directory, filename)
    
    print(f" Indexed {len(image_dict)} images")
     
    print(" Matching coordinates...")
    matched_indices = []
    test_image_paths = []
    
    for idx, row in test_df.iterrows():
        img_path = find_closest_image(row['ra'], row['dec'], image_dict, config.COORD_THRESHOLD)
        if img_path:
            test_image_paths.append(img_path)
            matched_indices.append(idx)
    
    print(f" Matched {len(test_image_paths)}/{len(test_df)} coordinates")
    
    if not test_image_paths:
        print(" No matches found!")
        return None 
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = PredictionDataset(test_image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Predict
    print(" Generating predictions...")
    predictions = predict_batch(model, dataloader, mlb)
    
    # Create results
    results = []
    pred_idx = 0
    
    for idx, row in test_df.iterrows():
        if idx in matched_indices:
            pred = predictions[pred_idx]
            results.append({
                'ra': row['ra'],
                'dec': row['dec'],
                'labels': pred['labels']
            })
            pred_idx += 1
        else:
            results.append({
                'ra': row['ra'],
                'dec': row['dec'],
                'labels': ['Unknown']
            })
    
    results_df = pd.DataFrame(results)
    save_labels_csv(results_df, 'test_labels.csv')
    
    print("\n Test predictions saved to test_labels.csv")
    
    # Statistics
    all_labels = [label for labels in results_df['labels'] for label in labels]
    unique, counts = np.unique(all_labels, return_counts=True)
    print("\n Test set predictions:")
    for label, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"   {label:25s}: {count:3d}")
    
    return results_df

# ==================== UNLABELED SET ====================

def predict_unlabeled_set(model, mlb):
    """Predict labels for unlabeled images"""
    print("\n" + "=" * 70)
    print("PREDICTING UNLABELED SET")
    print("=" * 70)
    
    if not os.path.exists(config.UNL_DIR):
        print(f" {config.UNL_DIR} not found!")
        return None
     
    files = [f for f in os.listdir(config.UNL_DIR) 
            if f.endswith('.png') and not f.startswith('.')]
    
    print(f" Found {len(files)} unlabeled images")
     
    image_data = []
    for filename in files:
        ra, dec = parse_filename_coordinates(filename)
        if ra is not None:
            image_data.append({
                'ra': ra,
                'dec': dec,
                'path': os.path.join(config.UNL_DIR, filename)
            })
    
    print(f" Parsed {len(image_data)} coordinates")
     
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_paths = [item['path'] for item in image_data]
    dataset = PredictionDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Predict
    print(" Generating predictions (this may take a while)...")
    predictions = predict_batch(model, dataloader, mlb)
    
    # Combine results
    results = []
    for i, item in enumerate(image_data):
        results.append({
            'ra': item['ra'],
            'dec': item['dec'],
            'labels': predictions[i]['labels']
        })
    
    results_df = pd.DataFrame(results)
    save_labels_csv(results_df, 'generated_labels.csv')
    
    print("\n Unlabeled predictions saved to generated_labels.csv")
    
    # Statistics
    all_labels = [label for labels in results_df['labels'] for label in labels]
    unique, counts = np.unique(all_labels, return_counts=True)
    
    print("\n Unlabeled set predictions:")
    for label, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        percentage = (count / len(all_labels)) * 100
        print(f"   {label:25s}: {count:5d} ({percentage:5.1f}%)")
    
    return results_df

# ==================== CSV SAVING ====================

def save_labels_csv(df, output_file):
    """Save predictions in CSV format"""
    max_labels = max(len(row['labels']) for _, row in df.iterrows())
    
    output_rows = []
    for _, row in df.iterrows():
        output_row = [row['ra'], row['dec']] + list(row['labels']) + [''] * (max_labels - len(row['labels']))
        output_rows.append(output_row)
    
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_file, index=False, header=False)
    print(f" Saved: {output_file}")

# ==================== MAIN ====================

def main():
    """Main execution""" 
    mlb = MultiLabelBinarizer(classes=config.CLASSES)
    mlb.fit([config.CLASSES]) 
    model = load_model()
    if model is None:
        print("\n Cannot proceed without model!")
        return
     
    test_results = predict_test_set(model, mlb)
    unlabeled_results = predict_unlabeled_set(model, mlb)
    
    # Summary
    print("\n" + "=" * 70)
    print(" PREDICTION COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    if test_results is not None:
        print(f"   test_labels.csv ({len(test_results)} rows)")
    if unlabeled_results is not None:
        print(f"   generated_labels.csv ({len(unlabeled_results)} rows)")
    print("\n #Files_Ready")
    print("=" * 70)

if __name__ == "__main__":
    main()