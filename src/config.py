from pathlib import Path

# Image paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# categories 
CLASS_NAMES = [
    "FR II",
    "typical",
    "Point Source",
    "Bent",
    "Should be discarded",
    "FR I",
    "Exotic",
    "S/Z shaped",
    "X-Shaped",
]

# Image Attribute for augmentation
IMG_SIZE = 256
CROP_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Training
EPOCHS = 25
BATCH_SIZE = 32
BASE_LR = 1e-4
WEIGHT_DECAY = 1e-4
FREEZE_BACKBONE = True     
UNFREEZE_AT_EPOCH = 5 
EARLY_STOP_PATIENCE = 6
SEED = 42
