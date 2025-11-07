import argparse, re, os, sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict

from .config import DATA_DIR, CLASS_NAMES, SEED

def load_labels(labels_csv: Path) -> pd.DataFrame:
 
    df = pd.read_csv(labels_csv, header=None)
    df = df.iloc[:, :3]
    df.columns = ["ra", "dec", "label"]
 
    df = df[df["label"].isin(CLASS_NAMES)].copy()
    return df.reset_index(drop=True)

def _norm_float_for_match(x: float) -> str:
    
    s1 = f"{x:.6f}".rstrip("0").rstrip(".")
    s2 = f"{x:.3f}".rstrip("0").rstrip(".")
    s3 = f"{x:.2f}".rstrip("0").rstrip(".")
    s4 = str(float(f"{x:.6f}")) 
    return {s1, s2, s3, s4}

def index_all_images(root: Path):
 
    exts = {".png", ".jpg", ".jpeg"}
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files

def match_filepath(ra: float, dec: float, all_files):
    ra_candidates = _norm_float_for_match(ra)
    dec_candidates = _norm_float_for_match(dec)
 
    best = None
    for f in all_files:
        name = f.name
        hit_ra = any(s in name for s in ra_candidates)
        hit_dec = any(s in name for s in dec_candidates)
        if hit_ra and hit_dec:
            best = f
            break
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", type=str, default=str(DATA_DIR / "labels.csv"))
    ap.add_argument("--train_csv", type=str, default=str(DATA_DIR / "train.csv"))
    ap.add_argument("--val_csv", type=str, default=str(DATA_DIR / "val.csv"))
    ap.add_argument("--val_size", type=float, default=0.2)
    args = ap.parse_args()

    labels = load_labels(Path(args.labels_csv))
    print(f"Loaded labels: {labels.shape}")

    all_imgs = index_all_images(DATA_DIR)
    if not all_imgs:
        print("No images found ", file=sys.stderr)
        sys.exit(1)

    filepaths = []
    missing = 0
    for i, row in labels.iterrows():
        fp = match_filepath(row.ra, row.dec, all_imgs)
        if fp is None:
            filepaths.append(None)
            missing += 1
        else:
            filepaths.append(str(fp))

    labels["filepath"] = filepaths
    labels = labels.dropna(subset=["filepath"]).reset_index(drop=True)
    if missing:
        print(f"Warning: {missing} images  not matched and were dropped.")

 
    train_df, val_df = train_test_split(
        labels, test_size=args.val_size, random_state=SEED, stratify=labels["label"]
    )

    Path(args.train_csv).parent.mkdir(parents=True, exist_ok=True)
    train_df[["filepath", "label"]].to_csv(args.train_csv, index=False)
    val_df[["filepath", "label"]].to_csv(args.val_csv, index=False)

    print(f"Wrote {args.train_csv} ({len(train_df)}) and {args.val_csv} ({len(val_df)})")

if __name__ == "__main__":
    main()
