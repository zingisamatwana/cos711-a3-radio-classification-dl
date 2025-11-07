import argparse
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from src.config import CLASS_NAMES
from src.data import GalaxySingleHead
from src.resnet.model import build_resnet50

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with 'filepath' (and any extra cols you want preserved)")
    ap.add_argument("--weights", required=True, help="best.pt")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset in 'eval' mode (train=False) ignores labels if missing
    df = pd.read_csv(args.csv)
    if "label" not in df.columns:
        df["label"] = CLASS_NAMES[0]  # dummy (not used)
    ds = GalaxySingleHead(args.csv, CLASS_NAMES, train=False)

    model = build_resnet50(num_classes=len(CLASS_NAMES), freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    preds, probs = [], []
    with torch.no_grad():
        for i in tqdm(range(len(ds))):
            x, _ = ds[i]
            x = x.unsqueeze(0).to(device)
            logit = model(x)
            p = torch.softmax(logit, dim=1).cpu().numpy()[0]
            k = p.argmax()
            preds.append(CLASS_NAMES[k])
            probs.append(p[k])

    out = df.copy()
    out["pred_label"] = preds
    out["pred_conf"] = probs
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} with {len(out)} rows.")

if __name__ == "__main__":
    main()
