import argparse, os, glob, torch
from PIL import Image
import pandas as pd
import collections
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from src.resnet.model import build_resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CLASS_ORDER = [
    "FR II","typical","Point Source","Bent","Should be discarded",
    "FR I","Exotic","S/Z shaped","X-Shaped"
]

class UnlabeledFolder(Dataset):
    def __init__(self, folder, img_size=224, files_glob="*.png"):
        self.paths = sorted(glob.glob(os.path.join(folder, files_glob)))
        self.tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert("RGB")
        return self.tfm(img), p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--unl_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--num_classes", type=int, default=9)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--classes_txt", default=None,
                    help="Opt fl")
    
    ap.add_argument("--low_thresh", type=float, default=0.35)
    args = ap.parse_args()

    # class order
    if args.classes_txt:
        with open(args.classes_txt) as f:
            class_order = [ln.strip() for ln in f if ln.strip()]
    else:
        class_order = DEFAULT_CLASS_ORDER[:args.num_classes]

    # dataloader
    ds = UnlabeledFolder(args.unl_dir, img_size=args.img_size)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # model 
    model = build_resnet50(num_classes=args.num_classes,
                           freeze_backbone=False, 
                           weights="IMAGENET1K_V2")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device).eval() 

    rows = [] 
    with torch.inference_mode():
        for x, paths in dl:
            x = x.to(device, non_blocking=True)
            logits = model(x)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            top1 = probs.argmax(axis=1)

            for p, i, vec in zip(paths, top1, probs):
                
                top3_idx = vec.argsort()[-3:][::-1]

                rows.append({
                    "path": p,
                    "pred_idx": int(i),
                    "pred_label": class_order[i] if i < len(class_order) else str(i),
                    "conf": float(vec[i]),

                    "top3_idx": ",".join(map(str, top3_idx.tolist())),
                    "top3_labels": ",".join([
                        class_order[j] if j < len(class_order) else str(j)
                        for j in top3_idx
                    ]),
                    "top3_conf": ",".join([f"{vec[j]:.4f}" for j in top3_idx]),
                })


    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Saved predictions → {args.out_csv}  (n={len(rows)})")

    counts = collections.Counter([r["pred_label"] for r in rows])
    total = len(rows)
    print("\nPredicted class distribution:")
    for k,v in counts.most_common():
        print(f"{k:>18}: {v:4d}  ({v/total:5.1%})")

    low = [r for r in rows if r["conf"] < 0.35]
    print(f"\nLow-confidence predictions (<0.35): {len(low)} / {len(rows)}")

if __name__ == "__main__":
    main()
