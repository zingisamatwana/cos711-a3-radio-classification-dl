import argparse
from pathlib import Path
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from src.config import (CLASS_NAMES, OUTPUT_DIR, FREEZE_BACKBONE, UNFREEZE_AT_EPOCH,
                    EPOCHS, BATCH_SIZE, BASE_LR, WEIGHT_DECAY, EARLY_STOP_PATIENCE, SEED)
from src.data import GalaxySingleHead
from src.resnet.model import build_resnet50
import random, os

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--out_dir", default=str(OUTPUT_DIR / "run_resnet50"))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=BASE_LR)
    args = ap.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = GalaxySingleHead(args.train_csv, CLASS_NAMES, train=True)
    val_ds   = GalaxySingleHead(args.val_csv,   CLASS_NAMES, train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    counts = np.array([ (train_ds.df["label"]==c).sum() for c in CLASS_NAMES ], dtype=float)
    w = 1.0 / np.clip(counts, 1.0, None)
    w = w / w.sum() * len(CLASS_NAMES)
    class_weights = torch.tensor(w, dtype=torch.float32, device=device)

    model = build_resnet50(num_classes=len(CLASS_NAMES), freeze_backbone=FREEZE_BACKBONE).to(device)
    crit  = torch.nn.CrossEntropyLoss(weight=class_weights)

    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=WEIGHT_DECAY)
    sch = CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    best_f1, patience = -1.0, 0

    for epoch in range(args.epochs):
        model.train()
        if epoch == UNFREEZE_AT_EPOCH:
            
            for p in model.parameters(): p.requires_grad = True
            opt = AdamW(model.parameters(), lr=args.lr/3, weight_decay=WEIGHT_DECAY)
            sch = CosineAnnealingLR(opt, T_max=args.epochs - epoch)

        # train
        tr_loss = 0.0
        for x, y in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step() 

            tr_loss += loss.detach().item() * x.size(0)
        sch.step()


        model.eval()
        va_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in tqdm(val_dl, desc=f"Epoch {epoch+1}/{args.epochs} [valid]"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = crit(logits, y)
                va_loss += float(loss) * x.size(0)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())

        ntr, nva = len(train_ds), len(val_ds)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"\nEpoch {epoch+1}: train_loss={tr_loss/ntr:.4f}  val_loss={va_loss/nva:.4f}  macroF1={macro_f1:.4f}")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), best_path)
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print("Early stopping.")
                break

    print(f"Best macro-F1: {best_f1:.4f}  (saved to {best_path})")

if __name__ == "__main__":
    main()
