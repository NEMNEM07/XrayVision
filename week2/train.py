# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from week2.dataset import XrayDataset, rsna_samples
from week2.model import XrayViT, DiceLoss
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(pred, target)


def main():
    DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS     = 50
    BATCH_SIZE = 32
    LR         = 2e-5
    ALPHA      = 0.7
    BETA       = 0.3
    PATIENCE   = 15
    SAVE_DIR   = 'checkpoints'
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")

    # RSNA 8:1:1 분할
    rsna_labels = [int(s['label']) for s in rsna_samples]
    train_data, temp_data, train_lbl, temp_lbl = train_test_split(
        rsna_samples, rsna_labels, test_size=0.2, stratify=rsna_labels, random_state=42)
    val_data, test_data, _, _ = train_test_split(
        temp_data, temp_lbl, test_size=0.5, stratify=temp_lbl, random_state=42)

    print(f"Train: {len(train_data)}장")
    print(f"Val:   {len(val_data)}장")
    print(f"Test:  {len(test_data)}장")

    train_ds = XrayDataset(train_data, augment=True)
    val_ds   = XrayDataset(val_data,   augment=False)

    # WeightedRandomSampler
    weights = [1.0 / (train_lbl.count(l) + 1e-8) for l in train_lbl]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    # 모델 / 손실 / 옵티마이저
    model     = XrayViT().to(DEVICE)
    bce_loss  = LabelSmoothingBCE(smoothing=0.1)
    dice_loss = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler('cuda')

    best_val_loss    = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # === Train ===
        model.train()
        train_loss    = 0.0
        train_correct = 0
        train_total   = 0

        for imgs, masks, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            imgs         = imgs.to(DEVICE)
            masks        = masks.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)

            optimizer.zero_grad()
            with autocast('cuda'):
                cls_prob, seg_map = model(imgs)
                loss_bce  = bce_loss(cls_prob.squeeze(), labels_batch)
                loss_dice = dice_loss(seg_map, masks)
                loss = ALPHA * loss_bce + BETA * loss_dice

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

            preds = (torch.sigmoid(cls_prob.squeeze()) >= 0.5).float()
            train_correct += (preds == labels_batch).sum().item()
            train_total   += labels_batch.size(0)

        train_loss /= len(train_loader)
        train_acc   = train_correct / train_total

        # === Validation ===
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for imgs, masks, labels_batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                imgs         = imgs.to(DEVICE)
                masks        = masks.to(DEVICE)
                labels_batch = labels_batch.to(DEVICE)
                with autocast('cuda'):
                    cls_prob, seg_map = model(imgs)
                    loss_bce  = bce_loss(cls_prob.squeeze(), labels_batch)
                    loss_dice = dice_loss(seg_map, masks)
                    loss = ALPHA * loss_bce + BETA * loss_dice
                val_loss += loss.item()

                preds = (torch.sigmoid(cls_prob.squeeze()) >= 0.5).float()
                val_correct += (preds == labels_batch).sum().item()
                val_total   += labels_batch.size(0)

        val_loss /= len(val_loader)
        val_acc   = val_correct / val_total

        scheduler.step()
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{SAVE_DIR}/best_model.pt')
            print(f"  ✅ Best model saved! (val_loss={val_loss:.4f}, val_acc={val_acc:.3f})")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"\n⏹ Early Stopping!")
                break

    print(f"\n학습 완료! ✅")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"체크포인트: {SAVE_DIR}/best_model.pt")


if __name__ == '__main__':
    main()