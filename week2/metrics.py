# metrics.py 전체 교체
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from week2.dataset import XrayDataset, rsna_samples
from week2.model import XrayViT
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_seg_metrics(pred_mask, gt_mask, threshold=0.5):
    """IoU + Dice Score 계산"""
    pred = (pred_mask >= threshold).astype(np.float32)
    gt   = gt_mask.astype(np.float32)

    intersection = (pred * gt).sum()
    union        = pred.sum() + gt.sum() - intersection
    dice_sum     = pred.sum() + gt.sum()

    iou  = (intersection + 1e-8) / (union + 1e-8)
    dice = (2 * intersection + 1e-8) / (dice_sum + 1e-8)

    return iou, dice


def evaluate(model, loader, device):
    model.eval()
    all_probs  = []
    all_labels = []
    iou_list   = []
    dice_list  = []

    with torch.no_grad():
        for imgs, masks, labels_batch in tqdm(loader, desc="Evaluating"):
            imgs         = imgs.to(device)
            masks        = masks.to(device)
            labels_batch = labels_batch.to(device)

            with autocast('cuda'):
                cls_prob, seg_map = model(imgs)

            probs = torch.sigmoid(cls_prob.squeeze()).cpu().numpy()
            if probs.ndim == 0:
                probs = [probs.item()]
            all_probs.extend(probs)
            all_labels.extend(labels_batch.cpu().numpy())

            # 이상 샘플만 세그멘테이션 평가
            seg_preds = seg_map.cpu().numpy()
            seg_gts   = masks.cpu().numpy()
            lbl_np    = labels_batch.cpu().numpy()

            for j in range(len(lbl_np)):
                if lbl_np[j] == 1.0:
                    iou, dice = compute_seg_metrics(
                        seg_preds[j][0], seg_gts[j][0])
                    iou_list.append(iou)
                    dice_list.append(dice)

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    auroc = roc_auc_score(all_labels, all_probs)
    preds = (all_probs >= 0.5).astype(int)

    mean_iou  = np.mean(iou_list)
    mean_dice = np.mean(dice_list)

    print(f"\n=== 평가 결과 ===")
    print(f"AUROC:      {auroc:.4f}")
    print(f"Mean IoU:   {mean_iou:.4f}")
    print(f"Mean Dice:  {mean_dice:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, preds, target_names=['정상', '이상']))

    return auroc, mean_iou, mean_dice


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rsna_labels = [int(s['label']) for s in rsna_samples]
    _, temp_data, _, temp_lbl = train_test_split(
        rsna_samples, rsna_labels, test_size=0.2, stratify=rsna_labels, random_state=42)
    _, test_data, _, _ = train_test_split(
        temp_data, temp_lbl, test_size=0.5, stratify=temp_lbl, random_state=42)

    print(f"Test: {len(test_data)}장")

    test_ds     = XrayDataset(test_data, augment=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    model = XrayViT().to(DEVICE)
    model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location=DEVICE))
    print("모델 로드 완료!")

    auroc, iou, dice = evaluate(model, test_loader, DEVICE)