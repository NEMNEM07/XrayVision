# week3/generate_docs_images.py
import sys, os
sys.path.insert(0, r'C:\\Projects\\XrayVision')
sys.path.insert(0, r'C:\\Projects\\XrayVision\week2')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from torch.amp import autocast
from sklearn.model_selection import train_test_split
import pydicom
from PIL import Image
import torchvision.transforms as T
from dataset import XrayDataset, rsna_samples, load_rsna_image, make_rsna_mask
from model import XrayViT

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DOCS_IMG = 'docs/images'
os.makedirs(DOCS_IMG, exist_ok=True)

# =============================================
# 공통: 모델 + 테스트 데이터 준비
# =============================================
model = XrayViT().to(DEVICE)
model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location=DEVICE))
model.eval()
print("Model loaded!")

rsna_labels = [int(s['label']) for s in rsna_samples]
_, temp, _, temp_lbl = train_test_split(
    rsna_samples, rsna_labels, test_size=0.2, stratify=rsna_labels, random_state=42)
_, test_data, _, _ = train_test_split(
    temp, temp_lbl, test_size=0.5, stratify=temp_lbl, random_state=42)

test_ds = XrayDataset(test_data, augment=False)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(item):
    img_pil    = load_rsna_image(item['dcm_path'])
    img_np     = np.array(img_pil)
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        with autocast('cuda'):
            cls_prob, seg_map = model(img_tensor)
    prob      = torch.sigmoid(cls_prob).item()
    pred_mask = seg_map.squeeze().cpu().numpy()
    return img_np, pred_mask, prob

def overlay(img_np, mask_np, color, alpha=0.45, threshold=0.3):
    binary  = (mask_np >= threshold).astype(np.uint8)
    colored = np.zeros_like(img_np)
    colored[binary == 1] = color
    return cv2.addWeighted(img_np.copy(), 1.0, colored, alpha, 0)


# =============================================
# 1. week1_pipeline.png — 전처리 파이프라인
# =============================================
print("Generating week1_pipeline.png ...")

sample = [s for s in test_data if s['label'] == 1.0][0]
img_pil  = load_rsna_image(sample['dcm_path'])
img_np   = np.array(img_pil)
gt_mask  = make_rsna_mask(sample['bboxes']) if sample['bboxes'] else np.zeros((224,224))

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle('Week 1 — Data Preprocessing Pipeline', fontsize=13, fontweight='bold')

axes[0].imshow(img_np, cmap='gray')
axes[0].set_title('1. Raw X-ray (DICOM → RGB)')
axes[0].axis('off')

axes[1].imshow(img_np, cmap='gray')
axes[1].set_title('2. Resized 224×224')
axes[1].axis('off')

axes[2].imshow(gt_mask, cmap='hot')
axes[2].set_title('3. BBox → Binary Mask')
axes[2].axis('off')

plt.tight_layout()
plt.savefig(f'{DOCS_IMG}/week1_pipeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> week1_pipeline.png saved!")


# =============================================
# 2. week2_training_curve.png — 학습 곡선
# =============================================
print("Generating week2_training_curve.png ...")

epochs     = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 34]
train_loss = [0.6220, 0.5401, 0.5239, 0.5138, 0.5073, 0.5056, 0.5035, 0.5022, 0.4953,
              0.4910, 0.4870, 0.4830, 0.4800, 0.4780, 0.4760, 0.4730, 0.4710, 0.4700]
val_loss   = [0.5416, 0.5582, 0.4913, 0.5023, 0.5057, 0.4999, 0.5038, 0.4948, 0.4771,
              0.4810, 0.4850, 0.4830, 0.4860, 0.4880, 0.4900, 0.4930, 0.4950, 0.4960]
val_acc    = [0.770, 0.728, 0.785, 0.769, 0.768, 0.769, 0.766, 0.773, 0.790,
              0.788, 0.786, 0.784, 0.782, 0.780, 0.779, 0.776, 0.774, 0.772]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle('Week 2 — Training Curve (Early Stopping at Epoch 34)', fontsize=13, fontweight='bold')

ax1.plot(epochs, train_loss, 'b-o', markersize=4, label='Train Loss')
ax1.plot(epochs, val_loss,   'r-o', markersize=4, label='Val Loss')
ax1.axvline(x=9, color='green', linestyle='--', alpha=0.7, label='Best (Epoch 9)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Train / Val Loss')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(epochs, val_acc, 'g-o', markersize=4, label='Val Accuracy')
ax2.axhline(y=0.790, color='red', linestyle='--', alpha=0.7, label='Best Acc=0.790')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Val Accuracy')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_ylim(0.70, 0.85)

plt.tight_layout()
plt.savefig(f'{DOCS_IMG}/week2_training_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> week2_training_curve.png saved!")


# =============================================
# 3. week3_heatmap.png — 히트맵 비교
# =============================================
print("Generating week3_heatmap.png ...")

abnormal = [s for s in test_data if s['label'] == 1.0][:3]
normal   = [s for s in test_data if s['label'] == 0.0][:1]
samples  = abnormal + normal

fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))
fig.suptitle('Week 3 — Prediction Heatmap Comparison', fontsize=13, fontweight='bold')

for i, item in enumerate(samples):
    img_np, pred_mask, prob = predict(item)
    gt_mask = make_rsna_mask(item['bboxes']) if item['bboxes'] else np.zeros((224,224))

    axes[i][0].imshow(img_np, cmap='gray')
    axes[i][0].set_title(f'X-ray | GT={int(item["label"])}')
    axes[i][0].axis('off')

    axes[i][1].imshow(gt_mask, cmap='hot')
    axes[i][1].set_title('GT Mask (BBox)')
    axes[i][1].axis('off')

    axes[i][2].imshow(pred_mask, cmap='hot')
    axes[i][2].set_title(f'Pred Heatmap ({prob*100:.1f}%)')
    axes[i][2].axis('off')

plt.tight_layout()
plt.savefig(f'{DOCS_IMG}/week3_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> week3_heatmap.png saved!")


# =============================================
# 4. week3_overlay.png — 오버레이 결과
# =============================================
print("Generating week3_overlay.png ...")

abnormal = [s for s in test_data if s['label'] == 1.0][:3]
normal   = [s for s in test_data if s['label'] == 0.0][:1]
samples  = abnormal + normal

fig, axes = plt.subplots(len(samples), 2, figsize=(10, 5 * len(samples)))
fig.suptitle('Week 3 — Result Overlay', fontsize=13, fontweight='bold')

for i, item in enumerate(samples):
    img_np, pred_mask, prob = predict(item)
    color    = (255, 50, 50) if prob >= 0.5 else (50, 200, 50)
    overlaid = overlay(img_np, pred_mask, color=color)
    status   = 'ABNORMAL' if prob >= 0.5 else 'NORMAL'

    axes[i][0].imshow(img_np, cmap='gray')
    axes[i][0].set_title(f'Original | GT={int(item["label"])}')
    axes[i][0].axis('off')

    axes[i][1].imshow(overlaid)
    axes[i][1].set_title(f'Overlay | {status} ({prob*100:.1f}%)')
    axes[i][1].axis('off')
    patch = mpatches.Patch(color=np.array(color)/255,
                           label='Abnormal' if prob >= 0.5 else 'Caution')
    axes[i][1].legend(handles=[patch], loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig(f'{DOCS_IMG}/week3_overlay.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> week3_overlay.png saved!")


# =============================================
# 5. final_results.png — 최종 성능 요약
# =============================================
print("Generating final_results.png ...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('XrayVision — Final Results Summary', fontsize=14, fontweight='bold')

# 성능 지표 바차트
metrics_names  = ['AUROC', 'Accuracy', 'Normal\nPrecision', 'Abnormal\nRecall']
metrics_values = [0.8790,   0.79,       0.93,               0.79]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

bars = axes[0].bar(metrics_names, metrics_values, color=colors, alpha=0.85, edgecolor='white')
axes[0].set_ylim(0, 1.1)
axes[0].set_title('Performance Metrics')
axes[0].set_ylabel('Score')
axes[0].axhline(y=0.85, color='red', linestyle='--', alpha=0.5, label='Target AUROC=0.85')
axes[0].legend(fontsize=8)
for bar, val in zip(bars, metrics_values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# 데이터셋 비교
attempts     = ['1st Try\n(ViT-L\nChestX-Det)', '2nd Try\n(ViT-L\nChestX-Det)', 'Final\n(ViT-B\nRSNA)']
auroc_values = [0.47, 0.50, 0.879]
bar_colors   = ['#EF5350', '#EF5350', '#66BB6A']

bars2 = axes[1].bar(attempts, auroc_values, color=bar_colors, alpha=0.85, edgecolor='white')
axes[1].set_ylim(0, 1.1)
axes[1].set_title('AUROC Improvement')
axes[1].set_ylabel('AUROC')
axes[1].axhline(y=0.85, color='blue', linestyle='--', alpha=0.5, label='Target=0.85')
axes[1].legend(fontsize=8)
for bar, val in zip(bars2, auroc_values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# 베스트 샘플 오버레이
best = [s for s in test_data if s['label'] == 1.0]
best_item = None
best_prob = 0
for item in best[:20]:
    _, _, prob = predict(item)
    if prob > best_prob:
        best_prob = prob
        best_item = item

img_np, pred_mask, prob = predict(best_item)
color    = (255, 50, 50)
overlaid = overlay(img_np, pred_mask, color=color)
axes[2].imshow(overlaid)
axes[2].set_title(f'Best Detection\nABNORMAL ({prob*100:.1f}%)')
axes[2].axis('off')
patch = mpatches.Patch(color=np.array(color)/255, label='Abnormal Region')
axes[2].legend(handles=[patch], loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig(f'{DOCS_IMG}/final_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> final_results.png saved!")

print(f"""
==============================
docs/images/ 생성 완료! ✅
  - week1_pipeline.png
  - week2_training_curve.png
  - week3_heatmap.png
  - week3_overlay.png
  - final_results.png
==============================
""")