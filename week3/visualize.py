# visualize.py
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.amp import autocast
from sklearn.model_selection import train_test_split
import cv2
from week2.dataset import XrayDataset, rsna_samples, load_rsna_image, make_rsna_mask
from week2.model import XrayViT
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
model = XrayViT().to(DEVICE)
model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location=DEVICE))
model.eval()
print("Model loaded!")

# 테스트 데이터 준비
rsna_labels = [int(s['label']) for s in rsna_samples]
_, temp, _, temp_lbl = train_test_split(
    rsna_samples, rsna_labels, test_size=0.2, stratify=rsna_labels, random_state=42)
_, test_data, _, _ = train_test_split(
    temp, temp_lbl, test_size=0.5, stratify=temp_lbl, random_state=42)

test_ds = XrayDataset(test_data, augment=False)
os.makedirs('results', exist_ok=True)


def overlay_mask(img_np, mask_np, color=(255, 0, 0), alpha=0.4, threshold=0.3):
    """X-ray 위에 마스크 반투명 오버레이"""
    img_rgb  = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_rgb  = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    overlay  = img_rgb.copy()
    binary   = (mask_np >= threshold).astype(np.uint8)
    colored  = np.zeros_like(img_rgb)
    colored[binary == 1] = color
    result   = cv2.addWeighted(overlay, 1.0, colored, alpha, 0)
    return result


# 이상 4개 + 정상 2개
abnormal = [s for s in test_data if s['label'] == 1.0][:4]
normal   = [s for s in test_data if s['label'] == 0.0][:2]
samples  = abnormal + normal

fig, axes = plt.subplots(len(samples), 2, figsize=(10, 5 * len(samples)))
fig.suptitle('XrayVision - Result Overlay', fontsize=16, y=1.01)

for i, item in enumerate(samples):
    img_pil    = load_rsna_image(item['dcm_path'])
    img_np     = np.array(img_pil)

    img_tensor, _, _ = test_ds[test_data.index(item)]
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with autocast('cuda'):
            cls_prob, seg_map = model(img_tensor)

    prob      = torch.sigmoid(cls_prob).item()
    pred_mask = seg_map.squeeze().cpu().numpy()

    # 원본
    axes[i][0].imshow(img_np, cmap='gray')
    axes[i][0].set_title(f'Original | GT={int(item["label"])} | Pred={prob:.2f}')
    axes[i][0].axis('off')

    # 오버레이
    color    = (255, 50, 50) if prob >= 0.5 else (50, 200, 50)
    overlaid = overlay_mask(img_np, pred_mask, color=color, alpha=0.45)
    axes[i][1].imshow(overlaid)

    status = 'ABNORMAL' if prob >= 0.5 else 'NORMAL'
    axes[i][1].set_title(f'Overlay | {status} ({prob*100:.1f}%)')
    axes[i][1].axis('off')

    patch = mpatches.Patch(
        color=np.array(color)/255,
        label='Abnormal Region' if prob >= 0.5 else 'Caution Region')
    axes[i][1].legend(handles=[patch], loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig('results/overlay.png', dpi=150, bbox_inches='tight')
print("results/overlay.png saved! ✅")
