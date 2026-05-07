# inference.py
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import argparse
import os
import pydicom
from PIL import Image
from torch.amp import autocast
from week2.model import XrayViT
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(path):
    """DCM 또는 PNG/JPG 이미지 로드"""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.dcm':
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        img = img.astype(np.uint8)
        img_pil = Image.fromarray(img).convert('RGB')
    else:
        img_pil = Image.open(path).convert('RGB')
    img_pil = img_pil.resize((224, 224), Image.BILINEAR)
    return img_pil


def preprocess(img_pil):
    import torchvision.transforms as T
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return transform(img_pil).unsqueeze(0)


def overlay_mask(img_np, mask_np, color, alpha=0.45, threshold=0.3):
    overlay = img_np.copy()
    binary  = (mask_np >= threshold).astype(np.uint8)
    colored = np.zeros_like(img_np)
    colored[binary == 1] = color
    return cv2.addWeighted(overlay, 1.0, colored, alpha, 0)


def run_inference(img_path, model, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)

    # 이미지 로드 + 전처리
    img_pil    = load_image(img_path)
    img_np     = np.array(img_pil)
    img_tensor = preprocess(img_pil).to(DEVICE)

    # 추론
    model.eval()
    with torch.no_grad():
        with autocast('cuda'):
            cls_prob, seg_map = model(img_tensor)

    prob      = torch.sigmoid(cls_prob).item()
    pred_mask = seg_map.squeeze().cpu().numpy()

    # 결과 판정
    status = 'ABNORMAL' if prob >= 0.5 else 'NORMAL'
    color  = (255, 50, 50) if prob >= 0.5 else (50, 200, 50)

    # 오버레이
    overlaid = overlay_mask(img_np, pred_mask, color=color)

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'XrayVision | {status} ({prob*100:.1f}%)', fontsize=14)

    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(pred_mask, cmap='hot')
    axes[1].set_title('Prediction Heatmap')
    axes[1].axis('off')

    axes[2].imshow(overlaid)
    axes[2].set_title(f'Overlay | {status}')
    axes[2].axis('off')
    patch = mpatches.Patch(color=np.array(color)/255,
                           label='Abnormal' if prob >= 0.5 else 'Caution')
    axes[2].legend(handles=[patch], loc='lower right')

    plt.tight_layout()

    # 저장
    fname    = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(save_dir, f'{fname}_result.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"============================")
    print(f"File:   {img_path}")
    print(f"Status: {status}")
    print(f"Prob:   {prob*100:.1f}%")
    print(f"Saved:  {out_path}")
    print(f"============================")

    return prob, status, out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XrayVision Inference')
    parser.add_argument('image', type=str, help='Path to X-ray image (.dcm or .png/.jpg)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--save_dir', type=str, default='results')
    args = parser.parse_args()

    # 모델 로드
    model = XrayViT().to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    print(f"Model loaded: {args.checkpoint}")

    run_inference(args.image, model, save_dir=args.save_dir)