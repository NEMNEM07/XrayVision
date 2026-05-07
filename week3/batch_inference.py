# week3/batch_inference.py
sys.path.append('.')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from torch.amp import autocast
from sklearn.model_selection import train_test_split
from week2.dataset import XrayDataset, rsna_samples, load_rsna_image
from week2.model import XrayViT
from week3.inference import run_inference
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
model = XrayViT().to(DEVICE)
model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location=DEVICE))
print("Model loaded!")

# 테스트 데이터 준비
rsna_labels = [int(s['label']) for s in rsna_samples]
_, temp, _, temp_lbl = train_test_split(
    rsna_samples, rsna_labels, test_size=0.2, stratify=rsna_labels, random_state=42)
_, test_data, _, _ = train_test_split(
    temp, temp_lbl, test_size=0.5, stratify=temp_lbl, random_state=42)

# 정상 5개 + 이상 5개
abnormal = [s for s in test_data if s['label'] == 1.0][:5]
normal   = [s for s in test_data if s['label'] == 0.0][:5]

os.makedirs('results/samples', exist_ok=True)

print("\n=== 이상 샘플 추론 ===")
for i, item in enumerate(abnormal):
    prob, status, path = run_inference(
        item['dcm_path'], model, save_dir='results/samples')

print("\n=== 정상 샘플 추론 ===")
for i, item in enumerate(normal):
    prob, status, path = run_inference(
        item['dcm_path'], model, save_dir='results/samples')

print(f"\n전체 샘플 저장 완료! results/samples/ 확인해봐 ✅")

# 최종 성능 요약
print("""
========================================
      XrayVision 3주차 최종 점검
========================================
  AUROC:      0.8790
  Accuracy:   0.79
  Mean IoU:   0.3617
  Mean Dice:  0.4997
  정상 Precision: 0.93
  이상 Recall:    0.79
  학습 에포크:    34 (Early Stopping)
  Best Val Loss:  0.4771
========================================
""")