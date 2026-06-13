import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from transformers import ViTModel
from PIL import Image
from torchvision import transforms
from pathlib import Path
import sys, os
import warnings
warnings.filterwarnings('ignore')

# ── 경로 설정 ──────────────────────────────────────────────
CHECKPOINT = "C:/Projects/XrayVision/checkpoints/best_model.pt"
IMAGE_DIR  = "C:/Projects/XrayVision/data/stage_2_train_images"
CSV_PATH   = "C:/Projects/XrayVision/data/stage_2_train_labels.csv/stage_2_train_labels.csv"
OUTPUT_DIR = "C:/Projects/XrayVision/pca_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SAMPLES  = 300   # 클래스당 샘플 수 (속도 조절)
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── 모델 정의 ──────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        mask = (target.sum(dim=(1,2,3)) > 0)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        pred, target = pred[mask], target[mask]
        pred_flat, target_flat = pred.view(-1), target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2*intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)

class XrayViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.cls_head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 1)
        )
        self.seg_head = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),  nn.ReLU(),
            nn.Conv2d(64, 1, 1), nn.Sigmoid()
        )
    def forward(self, pixel_values):
        out = self.vit(pixel_values=pixel_values)
        cls_token    = out.last_hidden_state[:, 0, :]
        cls_prob     = self.cls_head(cls_token)
        patch_tokens = out.last_hidden_state[:, 1:, :]
        B = patch_tokens.shape[0]
        patch_tokens = patch_tokens.permute(0,2,1).reshape(B, 768, 14, 14)
        seg_map = self.seg_head(patch_tokens)
        seg_map = F.interpolate(seg_map, size=(224,224), mode='bilinear', align_corners=False)
        return cls_prob, seg_map

# ── 전처리 ─────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

def load_dcm(path):
    dcm = pydicom.dcmread(path)
    arr = dcm.pixel_array.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    return transform(img)

# ── CSV 로드 및 샘플링 ─────────────────────────────────────
print("CSV 로드 중...")
df = pd.read_csv(CSV_PATH)

# Target 컬럼 확인 (1=폐렴, 0=정상)
label_col = "Target" if "Target" in df.columns else df.columns[-1]
df = df[["patientId", label_col]].drop_duplicates("patientId").reset_index(drop=True)
df.columns = ["patientId", "label"]

normal   = df[df.label == 0].sample(n=min(N_SAMPLES, (df.label==0).sum()), random_state=42)
pneumonia= df[df.label == 1].sample(n=min(N_SAMPLES, (df.label==1).sum()), random_state=42)
df_sample = pd.concat([normal, pneumonia]).reset_index(drop=True)
print(f"정상: {len(normal)}장 / 폐렴: {len(pneumonia)}장")

# ── 모델 로드 ──────────────────────────────────────────────
print("모델 로드 중...")
model = XrayViT().to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
state = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state)
model.eval()
print("모델 로드 완료")

# ── 특징벡터 추출 ──────────────────────────────────────────
features, labels = [], []
fail = 0

print("특징벡터 추출 중...")
with torch.no_grad():
    for i, row in df_sample.iterrows():
        dcm_path = Path(IMAGE_DIR) / f"{row.patientId}.dcm"
        if not dcm_path.exists():
            fail += 1
            continue
        try:
            img = load_dcm(dcm_path).unsqueeze(0).to(DEVICE)
            out = model.vit(pixel_values=img)
            cls_vec = out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # 768차원
            features.append(cls_vec)
            labels.append(row.label)
        except Exception as e:
            fail += 1
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(df_sample)} 처리 완료")

print(f"추출 완료: {len(features)}개 (실패: {fail}개)")

X = np.array(features)   # (N, 768)
y = np.array(labels)

# ── 공분산 행렬 직접 계산 (수학 탐구용) ────────────────────
print("\n공분산 행렬 계산 중...")
X_centered = X - X.mean(axis=0)
C = (X_centered.T @ X_centered) / len(X_centered)   # (768, 768)
print(f"공분산 행렬 크기: {C.shape}")

# ── numpy 고윳값 분해 ──────────────────────────────────────
print("고윳값 분해 중...")
eigenvalues, eigenvectors = np.linalg.eigh(C)
# 내림차순 정렬
idx = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

explained_ratio = eigenvalues / eigenvalues.sum()
cumulative      = np.cumsum(explained_ratio)

print(f"\n주성분별 설명 분산 비율:")
for i in range(5):
    print(f"  PC{i+1}: 고윳값={eigenvalues[i]:.4f} | 분산비율={explained_ratio[i]*100:.2f}% | 누적={cumulative[i]*100:.2f}%")

# ── 2D 투영 ────────────────────────────────────────────────
W = eigenvectors[:, :2]          # (768, 2)
Z = X_centered @ W               # (N, 2)

# ── 시각화 1: PCA 산점도 ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
colors = {0: "#4C9BE8", 1: "#E85C4C"}
cnames = {0: "Normal", 1: "Pneumonia"}

for label in [0, 1]:
    mask = y == label
    ax.scatter(Z[mask, 0], Z[mask, 1],
               c=colors[label], label=cnames[label],
               alpha=0.55, s=25, edgecolors='none')

ax.set_xlabel(f"PC1 ({explained_ratio[0]*100:.1f}%)", fontsize=12)
ax.set_ylabel(f"PC2 ({explained_ratio[1]*100:.1f}%)", fontsize=12)
ax.set_title("PCA of XrayVision CLS Token Features (768D → 2D)", fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_scatter.png"), dpi=150)
plt.close()
print("pca_scatter.png 저장 완료")

# ── 시각화 2: 설명 분산 비율 (Scree Plot) ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 상위 20개 고윳값
k = 20
axes[0].bar(range(1, k+1), explained_ratio[:k]*100, color="#4C9BE8", edgecolor='white')
axes[0].set_xlabel("Principal Component (PC)", fontsize=11)
axes[0].set_ylabel("Explained Variance Ratio (%)", fontsize=11)
axes[0].set_title("Scree Plot (Top 20 PCs)", fontsize=12, fontweight='bold')
axes[0].set_xticks(range(1, k+1))

# 누적 분산
axes[1].plot(range(1, k+1), cumulative[:k]*100, marker='o', color="#E85C4C", linewidth=2)
axes[1].axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='90%')
axes[1].axhline(y=95, color='gray', linestyle=':',  alpha=0.7, label='95%')
axes[1].set_xlabel("Number of Principal Components", fontsize=11)
axes[1].set_ylabel("Cumulative Explained Variance (%)", fontsize=11)
axes[1].set_title("Cumulative Explained Variance", fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].set_xticks(range(1, k+1))
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scree_plot.png"), dpi=150)
plt.close()
print("scree_plot.png 저장 완료")

# ── 결과 요약 ──────────────────────────────────────────────
print(f"\n=== 결과 요약 ===")
print(f"특징벡터 차원: 768")
print(f"PC1 설명 분산: {explained_ratio[0]*100:.2f}%")
print(f"PC2 설명 분산: {explained_ratio[1]*100:.2f}%")
print(f"PC1+PC2 누적:  {cumulative[1]*100:.2f}%")
n_90 = np.searchsorted(cumulative, 0.90) + 1
n_95 = np.searchsorted(cumulative, 0.95) + 1
print(f"90% 달성 주성분 수: {n_90}개")
print(f"95% 달성 주성분 수: {n_95}개")
print(f"\n결과 이미지 저장 위치: {OUTPUT_DIR}")