# XrayVision 

> 흉부 X-ray 이상 탐지 — ViT-Base/16 파인튜닝 + 멀티태스크 학습 (분류 + 세그멘테이션)

![최종 결과](docs/images/final_results.png)

---

## 프로젝트 소개

ViT-Base/16을 RSNA Pneumonia Detection 데이터셋(26,684장)으로 파인튜닝하여 흉부 X-ray에서 폐렴 이상 여부를 분류하고, 이상 영역을 세그멘테이션 마스크로 시각화하는 AI 시스템이다.

**AUROC 0.879** 달성 (목표 0.85 초과)

---

## 결과 예시

![오버레이 결과](docs/images/week3_overlay.png)

```
python week3/inference.py data/stage_2_train_images/[파일명].dcm

============================
File:   [파일명].dcm
Status: ABNORMAL
Prob:   84.6%
Saved:  results/[파일명]_result.png
============================
```

---

## 모델 구조

```
입력: [B, 3, 224, 224]
    ↓
ViT-Base/16 Backbone (Full Fine-tuning)
    ├── CLS 토큰 → Classification Head → 이상 확률 [0~1]
    └── 패치 토큰 → reshape(14×14) → Segmentation Head → 마스크 [224×224]
```

**Total Loss = 0.7 × LabelSmoothingBCE + 0.3 × DiceLoss**

---

## 성능

| 지표 | 값 |
|------|-----|
| **AUROC** | **0.8790** |
| Accuracy | 0.79 |
| 정상 Precision | 0.93 |
| 이상 Recall | 0.79 |
| Mean IoU | 0.3617 |
| Mean Dice | 0.4997 |

---

## 환경 설정

```bash
# 가상환경 생성
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows

# PyTorch (CUDA 12.8, RTX 5090 Blackwell 지원)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 나머지 패키지
pip install -r requirements.txt
```

환경 확인:
```bash
python week1/config.py
```

---

## 데이터셋

[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) (Kaggle)

다운로드 후 `data/` 폴더 구조:
```
data/
├── stage_2_train_images/    # DICOM 파일 (26,684장)
├── stage_2_train_labels.csv/
└── stage_2_detailed_class_info.csv/
```

---

## 사용법

### 학습

```bash
python week2/train.py
```

### 평가

```bash
python week2/metrics.py
```

### 단일 이미지 추론

```bash
python week3/inference.py [이미지 경로] --checkpoint checkpoints/best_model.pt
```

### 시각화

```bash
python week3/visualize.py
```

---

## 프로젝트 구조

```
XrayVision/
├── checkpoints/
│   └── best_model.pt          # 학습된 모델 (Epoch 9, Val Loss 0.4771)
├── data/                      # RSNA 데이터셋
├── docs/
│   ├── images/                # 문서용 이미지
│   ├── WEEK1_REPORT.md
│   ├── WEEK2_REPORT.md
│   ├── WEEK3_REPORT.md
│   └── FINAL_REPORT.md
├── results/
│   └── samples/               # 추론 결과 이미지
├── week1/
│   ├── config.py              # 환경 확인
├── week2/
│   ├── dataset.py             # RSNA Dataset 클래스
│   ├── model.py               # XrayViT 모델
│   ├── train.py               # 학습 루프
│   └── metrics.py             # AUROC, IoU, Dice 평가
├── week3/
│   ├── visualize.py           # 마스크 시각화
│   ├── inference.py           # CLI 추론
│   ├── batch_inference.py     # 배치 추론
│   └── generate_docs_images.py
├── README.md
└── requirements.txt
```

---

## 학습 곡선

![학습 곡선](docs/images/week2_training_curve.png)

Early Stopping이 Epoch 34에서 발동, Best model은 Epoch 9에서 저장됐다.

---

## 개발 환경

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA RTX 5090 Laptop |
| VRAM | 25.7 GB |
| CUDA | 12.8 (Blackwell, sm_120) |
| PyTorch | 2.12.0.dev+cu128 (nightly) |
| OS | Windows 11 |

---

## 참고 문헌

- Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words. *arXiv:2010.11929*
- Shih, G., et al. (2019). Augmenting the NIH Chest Radiograph Dataset. *Radiology: AI*
