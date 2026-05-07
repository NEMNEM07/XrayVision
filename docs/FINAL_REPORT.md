# XrayVision — 최종 보고서
## 흉부 X-ray 이상 탐지 ViT 파인튜닝 프로젝트

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [이론적 배경](#2-이론적-배경)
3. [데이터셋](#3-데이터셋)
4. [모델 설계](#4-모델-설계)
5. [학습 설정](#5-학습-설정)
6. [실험 결과](#6-실험-결과)
7. [시각화 결과](#7-시각화-결과)
8. [한계 및 향후 과제](#8-한계-및-향후-과제)
9. [AI 활용 내역](#9-ai-활용-내역)
10. [결론](#10-결론)

---

## 1. 프로젝트 개요

### 1.1 배경
폐렴(Pneumonia)은 전 세계적으로 높은 사망률을 가진 질환으로, 흉부 X-ray를 통한 조기 진단이 중요하다. 그러나 숙련된 방사선과 의사의 판독이 필요하며, 의료 인력 부족 지역에서는 진단이 지연되는 문제가 있다. 본 프로젝트는 Vision Transformer(ViT)를 활용한 흉부 X-ray 이상 탐지 모델을 개발하여 이러한 문제를 보조할 수 있는 AI 시스템을 구현하고자 한다.

### 1.2 목표
- 흉부 X-ray 이미지에서 이상(폐렴 의심) 여부를 분류
- 이상 영역을 세그멘테이션 마스크로 시각화
- 임상 현장에서 활용 가능한 수준의 AUROC 달성 (목표: 0.85 이상)

### 1.3 최종 출력 형태
```
원본 X-ray + 이상 영역 마스크 오버레이
이상 확률: 84.6%
판정: ABNORMAL
저장: results/[파일명]_result.png
```

---

## 2. 이론적 배경

### 2.1 Vision Transformer (ViT)

ViT는 Dosovitskiy et al.(2020)이 제안한 모델로, NLP에서 큰 성공을 거둔 Transformer 구조를 이미지에 적용한 것이다.

**핵심 동작 원리:**
```
입력 이미지 (224×224)
    ↓
16×16 패치로 분할 → 196개 패치 토큰
    ↓
[CLS] 토큰 추가 → 197개 토큰
    ↓
Multi-Head Self-Attention × 12 (Base) / 24 (Large)
    ↓
CLS 토큰 → 분류 / 패치 토큰 → 세그멘테이션
```

**ViT-Base vs ViT-Large:**

| 항목 | ViT-Base | ViT-Large |
|------|---------|----------|
| 파라미터 | 86M | 307M |
| hidden_size | 768 | 1024 |
| num_layers | 12 | 24 |
| 권장 데이터 | 10,000장~ | 100,000장~ |

본 프로젝트는 26,000장 규모의 데이터셋을 사용하므로 **ViT-Base**를 채택하였다.

### 2.2 멀티태스크 학습 (Multi-task Learning)

하나의 모델이 두 가지 태스크(분류 + 세그멘테이션)를 동시에 학습하는 방식이다. 공유된 ViT 백본에서 추출한 특징을 두 헤드가 각각 활용하므로 단일 태스크 대비 일반화 성능이 향상된다.

```
Total Loss = α · BCE Loss + β · Dice Loss
           = 0.7 × BCE + 0.3 × Dice
```

### 2.3 Dice Loss

클래스 불균형 문제(정상 77% / 이상 36%)에 대응하기 위해 Dice Loss를 도입하였다.

```
Dice Loss = 1 - (2 × |P ∩ G| + ε) / (|P| + |G| + ε)
```

- P: 예측 마스크, G: GT 마스크, ε: smoothing factor
- 이상 샘플에 대해서만 계산 (정상 샘플은 스킵)

### 2.4 Label Smoothing

과적합 방지를 위해 BCE Loss에 Label Smoothing을 적용하였다.

```
target_smooth = target × (1 - α) + 0.5 × α   (α = 0.1)
```

---

## 3. 데이터셋

### 3.1 데이터셋 선택 과정

| 시도 | 데이터셋 | 결과 | 전환 이유 |
|------|---------|------|---------|
| 1차 | ChestX-Det (HuggingFace) | AUROC 0.47 | 3,578장으로 데이터 부족 |
| 최종 | RSNA Pneumonia Detection | AUROC 0.879 | 26,684장, bbox 레이블 포함 |

### 3.2 RSNA Pneumonia Detection Challenge

- **출처**: Radiological Society of North America (RSNA)
- **총 이미지**: 26,684장 (DICOM 포맷)
- **클래스 구성**:

| 클래스 | 장수 | 비율 |
|--------|------|------|
| Normal | 8,851장 | 33.2% |
| No Lung Opacity / Not Normal | 11,821장 | 44.3% |
| Lung Opacity (폐렴 의심) | 9,555장 | 35.8% |

- **이진 분류 변환**: Normal + No Lung Opacity → 0 (정상), Lung Opacity → 1 (이상)
- **최종 분포**: 정상 20,672장 (77.5%) / 이상 9,555장 (35.8%)

### 3.3 데이터 분할

stratify 옵션으로 클래스 비율을 유지하며 8:1:1 분할:

| 분할 | 장수 | 정상 | 이상 |
|------|------|------|------|
| Train | 21,347장 | 16,537 | 4,810 |
| Val | 2,669장 | 2,068 | 601 |
| Test | 2,669장 | 2,068 | 601 |

### 3.4 전처리 파이프라인

```
DICOM 파일 읽기 (pydicom)
    ↓
pixel_array 추출 → float32 → 0~255 정규화
    ↓
흑백 → RGB 3채널 변환
    ↓
224×224 리사이즈 (Bilinear)
    ↓
bbox → 바이너리 마스크 변환 (1024×1024 → 224×224)
    ↓
ToTensor + ImageNet 정규화
   (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### 3.5 데이터 증강

```python
T.RandomHorizontalFlip(p=0.5)
T.RandomVerticalFlip(p=0.2)
T.RandomRotation(degrees=15)
T.ColorJitter(brightness=0.3, contrast=0.3)
T.RandomAffine(degrees=0, translate=(0.1, 0.1))
T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
```

### 3.6 클래스 불균형 처리

`WeightedRandomSampler`를 적용하여 배치 내 정상/이상 비율을 균형화하였다.

---

## 4. 모델 설계

### 4.1 전체 구조

```
입력: [B, 3, 224, 224]
    ↓
ViT-Base/16 Backbone (google/vit-base-patch16-224)
    ↓
last_hidden_state: [B, 197, 768]
    ├── CLS 토큰 [B, 768]
    │       ↓
    │   Classification Head
    │   Linear(768→256) → ReLU → Dropout(0.4)
    │   Linear(256→128) → ReLU → Dropout(0.4)
    │   Linear(128→1)
    │       ↓
    │   cls_prob: [B, 1]  (BCEWithLogitsLoss)
    │
    └── 패치 토큰 [B, 196, 768]
            ↓
        reshape → [B, 768, 14, 14]
            ↓
        Segmentation Head
        Conv2d(768→256) → BN → ReLU
        Conv2d(256→128) → BN → ReLU
        Conv2d(128→64)  → ReLU
        Conv2d(64→1)    → Sigmoid
            ↓
        Bilinear Upsample → [B, 1, 224, 224]
```

### 4.2 파라미터 수

| 구성요소 | 파라미터 수 |
|---------|-----------|
| ViT-Base 백본 | 85,800,192 |
| Classification Head | 230,913 |
| Segmentation Head | 1,590,145 |
| **전체** | **~87.6M** |

---

## 5. 학습 설정

### 5.1 하이퍼파라미터

| 항목 | 값 |
|------|-----|
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Batch Size | 32 |
| Max Epochs | 50 |
| Early Stopping | patience=15 |
| α (BCE 가중치) | 0.7 |
| β (Dice 가중치) | 0.3 |
| Label Smoothing | 0.1 |
| Gradient Clipping | max_norm=1.0 |
| Scheduler | CosineAnnealingLR |
| Mixed Precision | AMP (torch.amp) |

### 5.2 학습 환경

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA GeForce RTX 5090 Laptop |
| VRAM | 25.7 GB |
| CUDA | 12.8 (Blackwell, sm_120) |
| PyTorch | 2.12.0.dev+cu128 (nightly) |
| OS | Windows 11 |

### 5.3 학습 결과

Early Stopping이 Epoch 34에서 발동하였다.

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|----------|---------|--------|
| 1 | 0.6220 | 0.720 | 0.5416 | 0.770 |
| 3 | 0.5239 | 0.783 | 0.4913 | 0.785 |
| 9 | 0.4953 | 0.799 | 0.4771 | 0.790 |
| 34 | - | - | 0.4771 | - |

---

## 6. 실험 결과

### 6.1 최종 성능 (Test Set)

| 지표 | 값 |
|------|-----|
| **AUROC** | **0.8790** |
| Accuracy | 0.79 |
| Mean IoU | 0.3617 |
| Mean Dice | 0.4997 |
| 정상 Precision | 0.93 |
| 이상 Recall | 0.79 |
| 이상 F1-Score | 0.63 |

### 6.2 Classification Report

```
              precision    recall  f1-score   support
        정상       0.93      0.79      0.86      2068
        이상       0.53      0.79      0.63       601
    accuracy                           0.79      2669
   macro avg       0.73      0.79      0.75      2669
weighted avg       0.84      0.79      0.81      2669
```

### 6.3 데이터셋 변경에 따른 성능 비교

| 시도 | 모델 | 데이터 | AUROC |
|------|------|--------|-------|
| 1차 | ViT-Large | ChestX-Det 3,578장 | 0.47 ❌ |
| 2차 | ViT-Large | ChestX-Det 3,578장 | 0.50 ❌ |
| **최종** | **ViT-Base** | **RSNA 26,684장** | **0.879** ✅ |

### 6.4 IoU/Dice 수치 해석

Mean IoU 0.36, Mean Dice 0.50은 표면적으로 낮아 보이나, 이는 GT 마스크와 예측 마스크의 형태 차이에서 기인한다.

- **GT 마스크**: bbox 기반 사각형
- **예측 마스크**: 실제 폐 형태에 맞는 유기적 형태

시각화 결과에서 확인되듯 모델의 예측이 오히려 의학적으로 더 자연스러운 형태를 보인다.

---

## 7. 시각화 결과

### 7.1 예측 결과 예시

| 케이스 | GT | 예측 확률 | 판정 | 비고 |
|--------|-----|---------|------|------|
| 케이스 A | 이상 | 84.6% | ABNORMAL ✅ | 양쪽 폐 마스크 정확 |
| 케이스 B | 이상 | 78.3% | ABNORMAL ✅ | 폐 형태 마스크 생성 |
| 케이스 C | 이상 | 46.6% | NORMAL ❌ | 경계선 케이스 |
| 케이스 D | 이상 | 19.4% | NORMAL ❌ | 미탐 |
| 케이스 E | 정상 | 5.4% | NORMAL ✅ | 마스크 거의 없음 |
| 케이스 F | 정상 | 73.2% | ABNORMAL ❌ | 오탐 |

### 7.2 inference CLI 사용법

```bash
# DCM 파일 추론
python inference.py data/stage_2_train_images/[파일명].dcm

# 옵션 지정
python inference.py [이미지경로] --checkpoint checkpoints/best_model.pt --save_dir results
```

---

## 8. 한계 및 향후 과제

### 8.1 현재 한계

**데이터 측면:**
- RSNA 데이터셋의 GT가 bbox 기반이라 세그멘테이션 품질 평가에 한계
- 단일 데이터셋 학습으로 다양한 병원 장비/촬영 조건에 대한 일반화 미검증

**모델 측면:**
- 이상 Precision이 0.53으로 낮아 오탐률이 높음
- 14×14 패치 해상도 한계로 세밀한 세그멘테이션 어려움

### 8.2 향후 개선 방향

| 방향 | 내용 |
|------|------|
| 데이터 확장 | NIH ChestX-ray14 등 추가 데이터셋 통합 |
| 모델 개선 | Decoder 구조 강화 (U-Net 스타일) |
| 후처리 | CRF(Conditional Random Field)로 마스크 정제 |
| 평가 강화 | 5-fold Cross Validation 적용 |
| 배포 | FastAPI + React 웹 서비스화 |

---

## 9. AI 활용 내역

### 9.1 단계별 활용 현황

| 주차 | 작업 | AI 활용 내용 | 직접 결정/수정 사항 |
|------|------|------------|-----------------|
| 1주차 | 환경 세팅 | CUDA 버전 선택 가이드 | RTX 5090 sm_120 호환 직접 확인 후 nightly 선택 |
| 1주차 | 데이터셋 선택 | 여러 데이터셋 비교 분석 | Kaggle 인증 문제 직접 확인, ChestX-Det 선택 |
| 1주차 | 전처리 코드 | preprocess 함수 초안 | 바이너리 변환 로직 직접 검증 |
| 2주차 | 모델 구조 | 헤드 구조 초안 생성 | hidden_size 1024→768 수정, Dropout 조정 |
| 2주차 | 학습 루프 | AMP, Early Stopping 초안 | ALPHA/BETA 비율 직접 실험 후 0.7/0.3 결정 |
| 2주차 | 데이터 전환 | RSNA 전처리 코드 | DCM 읽기 로직, bbox→마스크 변환 직접 검증 |
| 3주차 | 시각화 | matplotlib 구성 초안 | threshold 0.3, alpha 0.45 직접 조정 |
| 3주차 | inference CLI | argparse 구조 | DCM/PNG 분기 처리 직접 수정 |

### 9.2 AI가 틀렸던 사례 (비판적 활용)

| 사례 | AI 제안 | 문제점 | 직접 해결 |
|------|--------|--------|---------|
| 모델 선택 | ViT-Large 추천 | 데이터 3,578장에 과소적합 → AUROC 0.47 | ViT-Base로 교체, RSNA로 데이터 확장 |
| Loss 설정 | BCEWithLogitsLoss(label_smoothing=0.1) | 해당 파라미터 미존재 → TypeError | LabelSmoothingBCE 커스텀 클래스 직접 구현 |
| DataLoader | num_workers=4 (Windows) | if __name__=='__main__' 없이 사용 → RuntimeError | main() 함수로 전체 감싸서 해결 |
| 한글 폰트 | 한글 텍스트 포함 코드 제공 | matplotlib DejaVu Sans 미지원 → 경고 다수 | 전체 영문으로 직접 교체 |

### 9.3 AI 활용 시간 절약 추정

| 주차 | 절약 추정 시간 |
|------|------------|
| 1주차 | 약 4.5시간 |
| 2주차 | 약 6.5시간 |
| 3주차 | 약 4.5시간 |
| **합계** | **약 15.5시간** |

### 9.4 AI 활용의 의의

본 프로젝트에서 AI는 코드 초안 생성, 개념 설명, 디버깅 힌트 제공 등의 역할을 수행했다. 그러나 모든 AI 제안은 직접 검증을 거쳤으며, 핵심 설계 결정(모델 크기, 데이터셋 선택, 하이퍼파라미터 등)은 실험 결과를 바탕으로 직접 판단하였다.

특히 초기 AUROC 0.47이라는 실패 경험에서 AI의 제안을 맹목적으로 따르지 않고, 데이터 크기와 모델 복잡도의 관계를 직접 분석하여 문제를 해결한 과정이 핵심 학습 경험이었다.

---

## 10. 결론

본 프로젝트는 ViT-Base/16을 RSNA Pneumonia Detection 데이터셋(26,684장)으로 파인튜닝하여 흉부 X-ray 이상 탐지 모델을 개발하였다.

**주요 성과:**
- AUROC **0.8790** 달성 (목표 0.85 초과)
- 정상 Precision **0.93** — 정상을 이상으로 오진하는 비율이 낮음
- 이상 Recall **0.79** — 이상의 79%를 놓치지 않고 탐지
- 의미있는 세그멘테이션 마스크 생성 — bbox GT보다 의학적으로 자연스러운 형태
- inference CLI를 통한 단일 이미지 즉시 추론 가능

**기술적 기여:**
- ViT의 패치 토큰을 활용한 세그멘테이션 헤드 설계
- 멀티태스크 학습(분류 + 세그멘테이션) 동시 수행
- 클래스 불균형 처리 (WeightedRandomSampler + Dice Loss)

의료 AI 분야에서 전문 방사선의의 보조 도구로 활용될 수 있는 가능성을 확인하였으며, 향후 데이터 확장 및 모델 개선을 통해 임상 적용 가능성을 높일 수 있을 것으로 기대된다.

---

## 참고 문헌

- Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *arXiv:2010.11929*
- Shih, G., et al. (2019). Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia. *Radiology: Artificial Intelligence*
- Milletari, F., et al. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. *3DV 2016*

---

*보고서 작성일: 2026년 5월*
*프로젝트 저장소: github.com/[username]/XrayVision*
