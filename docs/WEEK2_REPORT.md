# 2주차 활동 보고서

## 프로젝트: XrayVision
> 흉부 X-ray 이상 탐지 ViT 파인튜닝 동아리 프로젝트

---

## 주차 목표
- ViT 모델 구현 및 학습 파이프라인 구축
- 멀티태스크 학습 (분류 + 세그멘테이션)
- RSNA 데이터셋 통합
- 학습 실행 및 성능 평가

---

## 활동 요약 및 성과

### Day 8 — ViT-L/16 로드 및 구조 파악
- `google/vit-large-patch16-224` HuggingFace에서 로드
- 구조 확인:
  ```
  hidden_size: 1024
  num_layers:  24
  num_heads:   16
  patch_size:  16
  image_size:  224
  패치 토큰:   [B, 196, 1024] (14×14 = 196개)
  CLS 토큰:    [B, 1024]
  ```

### Day 9~10 — 멀티태스크 모델 구현
- **Classification Head**: CLS 토큰 → FC → BCEWithLogitsLoss
- **Segmentation Head**: 패치 토큰 → reshape(14×14) → Conv → Bilinear Upsample → 224×224
- **DiceLoss**: 이상 샘플만 세그멘테이션 loss 계산 (정상은 스킵)
- **Total Loss**: `0.7 × BCE + 0.3 × Dice`

### Day 11 — 모델 크기 최적화
- 초기 ViT-Large(307M) → **ViT-Base(86M)** 교체
  - 이유: 데이터 26,000장 기준 ViT-Large는 과소적합 위험
  - ViT-Base가 이 데이터 크기에 최적
- hidden_size 1024 → 768로 헤드 전체 수정

### Day 12 — 학습 루프 구현
- Mixed Precision (AMP) 적용
- Gradient Clipping (max_norm=1.0)
- CosineAnnealingLR 스케줄러
- Early Stopping (patience=15)
- Label Smoothing BCE (smoothing=0.1)
- 분류 정확도 실시간 모니터링 추가

### Day 13 — RSNA 데이터셋 통합
- ChestX-Det(3,578장) → **RSNA Pneumonia Detection(26,684장)** 으로 교체
  - 이유: 데이터 부족으로 초기 AUROC 0.47 (랜덤 수준) 발생
  - RSNA가 bbox 레이블 포함, 규모 7배 이상
- DCM(DICOM) 포맷 처리 파이프라인 구현
  - `pydicom`으로 DCM 읽기
  - bbox → 바이너리 마스크 변환
  - 3개 클래스 (Normal / No Lung Opacity / Lung Opacity) → 이진 분류

### Day 14 — 학습 실행 및 평가
- 학습 설정:
  ```
  EPOCHS:     50 (Early Stopping으로 34에서 종료)
  BATCH_SIZE: 32
  LR:         2e-5
  ALPHA:      0.7 (BCE)
  BETA:       0.3 (Dice)
  PATIENCE:   15
  ```
- **최종 성능 (Test set)**:
  ```
  AUROC:    0.8790 
  Accuracy: 0.79
  정상 Precision: 0.93
  이상 Recall:    0.79
  ```

---

## 주요 수치

| 항목 | 값 |
|------|-----|
| 데이터셋 | RSNA Pneumonia Detection |
| 전체 데이터 | 26,684장 |
| Train | 21,347장 |
| Val | 2,669장 |
| Test | 2,669장 |
| 모델 | ViT-Base/16 (86M 파라미터) |
| 학습 에포크 | 34 (Early Stopping) |
| Best Val Loss | 0.4771 |
| **AUROC** | **0.8790** |
| Accuracy | 0.79 |

---

## 성능 개선 과정

| 시도 | 모델 | 데이터 | AUROC |
|------|------|--------|-------|
| 1차 | ViT-Large | ChestX-Det 3,578장 | 0.47 ❌ |
| 2차 | ViT-Large | ChestX-Det 3,578장 | 0.50 ❌ |
| 3차 | ViT-Base | RSNA 26,684장 | **0.879** ✅ |

---

## AI 활용 내역

| 작업 | AI 활용 내용 | 직접 검토/수정 |
|------|-------------|--------------|
| 모델 구조 설계 | Classification/Segmentation Head 초안 | hidden_size 1024→768 직접 수정 |
| DiceLoss 구현 | 이상 샘플 필터링 로직 | smooth 파라미터 직접 조정 |
| 학습 루프 | AMP, Gradient Clipping, Early Stopping | ALPHA/BETA 비율 직접 실험 |
| RSNA 전처리 | DCM 읽기, bbox→마스크 변환 코드 | orig_size=1024 직접 확인 |
| 하이퍼파라미터 | LR, BATCH_SIZE 후보 제시 | 실제 성능 보며 직접 결정 |

### AI가 틀렸거나 수정한 사례
- **ViT-Large 선택**: AI가 처음에 ViT-Large 추천 → 데이터 부족으로 AUROC 0.47 → 직접 ViT-Base로 교체 결정
- **BCEWithLogitsLoss label_smoothing**: AI가 존재하지 않는 파라미터 제안 → `LabelSmoothingBCE` 커스텀 클래스로 직접 구현

### AI 활용 시간 절약 추정
- 모델 구조 설계: 약 3시간 절약
- RSNA DCM 전처리: 약 2시간 절약
- 디버깅 (BCE, multiprocessing): 약 1.5시간 절약
- **총 추정 절약 시간: 약 6.5시간**

---

## 트러블슈팅 기록

| 문제 | 원인 | 해결 |
|------|------|------|
| AUROC 0.47 (랜덤 수준) | ViT-Large + 데이터 3,578장 부족 | ViT-Base + RSNA 26,684장으로 교체 |
| BCELoss autocast 오류 | BCELoss는 autocast 미지원 | BCEWithLogitsLoss로 교체 |
| label_smoothing 파라미터 없음 | BCEWithLogitsLoss에 미구현 | LabelSmoothingBCE 커스텀 클래스 구현 |

---

## 📁 생성된 파일

```
XrayVision/
├── week2/
│   ├── model.py           # ViT 모델 초안
│   ├── vram_check.py      # VRAM 사용량 측정
│   └── rsna_explore.py    # RSNA 데이터 탐색
├── model.py               # XrayViT (ViT-Base + 멀티태스크 헤드)
├── dataset.py             # RSNA Dataset 클래스
├── train.py               # 학습 루프
├── metrics.py             # AUROC + Classification Report
└── checkpoints/
    └── best_model.pt      # 최적 모델 (Epoch 9, Val Loss 0.4771)
```

---

## 📝 다음 주 계획 (3주차)

- [ ] 세그멘테이션 마스크 시각화 구현
- [ ] 결과 이미지 오버레이 (X-ray + 마스크)
- [ ] IoU, Dice Score 측정
- [ ] inference.py CLI 구현
- [ ] 샘플 결과 이미지 생성
- [ ] 최종 성능 정리 및 보고서 작성
