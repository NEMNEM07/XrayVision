# 1주차 활동 보고서

## 프로젝트: XrayVision
> 흉부 X-ray 이상 탐지 ViT 파인튜닝 동아리 프로젝트

---

## 주차 목표
- 개발 환경 세팅 완료
- 데이터셋 수집 및 전처리 파이프라인 구축
- PyTorch Dataset/DataLoader 구현

---

## 활동 요약 및 성과

### Day 1 — 개발 환경 세팅
- Python 3.11 기반 venv 가상환경 구성
- PyTorch nightly (cu128) 설치 — RTX 5090 Laptop (Blackwell, sm_120) 지원 버전으로 교체
  - 기존 PyTorch 2.5.1+cu121은 sm_120 미지원 → 2.12.0.dev+cu128로 업그레이드
- HuggingFace Transformers, OpenCV, scikit-learn 등 핵심 패키지 설치 완료
- 환경 확인 결과:
  ```
  PyTorch: 2.12.0.dev20260408+cu128
  CUDA 사용 가능: True
  GPU: NVIDIA GeForce RTX 5090 Laptop GPU
  VRAM: 25.7 GB
  Transformers: 5.8.0
  OpenCV: 4.13.0
  ```

### Day 2 — 데이터셋 수집 (1차 시도: ChestX-Det)
- RSNA, VinBigData, NIH ChestX-ray14 등 여러 데이터셋 검토
- **1차 선택: `natealberti/ChestX-Det` (HuggingFace)**
  - 선택 이유: Kaggle 전화번호 인증 문제로 HuggingFace 기반으로 전환
  - 3,578장 (정상 611장 / 이상 2,967장)
  - 13개 질환, image / mask / label 컬럼 포함
  - bbox + 세그멘테이션 마스크 전체 제공

> **이후 데이터셋 변경**: 2주차 학습 결과 AUROC 0.47 (랜덤 수준) 발생 → 데이터 부족이 주요 원인으로 판단 → **RSNA Pneumonia Detection(26,684장)으로 최종 교체** (2주차 참고)

### Day 3 — 데이터 전처리 1 (ChestX-Det 기준)
- RGBA → RGB 변환
- 224×224 리사이즈 (ViT 입력 규격)
- RGB 마스크 → 바이너리 마스크 변환 (이상=1, 정상=0)
- 픽셀값 0~255 → 0.0~1.0 정규화
- 전처리 결과 시각화 검증 완료
- **이 파이프라인은 이후 RSNA 전처리 설계의 기반이 됨**

### Day 4 — 데이터 분할 (ChestX-Det 기준)
- 전체 3,578장 라벨 수집 및 분포 분석
  - 정상: 611장 (17.1%)
  - 이상: 2,967장 (82.9%) → 클래스 불균형 확인
- stratify 옵션으로 비율 유지하며 8:1:1 분할
  - Train: 2,862장 / Val: 358장 / Test: 358장
- **이 분할 방식은 이후 RSNA 분할에도 동일하게 적용됨**

### Day 5 — PyTorch Dataset 클래스 구현
- `XrayDataset` 클래스 구현
- 기본 변환: ToTensor + ImageNet 정규화 (mean/std)
- 증강 변환: RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter, RandomAffine, GaussianBlur
- 출력 형태:
  ```
  이미지: torch.Size([3, 224, 224]), float32
  마스크: torch.Size([1, 224, 224]), float32
  라벨:   tensor(0. or 1.)
  ```

### Day 6 — DataLoader + 클래스 불균형 처리
- `WeightedRandomSampler` 적용 — 정상/이상 배치 내 비율 균형화
- DataLoader 구성:
  - Train: batch=32, sampler=WeightedRandomSampler, num_workers=4, pin_memory=True
  - Val/Test: batch=32, shuffle=False
- 배치 시각화 검증 완료

---

## 데이터셋 전환 히스토리

1주차에서 구축한 ChestX-Det 파이프라인은 2주차 학습에서 성능 문제로 RSNA로 전환됐지만, 1주차에서 확립한 핵심 설계 원칙들은 그대로 유지됐다.

| 항목 | ChestX-Det (1주차) | RSNA (최종) |
|------|-------------------|------------|
| 장수 | 3,578장 | 26,684장 |
| 포맷 | PNG (HuggingFace) | DICOM (.dcm) |
| 마스크 | 세그멘테이션 마스크 | bbox → 마스크 변환 |
| 분할 방식 | 8:1:1 stratify | 8:1:1 stratify (동일) |
| 증강 방식 | RandomFlip 등 | 동일 |
| 전환 이유 | - | AUROC 0.47 → 데이터 부족 판단 |

---

## 주요 수치 (ChestX-Det 기준)

| 항목 | 값 |
|------|-----|
| 전체 데이터 | 3,578장 |
| Train | 2,862장 |
| Val | 358장 |
| Test | 358장 |
| 이상 비율 | 82.9% |
| 이미지 해상도 | 224×224 |
| 배치 사이즈 | 32 |

---

## AI 활용 내역

### 활용한 부분

| 작업 | AI 활용 내용 | 직접 검토/수정 내용 |
|------|-------------|-------------------|
| 환경 세팅 | PyTorch CUDA 버전 선택 가이드 | RTX 5090 sm_120 호환 문제 직접 확인 후 nightly로 전환 결정 |
| 데이터셋 선택 | 여러 데이터셋 비교 분석 | Kaggle 인증 문제, 마스크 유무 등 직접 확인하며 최종 선택 |
| 전처리 코드 | preprocess_image, preprocess_mask 초안 | 바이너리 변환 로직 직접 검증, 시각화로 결과 확인 |
| Dataset 클래스 | XrayDataset 초안 생성 | augment 파라미터 구조, transform 구성 직접 수정 |
| DataLoader 설정 | WeightedRandomSampler 적용 방법 | num_workers, pin_memory 설정 직접 실험 |

### AI가 틀렸거나 수정한 사례
- **데이터셋 선택**: AI가 RSNA → VinBigData → NIH 순으로 추천했으나, 인증/등록 문제로 HuggingFace ChestX-Det으로 직접 결정 (결과적으로 2주차에서 데이터 부족 문제 발생 → RSNA로 재전환)
- **pip 환경 문제**: venv 내 pip 손상 문제는 AI 제안(`ensurepip --upgrade`)으로 해결했으나, 시스템 Python과 venv 충돌 원인은 직접 디버깅
- **datasets 모듈 경로**: 시스템 Python에 설치된 datasets과 venv 충돌 — AI 제안 방법으로 해결

### AI 활용 시간 절약 추정
- 데이터셋 조사: 약 2시간 절약
- 전처리 코드 작성: 약 1.5시간 절약
- 디버깅 (pip, CUDA 버전): 약 1시간 절약
- **총 추정 절약 시간: 약 4.5시간**

---

## 트러블슈팅 기록

| 문제 | 원인 | 해결 |
|------|------|------|
| `conda` 명령어 인식 안 됨 | Anaconda 미설치, venv 사용으로 전환 | `python -m venv .venv` |
| RTX 5090 CUDA 경고 | PyTorch 2.5.1이 sm_120 미지원 | PyTorch nightly cu128로 교체 |

---

## 생성된 파일

```
XrayVision/
├── week1/
│   ├── config.py          # 환경 확인 스크립트
│   └── data_split.py      # Train/Val/Test 분할 생성 (ChestX-Det 기준)
├── data/
│   └── data_split.json    # 분할 인덱스 (ChestX-Det 기준, 이후 RSNA로 대체)
├── dataset.py             # XrayDataset 클래스 (이후 RSNA 기준으로 재작성)
└── train.py               # DataLoader + Sampler (1주차 버전)
```

---

## 다음 주 계획 (2주차)

- [ ] ViT-Base/16 백본 로드 및 구조 파악
- [ ] Classification Head 구현 (CLS 토큰 → FC → BCEWithLogitsLoss)
- [ ] Segmentation Head 구현 (패치 토큰 → reshape → Upsample)
- [ ] 멀티태스크 모델 통합 (Total Loss = BCE + Dice)
- [ ] 학습 루프 구현 (AdamW + CosineAnnealingLR + Mixed Precision + Early Stopping)
- [ ] 학습 실행 및 성능 평가 → 데이터 부족 시 RSNA로 전환 검토