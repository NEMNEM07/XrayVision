# XrayVision 🫁

흉부 X-ray 이상 탐지 ViT 파인튜닝 동아리 프로젝트

## 프로젝트 개요

**ViT-L/16 Full Fine-tuning → 흉부 X-ray 입력 → 이상 확률(%) + 이상 영역 마스크 오버레이된 결과 이미지 출력**

### 최종 출력 형태
```
원본 X-ray + 이상 영역 마스크 오버레이
이상 확률: 87.3%  ██████████░░
판정: ⚠️ 이상 의심
저장: results/[파일명]_result.png
```

---

## 모델 구조

```
ViT-L/16 Backbone (Full Fine-tuning)
        │
        ├──→ Classification Head
        │    CLS 토큰 → FC → Sigmoid → 이상 확률 (0~100%)
        │
        └──→ Segmentation Head
             패치 토큰 → reshape(14×14) → Bilinear Upsample → 224×224 마스크
```

### 손실 함수
```
Total Loss = α · BCE Loss (분류) + β · Dice Loss (세그멘테이션)
초기값: α=0.5, β=0.5
```

---

## 데이터셋

- **ChestX-Det** (HuggingFace: `natealberti/ChestX-Det`)
- 3,578장 (정상 611장 / 이상 2,967장)
- bbox + 세그멘테이션 마스크 포함
- Train/Val/Test = 8:1:1 (stratify 분할)

---

## 기술 스택

| 항목 | 선택 |
|------|------|
| 베이스 모델 | ViT-L/16 (HuggingFace Transformers) |
| 프레임워크 | PyTorch + HuggingFace |
| Optimizer | AdamW (lr=1e-5) |
| Scheduler | CosineAnnealingLR |
| 학습 기법 | Mixed Precision + Gradient Clipping |
| 배치 사이즈 | 16 |
| 입력 해상도 | 224×224 |
| 평가 지표 | AUROC, IoU, Dice Score |

---

## 환경 설정

```bash
# 가상환경 생성
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows

# PyTorch (CUDA 12.8, Blackwell 지원)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 나머지 패키지
pip install transformers accelerate
pip install opencv-python matplotlib pillow
pip install scikit-learn tqdm
pip install datasets
```

환경 확인:
```bash
python config.py
```

---

## 파일 구조

```
XrayVision/
├── config.py        # 환경 확인
├── dataset.py       # ChestX-Det 전처리 + PyTorch Dataset
├── train.py         # DataLoader + 학습 루프
├── model.py         # ViT-L/16 + 멀티태스크 헤드
├── metrics.py       # AUROC, IoU, Dice Score
├── visualize.py     # 마스크 오버레이 + 결과 이미지 저장
├── inference.py     # CLI 추론
├── data_split.json  # Train/Val/Test 분할 인덱스
├── requirements.txt
└── results/         # 출력 이미지 저장
```

---

## 진행 상황

### 1주차 — 환경 세팅 + 데이터 준비 ✅
- [x] Day 1: 환경 세팅 (PyTorch CUDA 12.8, venv)
- [x] Day 2: 데이터셋 준비 (ChestX-Det HuggingFace)
- [x] Day 3: 전처리 (이미지/마스크 바이너리 변환)
- [x] Day 4: Train/Val/Test 분할 (8:1:1, stratify)
- [x] Day 5: PyTorch Dataset 클래스 구현
- [x] Day 6: DataLoader + WeightedRandomSampler

### 2주차 — 모델 구현 + 학습 🔄
- [ ] Day 8: ViT-L/16 로드
- [ ] Day 9: Classification Head
- [ ] Day 10: Segmentation Head
- [ ] Day 11: 멀티태스크 모델 통합
- [ ] Day 12: 학습 루프
- [ ] Day 13: 학습 실행

### 3주차 — 평가 + 결과 이미지
- [ ] Day 15~21

### 4주차 — 보고서 + 발표
- [ ] Day 22~28

---

## GPU 환경

- NVIDIA GeForce RTX 5090 Laptop GPU
- VRAM: 25.7GB
- CUDA: 12.8 (Blackwell, sm_120)
- PyTorch: 2.12.0.dev (nightly)
