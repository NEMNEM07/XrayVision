# 3주차 활동 보고서

## 프로젝트: XrayVision
> 흉부 X-ray 이상 탐지 ViT 파인튜닝 동아리 프로젝트

---

## 📋 주차 목표
- 학습된 모델 시각화 및 평가
- 세그멘테이션 마스크 오버레이 구현
- inference CLI 구현
- 샘플 결과 이미지 생성

---

## ✅ 활동 요약 및 성과

### Day 15 — 세그멘테이션 마스크 시각화
- best_model.pt 로드 후 테스트셋 샘플 추론
- 원본 X-ray / GT 마스크 / 예측 마스크 3열 비교 시각화
- 주요 관찰:
  - 높은 확률(0.85) 샘플: GT bbox와 유사한 위치에 마스크 생성
  - 낮은 확률(0.19) 샘플: 마스크 강도 약함
  - **모델이 bbox 사각형이 아닌 실제 폐 형태로 예측** — 의학적으로 더 자연스러운 결과

### Day 16 — IoU / Dice Score 측정
- 이상 샘플(Target=1)에 대해서만 세그멘테이션 지표 계산
- 최종 결과:
  ```
  AUROC:     0.8790
  Mean IoU:  0.3617
  Mean Dice: 0.4997
  ```
- IoU/Dice가 상대적으로 낮은 이유:
  - GT가 bbox(사각형) 기반인 반면 모델은 폐 형태로 예측
  - 형태 불일치로 인한 수치 하락이지만 실제 시각적 품질은 우수

### Day 17 — 결과 오버레이 이미지
- X-ray 원본 위에 예측 마스크를 반투명 컬러로 오버레이
- 이상(ABNORMAL): 빨간색 오버레이
- 정상(NORMAL): 초록색 오버레이 (약하게)
- `results/overlay.png` 생성

### Day 18 — inference.py CLI 구현
- DCM 및 PNG/JPG 모두 지원하는 범용 추론 스크립트
- 커맨드라인 인터페이스:
  ```bash
  python inference.py <이미지 경로> --checkpoint checkpoints/best_model.pt
  ```
- 출력: 원본 / 히트맵 / 오버레이 3열 결과 이미지 자동 저장
- 테스트 결과:
  ```
  File:   00436515...dcm (실제 이상 케이스)
  Status: ABNORMAL
  Prob:   68.9%
  ```

### Day 19~20 — 샘플 배치 생성 + 최종 점검
- 테스트셋에서 이상 5개 + 정상 5개 = 총 10개 결과 이미지 일괄 생성
- `results/samples/` 에 저장
- 최종 성능 수치 확정

---

## 📊 최종 성능 요약

| 지표 | 값 | 평가 |
|------|-----|------|
| **AUROC** | **0.8790** | 논문급 🔥 |
| Accuracy | 0.79 | 양호 |
| Mean IoU | 0.3617 | 보통 (bbox GT 한계) |
| Mean Dice | 0.4997 | 보통 (bbox GT 한계) |
| 정상 Precision | 0.93 | 매우 좋음 ✅ |
| 이상 Recall | 0.79 | 좋음 ✅ |
| 학습 에포크 | 34 (Early Stopping) | - |
| Best Val Loss | 0.4771 | - |

---

## 🖼️ 생성된 결과물

### overlay.png
이상/정상 샘플 6개에 대한 오버레이 비교 이미지

### results/samples/
테스트셋 10개 샘플 개별 결과 이미지
- 이상 5개: ABNORMAL 빨간 오버레이
- 정상 5개: NORMAL 초록 오버레이 (약하게)

---

## 🤖 AI 활용 내역

| 작업 | AI 활용 내용 | 직접 검토/수정 |
|------|-------------|--------------|
| 시각화 코드 | matplotlib 3열 subplot 구성 | threshold 0.3 직접 조정 |
| 오버레이 구현 | cv2.addWeighted 적용 방법 | alpha 0.45, color 값 직접 실험 |
| IoU/Dice 구현 | 수식 코드화 | smooth=1e-8 직접 추가 |
| inference CLI | argparse 구조 | DCM/PNG 분기 처리 직접 수정 |
| 배치 추론 | 루프 구조 | 저장 경로 구조 직접 결정 |

### AI가 틀렸거나 수정한 사례
- **한글 폰트 경고**: AI가 한글 텍스트 포함 코드 작성 → matplotlib DejaVu Sans 폰트 미지원으로 경고 다수 발생 → 전체 영문으로 직접 수정

### AI 활용 시간 절약 추정
- 시각화 코드: 약 2시간 절약
- IoU/Dice 구현: 약 1시간 절약
- inference CLI: 약 1.5시간 절약
- **총 추정 절약 시간: 약 4.5시간**

---

## 🔧 트러블슈팅 기록

| 문제 | 원인 | 해결 |
|------|------|------|
| 한글 폰트 경고 다수 | matplotlib DejaVu Sans 한글 미지원 | 모든 텍스트 영문으로 교체 |
| IoU 낮은 수치 | GT bbox vs 모델 폐 형태 예측 불일치 | 수치 한계 인정, 시각적 품질로 보완 |

---

## 📁 생성된 파일

```
XrayVision/
├── week3/
│   └── batch_inference.py   # 배치 추론 스크립트
├── visualize.py             # 마스크 시각화 + 오버레이
├── inference.py             # CLI 추론 스크립트
├── metrics.py               # AUROC + IoU + Dice 평가
└── results/
    ├── visualization.png    # 마스크 비교 시각화
    ├── overlay.png          # 오버레이 결과
    └── samples/             # 개별 샘플 10개
```

---

## 📝 다음 주 계획 (4주차)

- [ ] 최종 보고서 작성
- [ ] 발표 자료 (PPT) 제작
- [ ] GitHub 정리 및 README 완성
- [ ] 시연 준비 (inference.py 데모)
