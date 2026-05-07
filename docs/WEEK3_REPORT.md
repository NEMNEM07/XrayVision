# 3주차 활동 보고서
## 프로젝트: XrayVision — 흉부 X-ray 이상 탐지

---

## 목표
학습된 모델 시각화, 세그멘테이션 마스크 오버레이, inference CLI 구현, 샘플 결과 이미지 생성

---

## 세그멘테이션 마스크 시각화

`best_model.pt`를 로드하여 테스트셋 샘플에 대한 예측 결과를 시각화했다.

원본 X-ray, GT 마스크(bbox 기반), 예측 히트맵을 3열로 비교했다.

![히트맵 비교](images/week3_heatmap.png)

**주요 관찰:**
- 모델이 bbox 사각형이 아닌 **실제 폐 형태**로 마스크를 예측함
- 높은 확률(0.85)의 이상 샘플에서 양쪽 폐 영역을 정확히 포착
- GT(bbox)와 형태가 달라 IoU/Dice 수치가 낮게 나오지만, 의학적으로는 더 자연스러운 형태

---

## 결과 오버레이 이미지

X-ray 위에 예측 마스크를 반투명 컬러로 오버레이했다.

- **ABNORMAL** (prob ≥ 0.5): 빨간색 오버레이
- **NORMAL** (prob < 0.5): 초록색 오버레이

![오버레이 결과](images/week3_overlay.png)

**오버레이 구현 (`week3/visualize.py`):**
```python
def overlay_mask(img_np, mask_np, color, alpha=0.45, threshold=0.3):
    binary  = (mask_np >= threshold).astype(np.uint8)
    colored = np.zeros_like(img_np)
    colored[binary == 1] = color
    return cv2.addWeighted(img_np.copy(), 1.0, colored, alpha, 0)
```

threshold=0.3, alpha=0.45는 시각적 품질을 직접 실험하며 결정했다.

---

## 세그멘테이션 평가 (`week2/metrics.py`)

이상 샘플에 대해서만 IoU와 Dice Score를 계산했다.

```
AUROC:     0.8790
Mean IoU:  0.3617
Mean Dice: 0.4997
```

**IoU/Dice가 낮은 이유:**
GT 마스크가 bbox 기반 사각형인 반면, 모델은 실제 폐 윤곽 형태로 예측한다. 형태 불일치로 수치가 낮게 나오지만 시각적 품질은 우수하다.

---

## inference CLI (`week3/inference.py`)

DCM 및 PNG/JPG 이미지를 입력받아 결과 이미지를 저장하는 CLI를 구현했다.

**사용법:**
```bash
python week3/inference.py data/stage_2_train_images/[파일명].dcm
python week3/inference.py [이미지경로] --checkpoint checkpoints/best_model.pt --save_dir results
```

**출력 형태:**
```
============================
File:   [이미지 경로]
Status: ABNORMAL
Prob:   84.6%
Saved:  results/[파일명]_result.png
============================
```

원본 / 히트맵 / 오버레이 3열 이미지를 자동 생성한다.

---

## 샘플 배치 결과

테스트셋에서 이상 5개 + 정상 5개 = 총 10개 결과 이미지를 `results/samples/`에 저장했다.

| 케이스 | GT | 예측 확률 | 판정 |
|--------|-----|---------|------|
| 이상 A | 1 | 84.6% | ABNORMAL ✅ |
| 이상 B | 1 | 78.3% | ABNORMAL ✅ |
| 이상 C | 1 | 46.6% | NORMAL ❌ (경계선) |
| 이상 D | 1 | 19.4% | NORMAL ❌ (미탐) |
| 정상 A | 0 | 5.4% | NORMAL ✅ |
| 정상 B | 0 | 73.2% | ABNORMAL ❌ (오탐) |

---

## AI 활용 내역

| 작업 | AI 활용 | 직접 판단/수정 |
|------|--------|--------------|
| 시각화 구성 | matplotlib subplot 초안 | threshold 0.3, alpha 0.45 직접 조정 |
| IoU/Dice 구현 | 수식 코드화 | smooth=1e-8 추가 |
| inference CLI | argparse 구조 | DCM/PNG 분기 처리 직접 수정 |

**AI가 틀린 사례:** 한글 텍스트 포함 코드를 제공했으나 matplotlib에서 깨짐 발생 → 전체 영문으로 교체.

---

## 생성 파일

```
week3/
├── visualize.py           # 마스크 시각화 + 오버레이
├── inference.py           # CLI 추론 스크립트
├── batch_inference.py     # 배치 추론 (10개 샘플)
└── generate_docs_images.py # 문서용 이미지 생성

results/
└── samples/               # 개별 샘플 10개 결과 이미지

docs/images/
├── week1_pipeline.png
├── week2_training_curve.png
├── week3_heatmap.png
├── week3_overlay.png
└── final_results.png
```