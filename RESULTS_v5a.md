# v5a 실험 정리 — bbox+point 마커 LoRA (2026-07-29)

DROID 990장으로 FLUX-Kontext LoRA 를 학습해, 마커(집을 물체 bbox + 놓을 위치 점)로
지정한 조작의 **실행 후 장면**을 생성한다. 손실 함수의 공간 가중을 축으로 4종을 비교했다.

**결론: 마커 영역에만 곱셈 가중을 건 `alpha990_wlm4` 가 최고.** copy 기준선을
PSNR·SSIM·LPIPS 모두에서 이긴 유일한 모델이다.

---

## 1. 데이터

| split | 개수 | 구성 |
|---|---|---|
| train | 990 | AUTOLab 224 / TRI 154 / CLVR 133 / IPRL 98 / PennPAL 82 / RPL 77 / GuptaLab 66 / RAIL 62 / RAD 47 / REAL 47 |
| test_unseen | 121 | **ILIAD 55 / WEIRD 35 / IRIS 31** (학습에 없는 랩, setup 해시 겹침 0) |
| test_seen | 64 | 학습과 같은 10개 랩의 다른 에피소드 |

전체 페어 1,212 중 pour/wipe 26개(마커와 동작 불일치) + 무정보 11개 제외 = 1,175.

- 이미지: `/data/datasets/DROID/train_v4/{clean,kontext,final}` + `train_v5a/kontext`(반투명)
- CSV: `papers/SWEET/data/prompt_flux/v5a_{train,test_unseen,test_seen}.csv`
- 캐시: `/data/sweet_cache/v5a_alpha990` (임베딩 사전계산, 990개)

### 마커 렌더링 (`poc/scripts/data_prep/build_v5_alpha.py`)

```python
cv2.rectangle(over, (g0,g1), (g2,g3), (0,210,0), 4)      # grasp bbox, 초록 4px
cv2.circle(over, pp, 16, (255,60,30), -1)                # place point, 반경 16px 고정
cv2.circle(over, pp, 16, (255,255,255), 2)               # 흰 테두리
cv2.addWeighted(over, 0.55, clean, 0.45, 0)              # 반투명 합성
```

### 프롬프트

```
Follow the markers in the image: the green box marks the object to grasp, the blue dot
marks where to place it. Show the scene after the object has been moved to the marked
location. <pick 문장>. <place 문장>.
```

pick/place 문장은 사람이 쓴 어노테이션(`place_target`)을 동사형으로 통일한 것.
동사로 시작하면 대문자화만, 명사구면 `Place` 부착 (`build_v5_prompts.py:verbify`).
전치사 분포: on 54% / in 28% / inside 4% / on top of 4%.

---

## 2. 학습 설정 (전 모델 공통)

| | |
|---|---|
| 베이스 | FLUX.1-Kontext-dev, LoRA rank 32, lr 1e-4, AdamW |
| 배치 | 2 (DDP 2×RTX5090, 각 1) |
| 스텝 | 1,485 = 990 ÷ 2 × 3 epoch (wl4long 만 2,970 = 6 epoch) |
| 해상도 | 1280×720 → latent 16×90×160 → 3,600 토큰(1토큰 = 원본 16×16px) |
| 시간 | 3.49 s/it, 약 87분 |
| 스크립트 | `papers/SWEET/train_cached.sh` (캐시 2-pass) |

---

## 3. 손실 4종

`diffsynth/diffusion/loss.py` — flow matching MSE 에 **공간 가중**만 추가.
`WLOSS_LAMBDA=0`, `WLOSS_BETA=0` 이면 원본과 완전히 동일한 경로로 빠진다.

| # | 이름 | 마스크 | 손실 식 |
|---|---|---|---|
| 1 | `alpha990` | — | `mean(se)` ← SWEET 원본 |
| 2 | `alpha990_wl4` | 변화영역(\|clean−GT\|>20) ∪ 마커 | `(se·w).sum()/w.sum()`, `w=1+4·m` |
| 2b | `alpha990_wl4long` | 위와 동일 | 위와 동일, 6 epoch |
| 3 | **`alpha990_wlm4`** | **마커만** (bbox + 원판 0.75×max변, 하한 24px) | `w=1+4·m` |
| 4 | `alpha990_bal5box` | 마커만 (원판 **0.5×**max변, 하한 8px) | `β·mean(마스크) + (1−β)·mean(배경)`, β=0.5 |

- 마스크는 픽셀 단위로 만든 뒤 `avg_pool2d(8)` 로 latent 격자에 내린다 → 경계는 소수값.
- 캐시 `.pth` 의 `d[0]["loss_weight"]` (1,1,90,160) 로 주입 (`inject_loss_weight.py`).
- 마스크는 **모델 입력이 아니다.** forward 에 관여하지 않고 손실 가중에만 쓰인다.

### 왜 2번이 실패했나

**변화영역의 90.7%가 로봇 팔이다** (unseen 121 실측, 중앙값 93.7%). 팔에 5배 가중이
걸려 팔을 그리느라 배경이 무너졌다 (정지 MAE 8.46 → 14.00).
시각 자료: `outputs/viz/loss_mask/grid.png`, `why_speckle.png`

### 왜 4번이 3번을 못 이겼나

곱셈 가중(`w=1+λm`)은 **마커 몫이 면적에 비례**해 작은 물체는 2%, 큰 물체는 27%로
13배 차이 난다. β 균형은 크기와 무관하게 항상 50%를 준다 — 의도는 옳았으나 결과는
보수적인 모델이 됐다(dst 41.18 로 copy 40.95 보다 나쁨). **원판 축소와 동시에 바꿔
원인 분리가 안 됨.** 필요하면 중간 조건(균형 손실 + 원판 0.75×)을 하나 더 돌려야 한다.

---

## 4. 결과

### test_unseen (121장)

| | PSNR↑ | SSIM↑ | LPIPS↓ | MAE변화↓ | MAE정지↓ | MAE src↓ | MAE dst↓ |
|---|---|---|---|---|---|---|---|
| Copy (아무것도 안 함) | 17.12 | 0.788 | 0.202 | 74.47 | 4.28 | 56.97 | 40.95 |
| 1. 균등 MSE (SWEET) | 17.04 | 0.778 | 0.208 | 59.15 | 8.46 | 29.36 | 40.43 |
| 2. λ4 · 변화영역∪마커 | 15.62 | 0.736 | 0.251 | **58.78** | 14.00 | 32.91 | 49.15 |
| 2b. 2번 + 6 epoch | 16.22 | 0.755 | 0.226 | 55.65 | 12.20 | **28.85** | 39.55 |
| **3. λ4 · 마커만** | **17.53** | **0.792** | **0.195** | 61.13 | 6.42 | 29.16 | **39.06** |
| 4. 영역균형 β0.5 + 원판축소 | 17.03 | 0.777 | 0.210 | 68.46 | 6.77 | 28.96 | 41.18 |
| (참) 균등 + 불투명 마커 | 16.97 | 0.776 | 0.205 | 59.70 | 8.59 | 31.17 | 41.15 |

### test_seen (64장) — 1·2·3번만 평가

| | PSNR↑ | SSIM↑ | LPIPS↓ | MAE변화↓ | MAE정지↓ | MAE src↓ | MAE dst↓ |
|---|---|---|---|---|---|---|---|
| Copy | 17.76 | 0.820 | 0.192 | 71.09 | 4.62 | 59.19 | 35.16 |
| 1. 균등 MSE | 18.02 | 0.818 | 0.186 | **53.62** | 8.10 | **30.82** | **33.70** |
| 2. λ4 · 변화영역∪마커 | 15.73 | 0.770 | 0.243 | 55.06 | 14.88 | 32.11 | 42.87 |
| **3. λ4 · 마커만** | **18.53** | **0.833** | **0.179** | 55.48 | 6.36 | 31.49 | 34.27 |

### 팔 제외 물체 영역 (unseen 45장 — 팔이 마커를 70% 미만 가린 것만)

| | src↓ | dst↓ | 평균↓ | copy 대비 |
|---|---|---|---|---|
| **3. 마커만** | **11.91** | **22.05** | **16.98** | **−26.5%** |
| 1. 균등 MSE | 15.12 | 22.55 | 18.84 | −18.4% |
| 2b. 6 epoch | 14.15 | 25.93 | 20.04 | −13.2% |
| 2. 변화영역∪마커 | 16.13 | 34.98 | 25.56 | +10.6% |
| copy | 31.69 | 14.51 | 23.10 | — |

---

## 5. 지표 해석 시 주의

| 지표 | 계산 | 주의 |
|---|---|---|
| MAE 변화 | `\|pred−GT\|` 를 `\|clean−GT\|>20` 영역에서 평균 | **90.7%가 로봇 팔.** 사실상 "팔을 그렸나"를 잰다 |
| MAE 정지 | 위의 여집합 | copy 값(4.28)은 정의상 20 미만 — 성능 아님 |
| MAE src | grasp bbox 안 | GT 에는 물체가 없어야 함 → 잘 지우면 낮음 |
| MAE dst | place point 원판(반경 max(24, 0.75×max변)) 안 | 원판이 물체 면적의 2.5배라 배경이 섞임 |
| PSNR/SSIM/LPIPS | 전체 프레임 | **copy 17.12 vs 최고 17.53 (2.4% 차)** — 태스크를 직접 재지 못함 |

**dst 가 안 뚫리는 이유는 모델이 아니라 지표·라벨 쪽일 가능성이 크다:**

1. **팔·그리퍼 가림** — unseen 121 중 76장(63%)에서 팔이 마커 영역의 70% 이상을 가림
2. **중력·굴러감** — place point 는 그리퍼가 *놓은* 지점이지 물체가 *안착한* 지점이 아님
3. **원판이 과대** — 물체 면적의 2.5배. src bbox 크기로 바꾸면 copy 대비 개선폭이
   4.6% → 9.1% 로 두 배가 된다

---

## 6. 인과 실험 (지표보다 설득력 있음)

| 실험 | 결과 | 파일 |
|---|---|---|
| place point 6곳 이동 | 물체가 **평균 6.4px 오차**로 따라감. 점이 멀티탭 위면 멀티탭 위에 올려놓음 | `outputs/viz/causal_multipoint/` |
| grasp bbox 2×2 교차 | bbox 가 **어떤 물체가 사라질지**, 문장이 **목적지에 무엇처럼 그릴지**를 결정. 충돌 시 키메라 | `outputs/viz/causal_grasp/` |
| 텍스트 제거 | 명사만 빼면 정상(src 12.3 vs 12.2), **지시문까지 빼면 마커째 복사** (src 40.5, 마커 잔존 +106.7) | `outputs/viz/ablate_text/` |

→ **마커가 위치(공간), 문장이 정체성(외형)** 을 담당한다는 역할 분담이 세 실험에서 일관되게 나타남.

---

## 7. 파일 위치

### 체크포인트

```
papers/SWEET/outputs/v5a/<모델>/step-<n>.safetensors        개당 293MB, 500스텝마다
  └ 최고 성능: alpha990_wlm4/step-1485.safetensors
```

### 추론에 필요한 것 (약 32GB)

| 항목 | 크기 | 경로 |
|---|---|---|
| 우리 LoRA | 293MB | 위 |
| FLUX Kontext DiT | 23GB | `/data/hg_models/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors` |
| T5 인코더 | 8.9GB | `/data/hg_models/FLUX.1-dev/text_encoder_2/` |
| CLIP 인코더 | 235MB | `/data/hg_models/FLUX.1-dev/text_encoder/` |
| VAE | 320MB | `/data/hg_models/FLUX.1-dev/ae.safetensors` |
| 토크나이저 | 5MB | `/data/hg_models/FLUX.1-dev/tokenizer{,_2}/` |

추론 VRAM 약 26GB (DiT 상주 + 인코더 CPU offload), 장당 25초(25 스텝).

### 시각 자료 (`papers/SWEET/outputs/viz/`)

```
compare_1234/       6열 비교 시트 25장 — [입력|1|2|3|4|GT], unseen 121 전부
compare_123/        5열 버전 (unseen 25 + seen 13)
test_unseen/        3열 시트 21장 + labels.tsv(사람 태그용)
causal_multipoint/  place point 인과 실험
causal_grasp/       grasp bbox 2×2 인과 실험
ablate_text/        텍스트 ablation
loss_mask/          손실 마스크 히트맵 (마스크 3종 비교 + 소수값 생기는 이유)
train_triplets/     학습 세쌍 예시 12장
train_vanish/       소멸형 예시 25장
metric_regions.png  src/dst 측정 영역 도해
latent_channels.png VAE 16채널 시각화
vae_marker.png      마커가 VAE 통과 후에도 95% 보존됨
lr_mismatch.png     좌우 표현이 시점에 따라 뒤집힌 7쌍
```

### bench 케이스 chaining (`poc/bench/`)

```
run_ours_chain.py                    우리 마커+LoRA 로 케이스 순차 실행
results/<case>/ours_alpha990_wlm4/   생성물 + run_log.json
comparison/row_<case>.png            케이스별 한 줄 [원본|입력|출력|입력|출력...]
```

---

## 8. 데이터 품질 점검 결과

| 항목 | 수치 |
|---|---|
| 좌우 표현이 시점과 어긋남 | 230개 다중시점 쌍 중 **7쌍(전체 0.6%)** — 마커가 위치를 전달하므로 영향 미미 |
| 소멸형(GT 에서 물체가 안 보임) | unseen 121 중 사람 태그 **4개**. 이 4개에서 오히려 성능이 좋음(src 17.3 vs 29.6) |
| 물체 크기 | 화면의 0.08%~4.38%, 중앙값 0.48%. 토큰으로는 중앙값 22개, **7.9%가 4토큰 미만** |
| 팔이 마커 영역을 70%↑ 가림 | unseen 121 중 76장(63%) |

---

## 9. 남은 것

- **seen 평가**: 4번(bal5box)과 2b(wl4long), opaque990 미완
- **원인 분리**: 4번이 손실 식 때문인지 원판 축소 때문인지 (중간 조건 1회 필요)
- **지표 개선**: dst 원판 → src bbox 크기 (`mae_dstbox`), VLM 일치율(Qwen3-VL, 비용 0)
- **bench chain 재실행**: 케이스의 `dst_bbox` 중심이 스텝마다 동일해 2번째 이후가 불리했다.
  목적지 분산 + 케이스 원문 문장 사용(`Place the cup on the towel`)으로 고칠 것
- 데이터 스케일링 곡선(sub250/sub500) 중단 상태, 캐시는 보존됨
