# 월드모델(FLUX-Kontext) 버전별 생성 이미지 · 결과 인벤토리

작성 2026-08-05. `SWEET/outputs/` 안의 각 버전 infer 결과(생성 이미지) 위치와 성능 요약.
**주의: `outputs/` 는 gitignore 라 git 에 없음 → 다른 서버로는 rsync 로 옮긴다(아래 §3).**

---

## 1. 생성 이미지 현황

| 버전 | 학습수 | 위치 (`SWEET/outputs/`) | png | 크기 |
|---|---:|---|---:|---|
| v5a | 990 | `v5a/alpha990_wlm4/infer_test_unseen` | 122 | 133M |
| v6 | 1,475 | `v6/v6_wlm4/infer_test_unseen` (+coord/coordraw 변형) | 209 | 233M |
| **v7** | 2,215 | `v7/v7_wlm4/infer_test_unseen` + `_seen` | 315 / 171 | 344M / 178M |
| **v8 cap6** | 1,509 | `v8/v8_wlm4/infer_test_unseen` (+`_seen`) | 315 | 347M |
| **v8 cap10** | ~1,900 | `v8cap10/v8cap10_wlm4/infer_test_unseen` (+`_seen`) | 315 | 347M |
| **v8 capinf** | ~2,200 | `v8capinf/v8capinf_wlm4/infer_test_unseen` (+`_seen`) | 315 | 346M |

같은 unseen 재비교: `cross_v7unseen/{v5a_990, v6_1475}` 에 315개씩 재생성본 있음.
- ⚠️ `cross_v7unseen/v7_2215` 는 png 0개(빈 폴더). v7 원본(`v7/v7_wlm4/infer_test_unseen`, 315개)으로 대체 비교 가능.
- 생성 이미지만 합계 **~3.1G**, LoRA 체크포인트까지 포함 시 **~24G**.

---

## 2. 성능 (같은 unseen 314, 픽셀 MAE↓ / lpips↓, 낮을수록 좋음)

| 모델 | 학습수 | act | src | dst | lpips |
|---|---:|---:|---:|---:|---:|
| v5a | 990 | 61.06 | 27.49 | 37.11 | 0.200 |
| v6 | 1,475 | 61.47 | 27.85 | 36.40 | 0.202 |
| **v7** | 2,215 | 60.05 | 26.46 | **33.95** | 0.204 |
| v8 cap6 | 1,509 | 60.58 | 27.61 | 35.88 | 0.211 |
| v8 cap10 | ~1,900 | 60.53 | 26.80 | 34.61 | 0.201 |
| **v8 capinf** | ~2,200 | **58.15** | **26.12** | 36.45 | 0.204 |

- `act`=액션영역 MAE, `src`=집는곳(비움), `dst`=놓는곳(채움).
- 지표 출처: `SWEET/outputs/<버전>/*/infer_test_unseen/metrics.json` (에피소드별 리스트, 평균).

**결론(픽셀 기준)**: 990→2215 로 **양 늘리면 대체로 개선**. cap6(다양성 최우선·양 최소)은 오히려 최하위권,
capinf(양 최다)가 act/src 최고 → **"다양성보다 양"** 이 일관된 그림.
⚠️ 단 이는 **픽셀 MAE**이며, **task 성공률(SRC 비움+DST 채움) 집계는 아직 없음**. 확정하려면 그걸 돌려야 함
(`poc/bench/eval_task_success.py`).

---

## 3. 다른 서버로 보내기 (rsync)

경로가 동일(`/home/hyunjeesong/icra2027/papers/SWEET/outputs`)하면 그대로 보낸다 → metrics/경로 수정 불필요.

**생성 이미지만 (~3.1G):**
```bash
cd /home/hyunjeesong/icra2027/papers/SWEET/outputs
DST=hyunjeesong@163.152.162.188:/home/hyunjeesong/icra2027/papers/SWEET/outputs
rsync -avP --mkpath --relative \
  v5a/alpha990_wlm4/infer_test_unseen \
  v6/v6_wlm4/infer_test_unseen \
  v7/v7_wlm4/infer_test_unseen v7/v7_wlm4/infer_test_seen \
  v8/v8_wlm4/infer_test_unseen v8/v8_wlm4/infer_test_seen \
  v8cap10/v8cap10_wlm4/infer_test_unseen v8cap10/v8cap10_wlm4/infer_test_seen \
  v8capinf/v8capinf_wlm4/infer_test_unseen v8capinf/v8capinf_wlm4/infer_test_seen \
  cross_v7unseen \
  $DST/
```

**outputs 통째로 (LoRA 포함 ~24G):**
```bash
rsync -avP --mkpath /home/hyunjeesong/icra2027/papers/SWEET/outputs/ \
  hyunjeesong@163.152.162.188:/home/hyunjeesong/icra2027/papers/SWEET/outputs/
```
- `-P` 라서 끊겨도 재실행 시 이어받기. 이미 보낸 LoRA 는 skip.
