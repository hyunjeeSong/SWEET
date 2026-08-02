# bbox 오버레이 조건부 키프레임 예측 모델 학습 계획

작성 2026-07-22 · 서버 RTX 5090 ×2 (각 32GB, sm_120)

> **문서 내 경로는 모두 `~/icra2027/` 기준의 상대경로다.**
> 이 파일 자체는 `~/icra2027/papers/SWEET/plan.md` 에 있다.
> 예: `poc/data/clips_selected.jsonl` = `~/icra2027/poc/data/clips_selected.jsonl`

---

## 1. 무엇을 왜 하는가

**목표**: FLUX.1-Kontext-dev 를 우리 DROID 데이터로 파인튜닝해서,
**"현재 관측 + bbox 마커 + 텍스트" → "행동 후 키프레임"** 을 예측하게 만든다.

**핵심 실험은 마커 표현 비교(ablation)다.** 베이스·하이퍼파라미터·데이터를 전부 고정하고
**시각 마커만** 바꿔 별도 LoRA 를 학습한다:

| | 마커 | 비고 |
|---|---|---|
| **LoRA-A** | **화살표** (SWEET 방식) | 재현 기준선 |
| **LoRA-B** | **bbox** (grasp/place 박스) | **우리 GT 의 원래 형태** |
| (선택) LoRA-C | bbox + 화살표 | 둘 다 주면 더 나은가 |

우리 GT 는 bbox 가 원본이고 화살표는 파생물이다. bbox 가 더 잘 먹힌다면 변환 손실 없이
GT 를 그대로 쓸 수 있고, 화살표가 낫다면 그 이유(방향성 인코딩)가 밝혀진다.

**왜 이게 프로젝트에 필요한가**: 지금 BAGEL/FreeFine 은 물체만 순간이동시킨다. 로봇 팔도,
도달 가능성도, 물리도 모델링하지 않으므로 **시킨 대로 그릴 뿐이고 검증 신호를 줄 수 없다**.
실제 로봇 롤아웃으로 학습한 모델이라야 "이 plan 이 물리적으로 말이 되는가"가 생성 품질에
반영될 수 있다. 그 전제 자체가 아직 미검증이며(§7), 이 학습이 그것을 검증할 수단이다.

---

## 2. 데이터 현황

### 2-1. 확보됨

`poc/data/clips_selected.jsonl` — 사용자가 직접 선별한 클립 (**재현 불가, 최우선 자산**)

| | 값 |
|---|---|
| 클립 | 514 (고유 에피소드 254, 에피소드당 2쌍) |
| **완전한 클립** | **469** (grasp_bbox·place_bbox 둘 다 보유) |
| **학습쌍 상한** | **938** (클립당 pick 1 + place 1) |
| 좌표계 | **320×180** → 영상 1280×720 기준 **4배 스케일 필요** |
| grasp_bbox 중앙값 | 21×21 (320 기준) → **84×84** (1280 기준) |
| 언어 | `pick_target` / `place_target` — **이미 subtask 단위** |
| 구간 | `pick_start/end`, `place_start/end` 프레임 인덱스 |
| 결측 | grasp_bbox 20, place_bbox 31, place_point 2 |

한 항목 예시:
```json
{"episode_key":"30314_exterior_image_2_left","pair_idx":0,
 "instruction":"transfer the light green cup ... and place the knife inside the cup",
 "video":"/home/dataset/DROID/.../recordings/MP4/right.mp4",
 "pick_start":0,"pick_end":75,"place_start":83,"place_end":109,
 "pick_target":"pick up the light green cup from the countertop",
 "place_target":"place the light green cup inside the light green bowl",
 "grasp_bbox":[116,117,134,141],"place_bbox":[107,103,129,126],
 "place_point":[118,114],"contact_points":[[127,124],[130,126]],
 "origin_shape":[320,180],"contact_frame":...}
```

참고: SWEET 은 2100쌍으로 6K step 학습. 우리는 938쌍이므로 `dataset_repeat 6` 이면 ≈5.6K step.

### 2-2. 미확보 — **Phase 0 의 전부**

**영상 254개 MP4** (`video` 필드가 옛 서버 경로를 가리킴). 이게 없으면 정답 프레임을 못 만든다.

```bash
# 옛 서버: kelly062001@163.152.162.236  (SSH 22 열려 있음, 내부망)
# 키 없으면 먼저:  ssh-copy-id kelly062001@163.152.162.236

# 용량 확인
rsync -avzn --stats kelly062001@163.152.162.236:/home/dataset/DROID/episodes_h5_only/ \
  /data/datasets/DROID/episodes_h5_only/ | tail -5

# 필요한 254개만 받기 (전체가 크면)
#   -> scripts/make_video_filelist.py 로 목록 생성 후 --files-from 사용
nohup rsync -avz --progress --partial \
  kelly062001@163.152.162.236:/home/dataset/DROID/episodes_h5_only/ \
  /data/datasets/DROID/episodes_h5_only/ > /tmp/rsync_droid.log 2>&1 &
```

**받은 뒤 반드시 확인**: 실제 영상 해상도. 1280×720 을 가정 중이며 `origin_shape [320,180]`
대비 정확히 4배여야 스케일이 맞는다. 다르면 스케일 계수를 고쳐야 한다.

`/data` 여유 1.4T.

---

## 3. 마커 설계

### 3-1. bbox 오버레이 (LoRA-B, 본 계획의 주인공)

**설계 원칙**: 화살표와 **정보량을 맞춘다**. 화살표는 (시작, 끝, 색) 3요소를 주므로
bbox 도 (출발 박스, 도착 박스, 색) 을 준다. 안 그러면 마커 비교가 아니라 정보량 비교가 된다.

```
grasp_bbox  → 실선 사각형, 색 = 그리퍼 상태 전이 (아래 표)
place_bbox  → 같은 색 점선(또는 반투명 채움) 사각형
```

| 스텝 | 색 | 의미 |
|---|---|---|
| pick | 초록 `(30,180,20)` | open → closed |
| place | 파랑 `(10,10,160)` | closed → closed |
| (놓기 완료) | 노랑 `(155,155,5)` | closed → open |

**선 두께**: **6px 로 확정** (1280 기준). 구현·검수 완료 — `poc/bench/draw_marker.py`,
미리보기 `poc/bench/viz_marker/`.

가장 작은 물체(case2 계란 **29×34px**)로 확대 검수한 결과:

| 마커 | 결과 |
|---|---|
| arrow (21px) | **화살표가 물체를 완전히 덮는다** — 어느 계란인지 식별 불가 |
| **bbox (6px)** | **테두리라 물체가 그대로 보인다.** 목적지 점선도 명확 |
| both | 화살표가 다시 박스 안을 가림 |

→ **사전 가설**: 우리 데이터는 작은 물체가 많으므로(계란 29×34, 오렌지 35×36, 빨대 두께 23)
**bbox 가 화살표보다 유리할 수 있다.** SWEET 이 화살표를 쓴 것은 캔·봉지처럼 큰 물체
위주였기 때문일 가능성이 있다. 이 가설을 마커 ablation 으로 검증한다.

**알파**: 0.9 (SWEET 실측값과 동일하게)

**RULE_PROMPT 도 bbox 용으로 새로 써야 한다.** SWEET 문장은 화살표 전제다:
```
Follow the box markers in the image: the solid box marks the object to manipulate,
the dashed box marks its target location. Color indicates gripper-state transition:
green = open to closed, blue = closed to closed, yellow = closed to open, red = open to open.
```

### 3-2. 화살표 (LoRA-A, 기준선)

이미 구현·검증 완료: `papers/SWEET/draw_arrow.py`
(SWEET 원본 재현 MAE 0.18~0.62 — 육안 구분 불가)

역산된 규격(1280×720): 샤프트 21px / 화살촉 40px / 화살촉 길이 46px / alpha 0.9

**⚠️ 화살표 방향 주의**: SWEET 의 화살표는 "**그리퍼가 갈 곳**"이지 "물체가 갈 곳"이 아니다.
실측 확인됨 — 물체→목적지 화살표를 주면 모델이 **팔을 목적지로** 옮긴다.
pick 스텝의 화살표 시작점(그리퍼 위치)이 우리 데이터에 없다. 세 가지 안:

- **(a)** RoboInter LMDB 의 `gripper_box` 조인 — 키 형식이 `episode_key` 와 동일해 매칭 가능. 가장 충실
- **(b)** `contact_points` 사용 — 338/514 만 보유
- **(c)** **place 스텝만 학습** (469쌍) — 가장 단순하고, "물체→목적지"가 우리 GT 규약과 일치

→ **(c) 로 시작 권장.** pick 은 확장 단계에서.

---

## 4. 파이프라인

### Phase 1 — 학습쌍 생성

```
클립 하나 →
  pick :  frame[pick_start]  + 마커(grasp_bbox)              →  frame[pick_end]
  place:  frame[place_start] + 마커(grasp_bbox, place_bbox)  →  frame[place_end]
```

산출물은 SWEET 과 동일한 CSV 3컬럼:

```csv
image,kontext_images,prompt
<정답 프레임 경로>,<마커 오버레이 프레임 경로>,"<RULE_PROMPT> <pick_target 또는 place_target> Step N."
```

작성할 스크립트 `poc/bench/build_train_pairs.py`:
1. `clips_selected.jsonl` 로드, 결측 클립 제외
2. ffmpeg(`/usr/bin/ffmpeg`)로 지정 프레임 추출
3. bbox 를 320×180 → 실제 해상도로 스케일
4. 마커 오버레이 (`--marker bbox|arrow|both`)
5. CSV 출력 + 샘플 그리드 이미지(육안 검수용)

**반드시 눈으로 검수할 것.** 좌표 스케일이 틀려도 픽셀 지표로는 안 잡힌다
(이번 세션에 MAE 에 두 번 속았다 — §8 참조).

### Phase 2 — 메모리 실측 ← **현재 위치, 최대 위험**

SWEET 은 **1280×720 풀 해상도**를 **단일 H20 96GB** 로 학습했다. 우리는 32GB ×2.

**단계적 시도** (각 단계에서 1 step 만 돌려 OOM 여부 확인):

1. `--task sft:data_process` 로 **텍스트 임베딩·VAE latent 사전계산**
   → 학습 중 T5(9.5GB)·VAE 미로드. DiT 23GB 만 남음
2. 1 GPU + 1280×720 + batch1 + `--use_gradient_checkpointing`
3. 안 되면 `--fp8_models` (DiT 23GB → ~12GB)
4. 안 되면 `--enable_optimizer_cpu_offload`, `--use_gradient_checkpointing_offload`
5. 안 되면 해상도 축소 (640×360) — **단 SWEET 과 직접 비교가 어긋남을 명시**

**2 GPU 활용**:
- **DDP(기본)** — 모델이 각 GPU 에 **복제**된다. 메모리는 안 줄고 **처리량만 2배**.
  즉 "32GB 에 들어가는가"의 답은 안 바뀐다. 들어간다면 학습 시간은 절반.
- **DeepSpeed ZeRO-3** — 파라미터를 **샤딩**하므로 메모리가 실제로 나뉜다(23GB → ~12GB/GPU).
  DiffSynth 에 연동 코드 있음 (`diffsynth/diffusion/runner.py: initialize_deepspeed_gradient_checkpointing`).
  별도 accelerate config 필요.

→ **순서: 1 GPU 로 되는지 먼저 확인 → 되면 DDP 로 2배 가속 → 안 되면 ZeRO-3.**

### Phase 3 — 학습

```bash
accelerate launch --num_processes 2 --mixed_precision bf16 \
  examples/flux/model_training/train.py \
  --dataset_base_path / \
  --dataset_metadata_path <우리 CSV> \
  --data_file_keys "image,kontext_images" \
  --extra_inputs "kontext_images" \
  --max_pixels 1048576 \
  --dataset_repeat 6 --num_epochs 1 \
  --learning_rate 1e-4 --gradient_accumulation_steps 1 \
  --lora_base_model "dit" --lora_rank 32 \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --align_to_opensource_format --use_gradient_checkpointing \
  --remove_prefix_in_ckpt "pipe.dit." \
  --save_steps 500 --output_path outputs/<marker>_lora
```

938쌍 × repeat 6 ≈ 5.6K step. 스텝당 1~3초 가정 시 **3~5시간** (2 GPU DDP 면 절반).

**LoRA-A / LoRA-B 는 `--dataset_metadata_path` 와 `--output_path` 만 다르게** 두 번 돌린다.

### Phase 4 — 평가

**같은 케이스에 세 LoRA + 원본 SWEET LoRA 를 비교.**

지표 (픽셀 MAE 는 보조로만 — §8):
1. **배경 드리프트** — 변하면 안 되는 영역의 변화량. 순수 생성 열화를 분리해 잰다.
   구현·검증 완료 (SWEET 에서 사이클당 +1.2~1.45 로 선형 누적 측정)
2. **VLM 판정** — "물체가 목표 위치에 있는가" / "원본 자리에 물체가 남아 있는가"
   (Qwen3-VL 서버 사용. 두 번째 질문이 **물체 복제 실패**를 잡는다)
3. **chaining 열화** — teacher forcing 대비 오차 누적 (SWEET 실측 +5.52 @ step3)

---

## 5. 산출물

```
poc/bench/
  build_train_pairs.py        # 클립 -> 프레임 추출 + 마커 오버레이 + CSV
  draw_marker.py              # bbox / arrow / both 렌더러 (draw_arrow.py 확장)
  train_data/
    bbox/{meta.csv, frames/}
    arrow/{meta.csv, frames/}
    preview_grid.png          # 육안 검수용
outputs/
  bbox_lora/step-*.safetensors
  arrow_lora/step-*.safetensors
```

---

## 6. 지금 시점의 미결정 사항

1. **영상 전송 범위** — 전체 vs 필요한 254개만
2. **pick 스텝 포함 여부** — (c) place 만(469쌍) 으로 시작할지, RoboInter `gripper_box` 조인해 938쌍 갈지
3. **bbox 선 두께** — 6px 시작, 작은 물체에서 육안 확인 후 조정
4. **해상도** — 1280×720 고수(비교 정합) vs 축소(메모리 안전)

---

## 7. 이 학습으로 답하려는 것

**전제 검증**: "생성 실패가 검증 신호가 되는가?"

불가능한 plan(닿지 않는 곳, 빈 공간, 벽 너머)을 주면 생성이 무너지는가?
- **무너지면** → 생성 품질이 검증 신호. 프로젝트 전제 성립. BAGEL/FreeFine 을 배제할 근거도 확보
- **멀쩡히 그리면** → 이미지 생성만으로는 검증 불가. 접근 자체를 재고해야 함

`poc/bench/arrow_studio.py` (Flask, 포트 5051) 로 손으로 바로 실험 가능.
학습 전에 원본 SWEET LoRA 로 먼저 해보면 파인튜닝 필요성 판단에 도움이 된다.

---

## 8. 알려진 함정 (이번 세션에서 실제로 겪은 것)

| 함정 | 내용 |
|---|---|
| **픽셀 MAE 를 믿지 말 것** | 화살표 재현이 **위치가 완전히 틀렸는데 MAE 1.5** 였다. 마커가 전체 픽셀의 0.2~1% 라 지표가 반응하지 않는다. **반드시 눈으로 확인**할 것 |
| **에피소드 이름 자르지 말 것** | 34자로 자르니 같은 날짜 7개가 접두어를 공유해 다른 에피소드 좌표를 가져왔다 |
| `--model_paths` 는 **파일 경로** | 디렉토리(`text_encoder_2/`)를 주면 `IsADirectoryError`. T5 는 샤드 2개를 각각 나열해야 함 |
| **좌표계 4배 차이** | bbox 는 320×180, 영상은 1280×720. 스케일 실수해도 그림은 그럴듯해 보인다 |
| GPU 점유 확인 | 다른 프로세스(vLLM 등)가 물고 있으면 2장을 지정해도 CPU 오프로드가 걸려 4배 느려진다. `nvidia-smi` 먼저 |
| sm_120 (Blackwell) | torch 는 **cu128 이상**만 동작. 레포 requirements 가 옛 CUDA 로 덮어쓰면 재고정 필요. 검증은 `is_available()` 말고 **실제 matmul** 로. 자세한 건 `SETUP_5090.md` §6 |

---

## 9. 다음에 할 일 (순서대로)

1. **[블로킹]** 영상 rsync — 이게 없으면 Phase 1 이 안 됨
2. 영상 해상도 확인 → 스케일 계수 확정
3. `build_train_pairs.py` 작성 → **preview 그리드 육안 검수**
4. Phase 2 메모리 실측 (1 step) → 해상도·fp8·GPU 수 확정
5. LoRA-B(bbox) 학습 → LoRA-A(arrow) 학습
6. Phase 4 비교 평가

**영상 없이 지금 할 수 있는 것**: Phase 2 메모리 실측.
SWEET 테스트 150쌍으로 만든 `papers/SWEET/data/prompt_flux/memtest_meta.csv` 가 준비돼 있다.
`--model_paths` 만 고치면 바로 돌릴 수 있다.

---

## 부록 A. 메모리·속도 실측 결과 (2026-07-22 밤)

**결론: 현재 세팅(RTX 5090 32GB)에서 SWEET 학습이 해상도 손실 없이 돌아간다.**

### 실측

| 시도 | 설정 | 결과 |
|---|---|---|
| 1 | 1 GPU, 오프로드 없음, 1280×720 | ❌ **OOM** (30.5GB 소요, 32GB 초과) |
| **2** | **1 GPU + `--enable_model_cpu_offload`, 1280×720** | ✅ **성공** |

**시도 2 상세** (150 스텝 완주, 체크포인트 306MB 생성)

| 항목 | 값 |
|---|---|
| GPU 메모리 | **12.1GB** / 32GB (여유 20GB) |
| GPU 사용률 | 99% |
| **스텝당** | **5.2초** |
| 150 스텝 | 13분 9초 |
| OOM | 없음 |

**논문 설정(6K step) 환산 → 약 8.7시간** (1 GPU 기준)

### 의미

- **해상도를 낮출 필요가 없다.** SWEET 은 1280×720 을 H20 96GB 로 학습했는데,
  우리는 32GB 1장으로 같은 해상도가 돌아간다. `--enable_model_cpu_offload` 하나로 해결.
  → `plan.md` §4 Phase 2 의 fp8·해상도 축소 대비책은 **불필요**했다.
- GPU 여유가 20GB 나 남으므로 오프로드를 줄이거나 batch 를 키울 여지가 있다.
- 2 GPU DDP 는 메모리를 안 줄이는 대신 **처리량 2배** → 약 4.4시간 예상.

### 재현 명령

```bash
# 원본 SWEET 학습 데이터 (2100쌍) CSV 는 아래에 생성해 둠
#   papers/SWEET/data/prompt_flux/training_meta_droid.csv

cd ~/icra2027/papers/SWEET/DiffSynth-Studio
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --num_processes 1 --mixed_precision bf16 \
  examples/flux/model_training/train.py \
  --dataset_base_path / \
  --dataset_metadata_path ../data/prompt_flux/training_meta_droid.csv \
  --data_file_keys "image,kontext_images" --extra_inputs "kontext_images" \
  --max_pixels 1048576 --dataset_repeat 3 --num_epochs 1 \
  --model_paths '["/data/hg_models/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors","/data/hg_models/FLUX.1-dev/text_encoder/model.safetensors",["/data/hg_models/FLUX.1-dev/text_encoder_2/model-00001-of-00002.safetensors","/data/hg_models/FLUX.1-dev/text_encoder_2/model-00002-of-00002.safetensors"],"/data/hg_models/FLUX.1-dev/ae.safetensors"]' \
  --tokenizer_1_path /data/hg_models/FLUX.1-dev/tokenizer \
  --tokenizer_2_path /data/hg_models/FLUX.1-dev/tokenizer_2 \
  --learning_rate 1e-4 --gradient_accumulation_steps 1 \
  --lora_base_model "dit" --lora_rank 32 \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --align_to_opensource_format --use_gradient_checkpointing \
  --enable_model_cpu_offload \
  --remove_prefix_in_ckpt "pipe.dit." --save_steps 500 \
  --output_path outputs/droid_kontext_lora_repro
```

**중요 — `--model_paths` 는 파일 경로여야 한다.** T5 는 샤드 2개이므로 **중첩 리스트**로 준다.
디렉토리(`text_encoder_2/`)를 주면 `IsADirectoryError` 가 난다.

### 확보된 데이터

- `data/RPL_train700/{first_arrow,final,first,meta}` 각 **2100개** (원본 SWEET 학습셋)
- `data/prompt_flux/training_meta_droid.csv` — **2100행**, 누락 0
- test50_seen 과 겹침 **0** (올바른 학습/평가 분리)

### 부록 A-2. 본 학습 가동 및 추가 확인 사항

**본 학습 진행 중** (2026-07-22 밤 시작)

```
papers/SWEET/logs/train_repro.log                 진행 로그
papers/SWEET/outputs/droid_kontext_lora_repro/    500 step 마다 체크포인트
papers/SWEET/run_train_repro.sh                   재실행 스크립트 (NP=2 로 2GPU)
```

원본 SWEET 데이터 2100쌍 × `dataset_repeat 3` = **6300 step**, 논문 설정과 동일.
실측 5.4~5.6 s/it → 완료까지 **약 9시간 45분**.

**추가로 확인된 것**

| 항목 | 내용 |
|---|---|
| `--dataset_num_workers` | 기본값 **0** (메인 프로세스에서 로딩). 어제 실패 원인과는 무관 — 첫 실패는 `--model_paths` 디렉토리 오류, 두 번째는 순수 OOM. GPU 여유가 20GB 남으므로 다음 학습 때 **4 정도로 올리면** 이미지 로딩(1280×720 PNG 2장)이 계산과 겹쳐져 빨라질 여지가 있다 |
| **`conda run` 은 출력을 버퍼링한다** | 9시간 학습에서 로그가 0바이트가 되어 진행·실패를 못 본다. **`accelerate` 를 직접 호출하고 `PYTHONUNBUFFERED=1` + `stdbuf -oL`** 로 실행할 것 |
| 2 GPU 병렬 사용 | GPU0 학습 + GPU1 추론을 동시에 돌려도 스텝당 속도 저하가 거의 없었다 (5.2 → 5.4 s/it) |

---

## 부록 B. 벤치 4-way 비교 완료 (2026-07-22)

`poc/bench/comparison/` — 6 케이스 × 4 모델. **전부 chaining** (`run_log.json` 에 `chaining=True` 기록).

| 모델 | 조건 | 관찰 |
|---|---|---|
| `bagel_put` | 언어 (put) | 목적지에 물체를 **그리지만 원본을 안 지운다**(복제). case0 에서 없던 쓰레기통까지 생성 |
| `bagel_move` | 언어 (move) | **`put` 과 거의 동일** → 복제는 프롬프트 표현 문제가 아니라 모델 특성 |
| `freefine` | 마스크+변환 | 원본은 제거되지만 3~4 스텝 후 **장면이 뭉개진다** |
| `sweet` | 화살표+언어 | **유일하게 로봇 팔 자세를 그린다.** 장면 구조 보존이 가장 좋음. 다만 물체를 용기 "안"에 넣지는 못하고, 멀고 큰 물체(바나나)는 미이동 |

**핵심**: BAGEL/FreeFine 은 물체만 편집하고 **로봇 팔은 원본 그대로**다. SWEET 만 "이 동작을 하려면
팔이 어디로 가야 하는가"를 그린다 — 실제 로봇 롤아웃으로 학습했기 때문. 이것이 §1 에서 말한
"검증 신호를 주려면 물리적으로 근거 있는 생성이어야 한다"는 주장의 시각적 근거다.

### 벤치에 SWEET 을 돌리는 법

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n sweet python poc/bench/run_sweet.py
```

화살표는 GT 의 `src_bbox` 중심 → `dst_bbox` 중심으로 그린다.
**색은 파랑(`closed2closed`, Step 2)** 이 맞다 — 우리 화살표는 "물체 → 목적지"이고
SWEET 규약에서 이는 "물체를 든 채 이동"에 해당한다. 초록(Step 1)은 "그리퍼 → 물체"라
의미가 달라, 실제로 초록을 주면 **팔만 목적지로 이동**한다(실측 확인).
