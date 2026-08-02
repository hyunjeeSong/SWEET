# FLUX-Kontext LoRA 학습 — 우리 데이터 (bbox+point overlay)

SWEET(화살표 마커) 대신 **bbox+point 마커**로 FLUX-Kontext LoRA 를 학습한다.
데이터·파이프라인·환경이 모두 준비/검증되어 있고, 아래 명령을 그대로 실행하면 된다.

시각화 예시: `papers/SWEET/train_ours_sample3.png` (BEFORE overlay / prompt / AFTER 3쌍)

---

## 1. 무엇을 하는가

한 pick-and-place clip = 학습 샘플 하나:

| 역할 | 내용 |
|---|---|
| `kontext_images` (입력) | **pick_start 프레임** + 초록 bbox(집을 물체) + 파란 점(놓을 위치) |
| `image` (정답) | **place_end 프레임** (물체 이동 완료) |
| `prompt` | `RULE + "{집을 물체}. {놓을 곳}."` |

즉 "실행 전(overlay) → 실행 후" 지도학습. SWEET 와 목표는 같고 **마커 규약만 다르다**
(SWEET = 화살표 3단계 / 우리 = bbox+point 1장).

---

## 2. 데이터 (이미 생성 완료)

**출처 체인:**
```
poc/data/clips_selected.jsonl (514 clips: video + pick_start/place_end + grasp_bbox + place_point + instruction)
  └─ poc/scripts/data_prep/build_train_pairs.py  (cv2 로 프레임 추출 + 마커 오버레이)
       └─ /data/datasets/DROID/train_ours/{kontext,final}/*.png   (493쌍, 514 중 21개 skip)
            └─ papers/SWEET/data/prompt_flux/training_meta_ours.csv  (493행, 학습 CSV)
```

**학습 CSV**: `papers/SWEET/data/prompt_flux/training_meta_ours.csv`
- 컬럼: `image,kontext_images,prompt` — SWEET 학습 CSV 와 **완전 동일 포맷**
- 493행, 이미지 1280×720 native

**마커 규약** (build_train_pairs.py):
- 초록 사각형 = grasp 대상 물체 bbox
- 파란 원(채움) = place 목표 지점
- 좌표는 **픽셀(1280×720) native** (SWEET 화살표와 달리 0–1000 정규화 아님)
- `RULE = "Follow the markers in the image: the green box marks the object to grasp, the blue dot marks where to place it. Show the scene after the object has been moved to the marked location."`

**마커 색 선택 주의** (직접 학습이라 색은 "약속"일 뿐, 하지만):
- **색 자체엔 마법 없음** — 사전학습 SWEET 는 색이 그리퍼 상태를 인코딩해 필수지만,
  우리는 처음부터 학습하므로 모델이 우리가 준 매핑을 배운다. 중요한 건 아래 3가지.
- **shape 와 중복**: 우리 마커는 색(초록/파랑) + 모양(박스/점)이 **둘 다** grasp/place 를
  구분한다 → 색은 부분적으로 중복 신호. (SWEET 는 모양이 항상 화살표라 색이 유일 구분자.)
- **색-물체 충돌 (실제 위험)**: 초록 박스가 "light green cup" 같은 초록 물체 위에 오면
  대비가 약하다. 파란 점도 파란 물체와 겹칠 수 있다. → 자연물에 드문 색(마젠타/시안)이나
  **대비 외곽선(흰 테두리)** 을 검토. build_train_pairs.py 에서 색만 바꾸면 된다.
- **ablation 여지**: "색이 실제로 도움 되나?" 를 LoRA-A(색 마커) vs LoRA-B(단색/모양만)
  로 비교 가능. 성능 같으면 색 불필요, 다르면 load-bearing.

**데이터 재생성이 필요하면** (마커 스타일·색 바꾸는 등):
```bash
~/miniforge3/envs/FreeFine/bin/python poc/scripts/data_prep/build_train_pairs.py [--limit N]
# cv2 필요 → FreeFine env. 비디오는 /data/datasets/DROID/videos_selected/*.mp4 (254개)
```

> 주의: `poc/sanity_check/overfit_26.jsonl` 은 **planner 학습용**이라 프레임 타이밍이 없어
> FLUX 학습 소스로 못 쓴다. FLUX 소스는 `clips_selected.jsonl`(타이밍 있음)이다.

---

## 3. 학습 파이프라인 (검증 완료)

**임베딩 캐싱 2-pass** 방식. 이유: FLUX-Kontext 전체(DiT 24GB + T5 9.5GB + VAE)= 34GB > 32GB(5090 1장).
인코더 출력(안 변함)을 미리 캐싱해 학습 때 DiT 24GB 만 GPU 상주 → **bf16 정확 + 최속**.
(대안인 CPU offload=느림, fp8=부정확은 안 씀.)

스크립트: **`papers/SWEET/train_cached.sh`** (이미 작성/검증됨)
- `cache` mode: 인코더로 임베딩 계산 → `.pth` 저장 (DiT 는 디스크 오프로드, VRAM ~11GB)
- `train` mode: DiT 만 bf16 상주, 캐시 읽어 LoRA 학습 (VRAM ~30GB/GPU)
- `NP=2` → DDP 2-GPU, `WANDB=1` → wandb 로깅

**실측 성능** (SWEET 2100샘플 기준): cache ~10분, train DDP 3.4s/it.
우리 데이터는 493샘플이라 훨씬 빠르다 (repeat 3 = 1479스텝, DDP ~15분 예상).

---

## 4. 실행 (그대로 복붙)

```bash
cd ~/icra2027/papers/SWEET
CSV=$PWD/data/prompt_flux/training_meta_ours.csv

# --- pass 1: 임베딩 캐싱 (GPU 1장, ~수분) ---
CUDA_VISIBLE_DEVICES=0 bash train_cached.sh cache "$CSV" /data/sweet_cache/ours_493

# --- pass 2: DDP bf16 학습 (tmux 안에서 — WiFi 끊겨도 생존) ---
tmux new-session -d -s ours_train \
  "CUDA_VISIBLE_DEVICES=0,1 NP=2 WANDB=1 WANDB_PROJECT=ours-bbox \
   bash train_cached.sh train /data/sweet_cache/ours_493 \
   $PWD/outputs/ours_bbox_lora 3 > /data/sweet_cache/ours_train.log 2>&1"

# 진행 확인
tail -f /data/sweet_cache/ours_train.log         # 또는 tmux attach -t ours_train
```

- 체크포인트: `outputs/ours_bbox_lora/step-N.safetensors` (500마다, LoRA rank 32, 각 306MB)
- wandb: https://wandb.ai/kelly062001/ours-bbox (loss 는 스텝별로 요동 — smoothing 0.9 로 추세만 보기)
- **머지 불필요**: LoRA 는 base 에 런타임 로드해 쓴다 (`pipe.load_lora`). 머지하면 24GB 로 폭발.

---

## 5. 평가

학습된 LoRA 를 base FLUX-Kontext 에 얹어 생성 → 정답(final) 대비 비교:
```bash
# run_sweet_droid.py 에 --lora 인자로 체크포인트 지정 가능 (poc/bench/)
# 단, 그 스크립트는 DROID unseen(RPL, 화살표) 용이라 우리 bbox 데이터엔
# validation split 을 별도로 떼서 써야 한다 (train_ours 에서 hold-out).
```
> 픽셀 MAE 는 오해를 부른다(마커가 화면의 1% 미만이라 "안 움직인 모델"이 이김).
> **변화영역/정지영역 분해 MAE + LPIPS**, 그리고 **육안 샘플**을 봐야 한다.
> 참고 구현: `poc/bench/eval_sweet_droid.py` (변화영역 분해 포함).

---

## 6. 환경·서버 주의사항 (RTX 5090 / sm_120)

- conda env: `~/miniforge3/envs/sweet` (torch 2.11.0+cu128, Blackwell sm_120 용)
- 모델: `/data/hg_models/FLUX.1-Kontext-dev`, `/data/hg_models/FLUX.1-dev` (인코더/VAE)
- **xformers 금지**: sm_120 에 fp32 백엔드 없음 → `enable_xformers_memory_efficient_attention()` 부르면 죽음. PyTorch SDPA 로 충분.
- wandb: `~/.env` 에 `WANDB_API_KEY` 등 있음. train_cached.sh 가 자동 source.
- **장시간 작업은 반드시 tmux** 에서. Bash 백그라운드/ nohup 은 claude 하니스·vscode 세션의 자식이라 SSH 끊기면 같이 죽는다. tmux 서버는 init 직속이라 생존.
- GPU 2장(각 32GB). 캐시는 `/data`(1.4T 여유)에, 샘플당 ~27MB.

## 7. LoRA 하이퍼파라미터 (SWEET 재현값)

rank 32, alpha 32, base=DiT only, lr 1e-4, grad_accum 1, batch 1(collate 구조상 고정),
gradient checkpointing on. target modules:
`a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp`
> 우리 데이터가 493개로 적으니 과적합 시 rank 16 으로 낮추거나 dataset_repeat 조정 검토.
