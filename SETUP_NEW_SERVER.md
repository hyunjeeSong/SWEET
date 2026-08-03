# SWEET 새 서버 세팅 — FLUX-Kontext LoRA 학습 (bbox+point 마커)

새 서버에서 우리 파이프라인(bbox+point overlay LoRA 학습)을 처음부터 세팅하는 가이드.
학습 담당 Claude Code 에게 이 문서 하나 던지면 됨.

> 우리가 실제로 돌린 환경: **RTX 5090 ×2 (Blackwell sm_120), torch 2.11+cu128**.
> `environment.yml` / `requirements.txt` 는 **cu124 로 낡음 → 5090 에선 그대로 쓰면 안 됨**(§2 주의).

---

## 0. 전체 그림 — 3개가 필요하다

이 학습은 **repo 2개 + 모델 + 데이터**가 맞물린다:

| 구성 | 무엇 | 어디서 |
|---|---|---|
| **SWEET** (이 repo) | 학습 스크립트 `train_cached.sh`, CSV `data/prompt_flux/*.csv` | `github.com/hyunjeeSong/SWEET.git` (branch main) |
| **DiffSynth-Studio** | 실제 학습 프레임워크(diffsynth) — **SWEET 에 추적 안 됨, 따로 clone** | `github.com/modelscope/DiffSynth-Studio.git` |
| **icra2027** | 오케스트레이션(`run_v8.sh`, `inject_loss_weight.py`), 데이터 파이프라인, metric | `github.com/hyunjeeSong/icra2027.git` |
| **모델** | FLUX.1-dev(9.5G), FLUX.1-Kontext-dev(23G) — **HF gated** | HuggingFace → `/data/hg_models/` |
| **학습 이미지** | `train_v8/{kontext,final,clean}` PNG (~3,200장) | git 밖, 데이터 디스크 (§5) |

**⚠️ 핵심 함정 2개:**
1. `git clone SWEET` 해도 **DiffSynth-Studio 는 안 따라온다**(중첩된 별도 git, 미추적). 반드시 별도 clone.
2. 학습 이미지(`/data/datasets/DROID/train_v8/...`)는 git 에 없다. CSV 는 절대경로로 이걸 가리키므로, **이미지를 rsync 하거나 icra2027 에서 재빌드**해야 한다(§5).

---

## 1. Clone

```bash
mkdir -p ~/icra2027/papers && cd ~/icra2027/papers
git clone https://github.com/hyunjeeSong/SWEET.git
cd SWEET
git clone https://github.com/modelscope/DiffSynth-Studio.git   # ★ 별도 clone 필수

# 오케스트레이션·데이터 파이프라인 (다른 위치면 경로만 맞추면 됨)
cd ~/icra2027 && git clone https://github.com/hyunjeeSong/icra2027.git .  # 또는 rsync
```

---

## 2. Conda 환경 (RTX 5090 = cu128 필수)

`environment.yml`/`requirements.txt` 는 **torch 2.6+cu124** 로 고정돼 있는데, Blackwell(sm_120)
에선 cu124 커널이 없어 `no kernel image` 로 죽는다. **torch 만 cu128 로 덮어써야 한다.**

```bash
conda create -n sweet python=3.10 -y
conda activate sweet

# 1) torch 를 cu128 로 (5090 필수) — requirements 의 cu124 무시
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2) 나머지 의존성 (requirements 에서 torch 계열만 빼고)
grep -ivE '^torch' requirements.txt | pip install -r /dev/stdin

# 3) DiffSynth-Studio editable 설치 (이게 diffsynth 모듈 제공)
pip install -e DiffSynth-Studio

# 검증 (5090 이면 capability (12,0) + 실제 matmul 성공해야 함)
python -c "import torch,diffsynth; x=torch.randn(64,64,device='cuda'); print('OK', torch.__version__, (x@x).sum().item())"
```

> 24GB GPU(구서버)면 cu121 도 됐지만, **5090 은 cu128 아니면 학습 자체가 안 돈다.**

---

## 3. 모델 다운로드 (HF gated → 토큰 필요)

```bash
export HF_TOKEN=hf_xxx   # ~/.env 에 있으면 source
MODELS=/data/hg_models; mkdir -p $MODELS
hf download black-forest-labs/FLUX.1-Kontext-dev --local-dir $MODELS/FLUX.1-Kontext-dev  # 23G, DiT
hf download black-forest-labs/FLUX.1-dev         --local-dir $MODELS/FLUX.1-dev           # 9.5G, text enc/T5/AE/tokenizer
```

`train_cached.sh` 가 참조하는 정확한 파일(경로 고정 `HG=/data/hg_models`):
- DiT: `FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors`
- text encoder: `FLUX.1-dev/text_encoder/model.safetensors`
- T5: `FLUX.1-dev/text_encoder_2/model-0000{1,2}-of-00002.safetensors`
- VAE: `FLUX.1-dev/ae.safetensors`, tokenizer: `FLUX.1-dev/tokenizer{,_2}`

> 모델 경로가 `/data/hg_models` 가 아니면 `train_cached.sh` 상단 `HG=` 를 고칠 것.

---

## 4. 학습 실행

**2-pass 구조** (`train_cached.sh`): pass1 = VAE/텍스트 인코더로 latent 캐시 생성,
pass2 = DiT LoRA 만 학습(캐시 읽음, 32GB 에 bf16 상주).

### 직접 실행 (최소)
```bash
cd ~/icra2027/papers/SWEET
CSVD=data/prompt_flux
# pass1: 캐시
CUDA_VISIBLE_DEVICES=0 bash train_cached.sh cache $CSVD/v8_train.csv /data/sweet_cache/v8
# 마커-전용 손실 가중(λ=4) 마스크 주입 (icra2027 스크립트)
~/miniforge3/envs/sweet/bin/python ~/icra2027/poc/scripts/sweet_train/inject_loss_weight.py \
  --cache /data/sweet_cache/v8 --csv $CSVD/v8_train.csv --marker-only
# pass2: 학습 (λ=4, 3 epoch, 2 GPU)
env WLOSS_LAMBDA=4 CUDA_VISIBLE_DEVICES=0,1 NP=2 \
  bash train_cached.sh train /data/sweet_cache/v8 outputs/v8/v8_wlm4 3 500
```

### 오케스트레이션 스크립트 (캐시→주입→학습→평가 한 번에)
```bash
# icra2027 에 준비돼 있음 — 버전만 바꿔 쓰면 됨
bash ~/icra2027/poc/scripts/sweet_train/run_v8.sh    # v8 전체 체인
```

- 결과 LoRA: `outputs/<ver>/<ver>_wlm4/step-*.safetensors` (`outputs/` 는 gitignore).
- 평가: `~/icra2027/poc/bench/run_ours_infer.py`(생성) + `eval_ours.py`(픽셀) / `eval_task_success.py`(task 성공률).

---

## 5. 학습 이미지 확보 (git 밖)

CSV 의 `image`/`kontext_images` 는 `/data/datasets/DROID/train_v8/...` 절대경로를 가리킨다.
이 PNG 들은 git 에 없으므로 둘 중 하나:

- **(빠름) rsync**: 기존 서버에서 통째로
  ```bash
  rsync -avz <기존서버>:/data/datasets/DROID/train_v8/ /data/datasets/DROID/train_v8/
  ```
- **(재빌드) icra2027 파이프라인**: DROID 영상+candidate 데이터가 있으면
  `~/icra2027/poc/scripts/data_prep/build_v7_dataset.py --ver v8 ...` 로 재생성 (데이터 파이프라인은
  `~/icra2027/poc/data/annotation/LABELING_HANDOFF.md` 참조).

> CSV 경로가 새 서버와 다르면 CSV 의 절대경로를 `sed` 로 일괄 치환하거나 재빌드할 것.

---

## 6. 버전 맵 (data/prompt_flux)

| 버전 | train | 특징 |
|---|---|---|
| v7 | 2,215 | 저다양성·고물량 (구 스냅샷) |
| v8 | 1,509 | **cap6 재분배 + 다양성↑** (홀드아웃 ILIAD/IRIS/WEIRD 보존) |
| v8cap10 / v8capinf | 1,762 / 2,165 | cap 스윕(다양성 고정·물량만↑) |

- test_unseen(314)은 v7/v8/cap10/capinf 동일 → 직접 비교 가능. test_seen 은 버전마다 다름.
- 데이터 다양성·cap 결정 배경: `~/icra2027/poc/DATA_REDUNDANCY_OVERFIT_REPORT.md`.

---

## 7. 세팅 검증 체크리스트

```bash
conda activate sweet
python -c "import torch,diffsynth; assert torch.cuda.get_device_capability(0)==(12,0); \
  x=torch.randn(64,64,device='cuda'); print('gpu ok', (x@x).sum().item())"   # 5090
ls /data/hg_models/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors           # DiT 있음
ls /data/hg_models/FLUX.1-dev/ae.safetensors                                  # VAE 있음
head -1 data/prompt_flux/v8_train.csv                                          # CSV 있음
ls $(python -c "import csv;print(next(csv.DictReader(open('data/prompt_flux/v8_train.csv')))['image'])")  # 이미지 실제 존재
```
전부 통과하면 §4 학습 바로 가능.
