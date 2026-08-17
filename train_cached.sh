#!/bin/bash
# 임베딩 캐싱 2-pass 학습 (bf16 정확, DiT 만 GPU 상주 → CPU offload/fp8 불필요).
#
#   pass 1 (data_process): 인코더(T5/CLIP/VAE)로 임베딩을 미리 계산해 .pth 로 저장.
#                          모델 4개 다 로드하지만 no_grad 라 VRAM 여유. 샘플당 1회.
#   pass 2 (train):        DiT 만 로드(24GB, 32GB 에 bf16 상주). 캐시를 읽어 DiT 만 학습.
#                          인코더가 없으니 CPU 스트리밍/ fp8 근사 둘 다 불필요.
#
# 사용:
#   bash train_cached.sh cache <META_CSV> <CACHE_DIR>      # pass 1
#   bash train_cached.sh train <CACHE_DIR> <OUT_DIR> [STEPS]  # pass 2
set -e
cd ~/icra2027/papers/SWEET/DiffSynth-Studio
# conda env 위치는 서버마다 다르다(miniforge3 / .conda). 있는 쪽을 고르고, 없으면 즉시 실패한다.
# 9시간짜리 학습이 엉뚱한 python 으로 조용히 도는 것보다 여기서 죽는 게 낫다.
ENVDIR=${SWEET_ENV:-}
if [ -z "$ENVDIR" ]; then
  for d in "$HOME/.conda/envs/sweet" "$HOME/miniforge3/envs/sweet"; do
    [ -x "$d/bin/accelerate" ] && { ENVDIR=$d; break; }
  done
fi
[ -n "$ENVDIR" ] || { echo "ERROR: sweet env 를 못 찾음. SWEET_ENV=<envdir> 로 지정할 것"; exit 1; }
export PATH=$ENVDIR/bin:$PATH
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# wandb: .env 의 WANDB_API_KEY 등을 불러온다. WANDB=1 일 때만 로깅 켠다.
# 프로젝트는 항상 하나(sweet)로 모으고, 구분은 **run 이름**으로 한다 (출력 폴더명 = run 이름).
[ -f "$HOME/.env" ] && { set -a; . "$HOME/.env"; set +a; }
WANDB_FLAGS=""
[ "${WANDB:-0}" = "1" ] && WANDB_FLAGS="--enable_wandb_log --wandb_project ${WANDB_PROJECT:-sweet}"

HG=/data1/hg_models
DIT=$HG/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors
ENC1=$HG/FLUX.1-dev/text_encoder/model.safetensors
T5A=$HG/FLUX.1-dev/text_encoder_2/model-00001-of-00002.safetensors
T5B=$HG/FLUX.1-dev/text_encoder_2/model-00002-of-00002.safetensors
AE=$HG/FLUX.1-dev/ae.safetensors
TOK1=$HG/FLUX.1-dev/tokenizer
TOK2=$HG/FLUX.1-dev/tokenizer_2
# NP=2 로 주면 DDP(2 GPU). data_process 는 항상 1 프로세스(인코딩은 병렬 이득 적고 캐시 1회면 끝).
NP=${NP:-1}
MG=""; [ "$NP" -gt 1 ] && MG="--multi_gpu"
ACC="$ENVDIR/bin/accelerate launch $MG --num_processes $NP --mixed_precision bf16"
ACC1="$ENVDIR/bin/accelerate launch --num_processes 1 --mixed_precision bf16"
LT="a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp"

MODE=$1
case "$MODE" in
direct)
  # 캐시 없이 CSV 에서 바로 학습한다. 이 서버 실측으로 인코딩 오버헤드는 스텝당 +4.4% 뿐인데
  # 캐시 생성에 105분(1509샘플)이 들어 3 epoch 한 번 돌리는 데엔 캐시가 오히려 손해다.
  #   --task "sft"        접미사 없는 태스크만 인코더 unit 을 살린다("sft:train" 은 캐시 전용)
  #   loss_weight         CSV 컬럼(마스크 PNG). image 와 같은 오퍼레이터를 타 정렬이 맞고,
  #                       parse_extra_inputs 가 실어 나르면 loss.py 가 pop 해서 가중에 쓴다.
  META=$2; OUT=$3; REPEAT=${4:-3}; SAVE=${5:-500}
  NSAMP=$(( $(wc -l < "$META") - 1 ))
  STEPS=$(( NSAMP * REPEAT / NP ))
  export WANDB_NAME="${WANDB_NAME:-$(basename "$OUT")_n${NSAMP}x${REPEAT}_${STEPS}st_${NP}gpu}"
  export WANDB_NOTES="csv=$(basename "$META") samples=$NSAMP repeat=$REPEAT steps=$STEPS \
gpus=$NP lambda=${WLOSS_LAMBDA:-0} nocache lora_rank=32 lr=1e-4 bf16 out=$OUT"
  export WANDB_TAGS="lora32,nocache,${NP}gpu"
  $ACC examples/flux/model_training/train.py \
    --task "sft" \
    --dataset_base_path / --dataset_metadata_path "$META" \
    --data_file_keys "image,kontext_images,loss_weight" \
    --extra_inputs "kontext_images,loss_weight" \
    --max_pixels 1048576 --dataset_repeat $REPEAT --num_epochs 1 \
    --model_paths "[\"$DIT\",\"$ENC1\",[\"$T5A\",\"$T5B\"],\"$AE\"]" \
    --tokenizer_1_path "$TOK1" --tokenizer_2_path "$TOK2" \
    --learning_rate 1e-4 --gradient_accumulation_steps 1 \
    --lora_base_model "dit" --lora_target_modules "$LT" --lora_rank 32 \
    --align_to_opensource_format --use_gradient_checkpointing \
    --remove_prefix_in_ckpt "pipe.dit." --save_steps $SAVE \
    $WANDB_FLAGS \
    --output_path "$OUT"
  ;;
cache)
  META=$2; CACHE=$3
  # pass 1: 인코더(CLIP/T5/VAE)는 GPU, DiT 는 디스크 오프로드로 올린다.
  # DiT 가중치는 안 쓰지만(가중치 없는 prepare_image_ids 헬퍼만 호출) 객체는 있어야 한다.
  # 디스크 오프로드라 VRAM 은 인코더 ~10GB 만 차지 → OOM 없음. data_process 는 no_grad·1회.
  $ACC1 examples/flux/model_training/train.py \
    --task "sft:data_process" \
    --dataset_base_path / --dataset_metadata_path "$META" \
    --data_file_keys "image,kontext_images" --extra_inputs "kontext_images" \
    --max_pixels 1048576 --dataset_repeat 1 --num_epochs 1 \
    --model_paths "[\"$DIT\",\"$ENC1\",[\"$T5A\",\"$T5B\"],\"$AE\"]" \
    --offload_models "$DIT" \
    --tokenizer_1_path "$TOK1" --tokenizer_2_path "$TOK2" \
    --lora_base_model "dit" --lora_target_modules "$LT" --lora_rank 32 \
    --use_gradient_checkpointing \
    --output_path "$CACHE"
  ;;
train)
  CACHE=$2; OUT=$3; REPEAT=${4:-3}; SAVE=${5:-500}
  # pass 2: DiT 만 로드. metadata_path 를 안 주면 load_from_cache=True 로 캐시를 읽는다.
  # repeat 3 → 캐시 2100개 × 3 = 6300 스텝 (원본과 동일). 5번째 인자로 save_steps 조정.
  # wandb (프로젝트는 sweet 하나). run 이름에 무슨 데이터로 몇 스텝 돌렸는지가 보이게 만든다.
  #   예) C_full485_n485x6_1455st_2gpu   ← 485샘플을 6번 반복, DDP 2장이라 1455스텝
  NSAMP=$(find "$CACHE" -name '*.pth' 2>/dev/null | wc -l)
  STEPS=$(( NSAMP > 0 ? NSAMP * REPEAT / NP : 0 ))
  export WANDB_NAME="${WANDB_NAME:-$(basename "$OUT")_n${NSAMP}x${REPEAT}_${STEPS}st_${NP}gpu}"
  export WANDB_NOTES="data=$(basename "$CACHE") samples=$NSAMP repeat=$REPEAT steps=$STEPS \
gpus=$NP batch=$NP lora_rank=32 lr=1e-4 bf16 save_every=$SAVE out=$OUT"
  export WANDB_TAGS="lora32,$(basename "$CACHE"),${NP}gpu"
  $ACC examples/flux/model_training/train.py \
    --task "sft:train" \
    --dataset_base_path "$CACHE" \
    --data_file_keys "image,kontext_images" --extra_inputs "kontext_images" \
    --max_pixels 1048576 --dataset_repeat $REPEAT --num_epochs 1 \
    --model_paths "[\"$DIT\"]" \
    --tokenizer_1_path "$TOK1" --tokenizer_2_path "$TOK2" \
    --learning_rate 1e-4 --gradient_accumulation_steps 1 \
    --lora_base_model "dit" --lora_target_modules "$LT" --lora_rank 32 \
    --align_to_opensource_format --use_gradient_checkpointing \
    --remove_prefix_in_ckpt "pipe.dit." --save_steps $SAVE \
    ${VAL_CACHE:+--val_cache "$VAL_CACHE" --val_every ${VAL_EVERY:-100}} \
    $WANDB_FLAGS \
    --output_path "$OUT"
  ;;
*) echo "usage: bash train_cached.sh {cache|train} ..."; exit 1;;
esac
