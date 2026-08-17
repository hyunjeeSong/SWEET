#!/usr/bin/env bash
# v10n 학습 실행 (nocache/direct). GPU 두 장을 인자로 받는다.
#
#   bash run_v10n.sh 4,5
#
# **경로는 반드시 절대경로다** — train_cached.sh 가 line 13 에서
# `cd ~/icra2027/papers/SWEET/DiffSynth-Studio` 를 하므로 상대경로를 주면
# CSV 를 못 찾고 죽는다(한 번 그렇게 터졌다). outputs 도 같은 이유로 절대경로.
#
# 설정은 capinf_disc 와 동일하게 맞췄다 — 파이프라인을 고정하고 데이터만 바꾼다.
#   direct(nocache) / NP=2 / 유효배치 2 / LoRA rank32 / lr 1e-4 / bf16 / WLOSS_LAMBDA=4 / repeat 3
# save_steps 만 500→250 (1686 step 이라 500 이면 체크포인트가 3개뿐. 학습에는 영향 없음)
set -euo pipefail

GPUS=${1:?"사용할 GPU 를 주세요 (예: 4,5)"}
S=/home/hyunjeesong/icra2027/papers/SWEET
CSV=$S/data/prompt_flux/v10n_train_wl.csv     # loss_weight 컬럼 포함 (build_loss_masks_v10.py 산출)
OUT=$S/outputs/v10n_wlm4
REPEAT=${2:-3}
SAVE=${3:-250}

[ -f "$CSV" ] || { echo "ERROR: $CSV 없음"; exit 1; }
NP=$(awk -F, '{print NF}' <<<"$GPUS")

mkdir -p "$S/logs"
LOG=$S/logs/v10n_wlm4.log
echo "GPU=$GPUS NP=$NP  csv=$(basename "$CSV") repeat=$REPEAT save=$SAVE"
echo "예상 step = $(( ($(wc -l < "$CSV") - 1) * REPEAT / NP ))"
echo "로그 → $LOG"

cd "$S"
CUDA_VISIBLE_DEVICES=$GPUS NP=$NP WLOSS_LAMBDA=4 WANDB=1 \
  nohup bash train_cached.sh direct "$CSV" "$OUT" "$REPEAT" "$SAVE" > "$LOG" 2>&1 &
echo "PID $!"
