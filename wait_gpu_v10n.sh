#!/usr/bin/env bash
# GPU 0-5 중 40GB 이상 비어 있는 장이 2개 생기면 v10n 학습을 시작한다.
# 6,7 은 쓰지 않는다. 이미 학습이 돌고 있으면 아무것도 하지 않는다.
set -uo pipefail
S=/home/hyunjeesong/icra2027/papers/SWEET
NEED=40000        # MiB. direct 모드는 가중치 32.5GB + LoRA 2GB + activation
while true; do
  if pgrep -f "flux/model_training/train.py" >/dev/null; then echo "이미 학습 중"; exit 0; fi
  FREE=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
         | awk -F', ' -v n=$NEED '$1<=5 && ($2-$3)>=n {print $1}' | head -2 | paste -sd,)
  if [ "$(awk -F, '{print NF}' <<<"${FREE:-}")" = "2" ] && [ -n "$FREE" ]; then
    echo "[$(date +%H:%M)] GPU $FREE 확보 → 학습 시작"
    bash "$S/run_v10n.sh" "$FREE"
    exit 0
  fi
  sleep 120
done
