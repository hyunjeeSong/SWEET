#!/bin/bash
# SWEET (DROID FLUX planner) 실행에 필요한 자산만 선별 다운로드.
# 스크립트가 origin_file_pattern 으로 특정 파일만 쓰므로 FLUX 레포 전체(각 57.9GB)는 받지 않는다.
#
# 실행: bash download_assets.sh

# hf CLI: ~/.local/bin 의 huggingface_hub 는 사라졌으므로 conda env 것을 쓴다.
HF=$HOME/miniforge3/envs/vllm-qwen/bin/hf
[ -x "$HF" ] || { echo "hf CLI 없음: $HF"; exit 1; }
[ -f "$HOME/.env" ] && { set -a; . "$HOME/.env"; set +a; }
export HF_HOME=/data/hg_models/.hf_cache

M=/data/hg_models
S=$HOME/icra2027/papers/SWEET
mkdir -p "$M/FLUX.1-Kontext-dev" "$M/FLUX.1-dev" "$S/library" "$S/data"

log(){ echo "[$(date '+%H:%M:%S')] $*"; }
FAIL=()

get(){ # get <repo> <dest> <type> <include...>
  local repo=$1 dest=$2 typ=$3; shift 3
  local inc=(); for p in "$@"; do inc+=(--include "$p"); done
  log "받는 중: $repo  ${*}"
  if "$HF" download "$repo" ${typ:+--repo-type "$typ"} --local-dir "$dest" "${inc[@]}"; then
    log "  완료: $repo"
  else
    FAIL+=("$repo"); log "  실패: $repo"
  fi
}

# 1) FLUX.1-Kontext-dev — 단일 통합 체크포인트만 (23.8GB)
get black-forest-labs/FLUX.1-Kontext-dev "$M/FLUX.1-Kontext-dev" "" "flux1-kontext-dev.safetensors"

# 2) FLUX.1-dev — 텍스트 인코더 2종 + VAE 만 (~9.9GB). transformer 는 Kontext 것을 쓰므로 제외.
get black-forest-labs/FLUX.1-dev "$M/FLUX.1-dev" "" \
    "text_encoder/model.safetensors" "text_encoder_2/*" "ae.safetensors" \
    "tokenizer/*" "tokenizer_2/*"

# 3) SWEET LoRA + DROID 테스트 데이터 (공개 데이터셋)
get VEHwang/SWEET_data "$S/library" dataset "planner_library.zip"
get VEHwang/SWEET_data "$S/data"    dataset "DROID_labeled_for_flux_test50_seen.zip" "prompt_flux.zip"

log "=== 용량 ==="
du -sh "$M/FLUX.1-Kontext-dev" "$M/FLUX.1-dev" "$S/library" "$S/data" 2>/dev/null

if [ ${#FAIL[@]} -eq 0 ]; then log "전부 성공"; else log "실패: ${FAIL[*]}"; fi
