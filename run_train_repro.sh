#!/bin/bash
# SWEET 원본 재현 학습 — 논문 설정 (2100쌍 x repeat 3 = 6300 step)
#
# 실측(2026-07-22): 1 GPU + --enable_model_cpu_offload 로 1280x720 원본 해상도 학습 가능.
#   GPU 12.1GB / 스텝당 5.2초  -> 6300 step ≈ 9시간
#
# 실행:  bash run_train_repro.sh [출력경로]
#   NP=2 로 주면 2 GPU DDP (처리량 2배)

OUT=${1:-outputs/droid_kontext_lora_repro}
cd ~/icra2027/papers/SWEET/DiffSynth-Studio

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --num_processes ${NP:-1} --mixed_precision bf16 \
  examples/flux/model_training/train.py \
  --dataset_base_path / \
  --dataset_metadata_path ../data/prompt_flux/training_meta_droid.csv \
  --data_file_keys "image,kontext_images" --extra_inputs "kontext_images" \
  --max_pixels 1048576 \
  --dataset_repeat 3 --num_epochs 1 \
  --model_paths '["/data/hg_models/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors","/data/hg_models/FLUX.1-dev/text_encoder/model.safetensors",["/data/hg_models/FLUX.1-dev/text_encoder_2/model-00001-of-00002.safetensors","/data/hg_models/FLUX.1-dev/text_encoder_2/model-00002-of-00002.safetensors"],"/data/hg_models/FLUX.1-dev/ae.safetensors"]' \
  --tokenizer_1_path /data/hg_models/FLUX.1-dev/tokenizer \
  --tokenizer_2_path /data/hg_models/FLUX.1-dev/tokenizer_2 \
  --learning_rate 1e-4 --gradient_accumulation_steps 1 \
  --lora_base_model "dit" --lora_rank 32 \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --align_to_opensource_format --use_gradient_checkpointing \
  --enable_model_cpu_offload \
  --remove_prefix_in_ckpt "pipe.dit." --save_steps 500 \
  --output_path "$OUT"
