#!/bin/bash

TRAIN_SCRIPT="examples/wanvideo/model_training/train.py"

accelerate launch --mixed_precision="bf16" ${TRAIN_SCRIPT} \
  --dataset_base_path / \
  --dataset_metadata_path ./data/prompt_wan/training_meta_droid_wan.csv \
  --data_file_keys "video,input_image" \
  --height 736 \
  --width 1280 \
  --num_frames 81 \
  --dataset_repeat 5 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-4 \
  --gradient_accumulation_steps 1 \
  --num_epochs 1 \
  --save_steps 500 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./outputs/droid_wan_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload