#!/bin/bash

accelerate launch examples/flux/model_training/train.py \
  --dataset_base_path / \
  --dataset_metadata_path ./data/prompt_flux/training_meta_droid.csv \
  --data_file_keys "image,kontext_images" \
  --max_pixels 1048576 \
  --dataset_repeat 3 \
  --num_epochs 1 \
  --model_id_with_origin_paths "black-forest-labs/FLUX.1-Kontext-dev:flux1-kontext-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors" \
  --learning_rate 1e-4 \
  --gradient_accumulation_steps 1 \
  --save_steps 1000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./outputs/droid_kontext_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" \
  --lora_rank 32 \
  --align_to_opensource_format \
  --extra_inputs "kontext_images" \
  --use_gradient_checkpointing
