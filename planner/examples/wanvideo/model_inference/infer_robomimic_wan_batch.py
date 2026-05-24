import torch
import os
import json
from PIL import Image
from tqdm import tqdm
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

# ================= configuration =================
LORA_PATH = "./DiffSynth-Studio/outputs/robomimic_wan_lora/step-3000.safetensors"

JSON_PATH = "./data/prompt_wan/robomimic_video_prompt_list.json"

SMALL_TEST_DIR = "./data/video_test/"

OUTPUT_DIR = "./data/test_results_wan/step-3000/"

RULE_PROMPT = "Follow the semi-transparent motion marker in the image: an arrow indicates movement direction, while a circle indicates an in-place action. Color indicates gripper-state transition: green = open to closed, blue = closed to closed, yellow = closed to open, red = open to open."
# ============================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. load WanVideoPipeline
    print("Loading Wan2.2-TI2V-5B Baseline...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
        ],
    )
    
    # 2. load LoRA checkpoint if exists
    if os.path.exists(LORA_PATH):
        print(f"Loading LoRA Checkpoint: {LORA_PATH}")
        pipe.load_lora(pipe.dit, LORA_PATH, alpha=1.0)
    else:
        print(f"Warning: LoRA not found at {LORA_PATH}, doing zero-shot generation.")

    # 3. read JSON with prompts and corresponding test images
    print(f"Reading {JSON_PATH}...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    valid_data = []

    for item in data:
        filename_img = os.path.basename(item["source_image"])
        test_img_path = os.path.join(SMALL_TEST_DIR, filename_img)
        if os.path.exists(test_img_path):
            valid_data.append((item, test_img_path))
            
    print(f"Total entries in JSON: {len(data)}, Matched in Small Test Set: {len(valid_data)}")

    # 4. batch inference and video generation
    for i, (item, test_img_path) in enumerate(tqdm(valid_data, desc="Wan Video Batch Inference", unit="video")):
        # Extract base name for output video naming
        filename_img = os.path.basename(test_img_path)
        base_name = os.path.splitext(filename_img)[0]
        output_video_path = os.path.join(OUTPUT_DIR, f"{base_name}.mp4")

        if os.path.exists(output_video_path):
            continue

        # Load and resize the condition image (crucial: 1280x736)
        input_image = Image.open(test_img_path).convert("RGB")
        input_image = input_image.resize((1280, 736))

        task_prompt = item["prompt"]
        full_prompt = f"{RULE_PROMPT} {task_prompt}"

        # 5. generate video with WanVideoPipeline
        video = pipe(
            prompt=full_prompt,
            input_image=input_image,
            height=736,
            width=1280,
            num_frames=81,   
            seed=42,         
            tiled=False,     
        )
        
        # 6. save video
        save_video(video, output_video_path, fps=20, quality=7)

    print(f"✅ Small batch video inference completed! Check results at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
