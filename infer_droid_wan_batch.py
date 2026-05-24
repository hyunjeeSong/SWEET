import torch
import os
import csv
from PIL import Image
from tqdm import tqdm
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

# ================= configuration =================
SPLIT = "seen"  # "seen" or "unseen"

CSV_PATH = f"./data/prompt_wan/test_meta_droid_wan_{SPLIT}.csv"
LORA_PATH = "./DiffSynth-Studio/outputs/droid_wan_lora/step-3000.safetensors"
OUTPUT_DIR = f"./data/test_results_droid/wan_step3000/{SPLIT}/"
# ============================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading Wan2.2 Pipeline...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
        ],
    )
    if os.path.exists(LORA_PATH):
        print(f"Loading LoRA: {LORA_PATH}")
        pipe.load_lora(pipe.dit, LORA_PATH, alpha=1.0)
    else:
        print(f"Warning: LoRA missing at {LORA_PATH}")

    records = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    for item in tqdm(records, desc=f"Wan Inference ({SPLIT})"):
        # Assuming the CSV has columns: "video", "prompt", "input_image"
        input_image_path = item["input_image"]
        base_name = os.path.splitext(os.path.basename(item["video"]))[0]
        output_video_path = os.path.join(OUTPUT_DIR, f"{base_name}.mp4")

        if os.path.exists(output_video_path):
            continue

        # Wan2.2 expects 1280x736 input images
        input_image = Image.open(input_image_path).convert("RGB").resize((1280, 736))

        video = pipe(
            prompt=item["prompt"],
            input_image=input_image,
            height=736,
            width=1280,
            num_frames=81,         
            seed=42,         
            tiled=False,     
        )
        
        save_video(video, output_video_path, fps=20, quality=7)

    print(f"✅ Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()