import torch
import os
import csv
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from PIL import Image
from tqdm import tqdm

# ================= Configuration =================
# Set SPLIT to "seen" or "unseen" based on which subset you want to run inference on. Make sure the corresponding CSV and output directory are set correctly.
SPLIT = "seen"  

CSV_PATH = f"./data/prompt_flux/test_meta_droid_{SPLIT}.csv"
LORA_PATH = "./DiffSynth-Studio/outputs/droid_kontext_lora/step-6000.safetensors"
OUTPUT_DIR = f"./data/test_results_droid/flux_step6000/{SPLIT}/"
# ============================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading FLUX Pipeline...")
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="black-forest-labs/FLUX.1-Kontext-dev", origin_file_pattern="flux1-kontext-dev.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ],
    )
    if os.path.exists(LORA_PATH):
        print(f"Loading LoRA: {LORA_PATH}")
        pipe.load_lora(pipe.dit, LORA_PATH, alpha=1.0)
    else:
        print(f"Warning: LoRA missing at {LORA_PATH}")

    # Read the test set CSV
    records = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    for item in tqdm(records, desc=f"FLUX Inference ({SPLIT})"):
        first_arrow_path = item["kontext_images"]
        # Use the filename of the target image for saving, making it easy to align metrics later
        output_filepath = os.path.join(OUTPUT_DIR, os.path.basename(item["image"]))

        if os.path.exists(output_filepath):
            continue

        input_img = Image.open(first_arrow_path).convert("RGB")
        w, h = input_img.size
        w_new, h_new = (w // 16) * 16, (h // 16) * 16
        if w != w_new or h != h_new:
            input_img = input_img.resize((w_new, h_new), resample=Image.Resampling.LANCZOS)
        
        image = pipe(
            prompt=item["prompt"],
            kontext_images=input_img,
            height=h_new,
            width=w_new,
            num_inference_steps=25, 
            # guidance_scale=3.5,
            seed=42
        )
        image.save(output_filepath)

    print(f"✅ Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()