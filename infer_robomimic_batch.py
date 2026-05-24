import torch
import os
import json
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from PIL import Image
from tqdm import tqdm

# ================= Configuration =================
LORA_PATH = "./DiffSynth-Studio/outputs/robomimic_kontext_lora/step-6000.safetensors"

JSON_PATH = "./data/prompt_flux/robomimic_prompt_list.json"

OUTPUT_DIR = "./data/robomimic_inf_flux_6000step/"

# Rule Prompt
RULE_PROMPT = "Follow the semi-transparent motion marker in the image: an arrow indicates movement direction, while a circle indicates an in-place action. Color indicates gripper-state transition: green = open to closed, blue = closed to closed, yellow = closed to open, red = open to open."
# ============================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. load model pipeline
    print("Loading FLUX pipeline...")
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

    # 2. load LoRA
    if os.path.exists(LORA_PATH):
        print(f"Loading LoRA from {LORA_PATH}")
        pipe.load_lora(pipe.dit, LORA_PATH, alpha=1.0)
    else:
        raise FileNotFoundError(f"Error: LoRA file missing at {LORA_PATH}")

    # 3. parse JSON dataset
    print(f"Reading {JSON_PATH}...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total samples to process: {len(data)}")

    # 4. iterate through all data and generate
    for i, item in enumerate(tqdm(data, desc="Batch Inference", unit="img")):
        first_arrow_path = item["first_arrow_path"]
        final_path = item["final_path"]
        task_prompt = item["prompt"]

        # for convenience of later comparison with ground truth, keep the generated filename consistent with the final true image
        filename = os.path.basename(final_path)
        output_filepath = os.path.join(OUTPUT_DIR, filename)

        # Checkpoint protection: if the image has already been generated, skip to save time
        if os.path.exists(output_filepath):
            continue

        if not os.path.exists(first_arrow_path):
            print(f"\nWarning: Input image not found: {first_arrow_path}")
            continue

        # Load context image and force resize to a multiple of 16 (critically important)
        input_img = Image.open(first_arrow_path).convert("RGB")
        w, h = input_img.size
        # Round down to the nearest multiple of 16
        w_new = (w // 16) * 16
        h_new = (h // 16) * 16
        if w != w_new or h != h_new:
            input_img = input_img.resize((w_new, h_new), resample=Image.Resampling.LANCZOS)
        
        # Concatenate the final prompt
        full_prompt = f"{RULE_PROMPT} {task_prompt}"

        # 5. Inference generation
        image = pipe(
            prompt=full_prompt,
            kontext_images=input_img,
            height=h_new,
            width=w_new,
            num_inference_steps=25, 
            # guidance_scale=3.5,
            seed=42  
        )

        # 6. Save
        image.save(output_filepath)

    print(f"✅ Batch inference completed! All images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()


