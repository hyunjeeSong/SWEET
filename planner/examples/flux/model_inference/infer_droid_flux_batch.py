import torch
import os
import csv
# [수정] 원본은 flux_image_new 를 import 하지만 현재 DiffSynth-Studio(2.0.18) 에는 그 모듈이
# 없다(flux_image / flux2_image / wan_video 만 존재). 학습 스크립트(train.py)도 flux_image 를
# 쓰고, from_pretrained / load_lora / __call__(kontext_images=...) 시그니처가 동일해 그대로 대체 가능.
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from PIL import Image
from tqdm import tqdm

# ================= Configuration =================
# Set SPLIT to "seen" or "unseen" based on which subset you want to run inference on. Make sure the corresponding CSV and output directory are set correctly.
SPLIT = "seen"  

CSV_PATH = f"./data/prompt_flux/test_meta_droid_{SPLIT}.csv"
# [수정] 원본은 직접 학습해 DiffSynth-Studio/outputs/ 에 생긴 체크포인트를 가정한다.
# 우리는 배포된 planner_library.zip 을 풀어 쓰므로 그 경로를 가리킨다.
LORA_PATH = "./library/planner_library/droid_kontext_lora/step-6000.safetensors"
OUTPUT_DIR = f"./data/test_results_droid/flux_step6000/{SPLIT}/"

# [수정] 모델은 미리 /data/hg_models 에 받아뒀다(스킬 규칙: 체크포인트는 /data/hg_models).
# ModelConfig(path=...) 로 로컬 파일을 직접 지정해 재다운로드를 막는다.
HG = "/data/hg_models"

# 배치 방식: SWEET_PLACEMENT=dit_resident (기본, 빠름) | cpu_offload (느리지만 확실)
#
#   dit_resident : DiT(23GB)만 GPU 에 상주시키고, 텍스트 인코더(T5 9.5GB)+VAE 는 CPU 에 둔다.
#       느린 원인은 DiT 였다 — denoising 25 스텝마다 23GB 를 CPU 에서 스트리밍했기 때문.
#       인코더/VAE 는 이미지당 한 번만 쓰므로 오프로드해도 손해가 거의 없다.
#       모든 계산이 cuda:0 에서 일어나 교차 GPU 문제도 없다.
#
#   ※ GPU 2장에 모델을 쪼개는 방식(DiT->cuda:0, 인코더->cuda:1)은 실패한다.
#     DiffSynth 가 모델 간 입력 텐서를 자동으로 옮기지 않아
#     "Expected all tensors to be on the same device" 가 난다.
PLACEMENT = os.environ.get("SWEET_PLACEMENT", "dit_resident")

_RESIDENT = {"offload_dtype": torch.bfloat16, "offload_device": "cuda:0",
             "onload_dtype": torch.bfloat16, "onload_device": "cuda:0",
             "preparing_dtype": torch.bfloat16, "preparing_device": "cuda:0",
             "computation_dtype": torch.bfloat16, "computation_device": "cuda:0"}
_OFFLOAD = {"offload_dtype": torch.bfloat16, "offload_device": "cpu",
            "onload_dtype": torch.bfloat16, "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16, "preparing_device": "cuda:0",
            "computation_dtype": torch.bfloat16, "computation_device": "cuda:0"}

if PLACEMENT == "dit_resident":
    DIT_CFG, ENC_CFG, VRAM_LIMIT = _RESIDENT, _OFFLOAD, None
else:
    DIT_CFG = ENC_CFG = _OFFLOAD
    VRAM_LIMIT = 0
PIPE_DEVICE = "cuda:0"
# ============================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading FLUX Pipeline...")
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=PIPE_DEVICE,
        # [수정] 32GB 1장에는 안 들어간다 (Kontext DiT ~24GB + T5-XXL ~9.5GB + VAE > 32GB).
        # DiffSynth 저VRAM 패턴: 가중치는 CPU 에 두고(onload_device="cpu") 계산할 때만
        # GPU 로 보낸다(computation_device="cuda"). offload_device 만 주면 여전히 GPU 에
        # 상주해 T5 로딩에서 OOM 난다.
        # 참고: DiffSynth-Studio/examples/z_image/model_inference_low_vram/Z-Image-i2L.py
        model_configs=[
            ModelConfig(path=f"{HG}/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors", **DIT_CFG),
            ModelConfig(path=f"{HG}/FLUX.1-dev/text_encoder/model.safetensors", **ENC_CFG),
            ModelConfig(path=[f"{HG}/FLUX.1-dev/text_encoder_2/model-00001-of-00002.safetensors",
                              f"{HG}/FLUX.1-dev/text_encoder_2/model-00002-of-00002.safetensors"],
                        **ENC_CFG),
            ModelConfig(path=f"{HG}/FLUX.1-dev/ae.safetensors", **ENC_CFG),
        ],
        tokenizer_1_config=ModelConfig(path=f"{HG}/FLUX.1-dev/tokenizer"),
        tokenizer_2_config=ModelConfig(path=f"{HG}/FLUX.1-dev/tokenizer_2"),
        vram_limit=VRAM_LIMIT,
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
