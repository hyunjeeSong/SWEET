"""SWEET 을 teacher forcing 대신 chaining 으로 돌려 오차 누적을 측정한다.

배포된 infer_droid_flux_batch.py 는 매 스텝 GT 기반 first_arrow 를 입력으로 쓴다(teacher forcing).
실제 배포는 step N 의 예측이 step N+1 의 입력이 되므로 오차가 누적된다. 그 차이를 본다.

체이닝 입력 만드는 법:
  step N+1 의 화살표는 GT 프레임 위에 그려져 있다. 화살표만 떼어내려면
  first_arrow(N+1) 과 first(N+1) 의 차이를 쓰면 된다(둘은 화살표 빼고 동일 — 실측 확인).
  그 화살표 픽셀을 "내 예측 이미지" 위에 붙여 다음 입력을 만든다.

실행: SWEET_PLACEMENT=dit_resident conda run -n sweet python run_chaining_test.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig  # noqa: E402

HG = "/data/hg_models"
ROOT = Path(__file__).parent
BASE = ROOT / "data/RPL_test50_seen"
EP = "RPL+success+2023-05-25+Thu_May_25_10_50_40_2023+24013089"
TASK = "Remove the green can from the sink and put it on the counter."
RULE = ("Follow the semi-transparent motion marker in the image: an arrow indicates movement "
        "direction, while a circle indicates an in-place action. Color indicates gripper-state "
        "transition: green = open to closed, blue = closed to closed, yellow = closed to open, "
        "red = open to open.")
LORA = ROOT / "library/planner_library/droid_kontext_lora/step-6000.safetensors"
OUT = ROOT / "outputs/artifacts/chaining"
OUT.mkdir(parents=True, exist_ok=True)

RESIDENT = {"offload_dtype": torch.bfloat16, "offload_device": "cuda:0",
            "onload_dtype": torch.bfloat16, "onload_device": "cuda:0",
            "preparing_dtype": torch.bfloat16, "preparing_device": "cuda:0",
            "computation_dtype": torch.bfloat16, "computation_device": "cuda:0"}
OFFLOAD = {"offload_dtype": torch.bfloat16, "offload_device": "cpu",
           "onload_dtype": torch.bfloat16, "onload_device": "cpu",
           "preparing_dtype": torch.bfloat16, "preparing_device": "cuda:0",
           "computation_dtype": torch.bfloat16, "computation_device": "cuda:0"}


def load(p):
    return Image.open(p).convert("RGB")


def snap16(im):
    w, h = im.size
    w2, h2 = (w // 16) * 16, (h // 16) * 16
    return im.resize((w2, h2), Image.LANCZOS) if (w, h) != (w2, h2) else im


def transplant_arrow(pred_img, step):
    """step 의 화살표만 떼어 pred_img 위에 붙인다."""
    clean = np.asarray(load(BASE / "first" / f"{EP}_step{step}.png")).astype(np.int16)
    arrow = np.asarray(load(BASE / "first_arrow" / f"{EP}_step{step}.png")).astype(np.int16)
    mask = (np.abs(arrow - clean).max(axis=2) > 12)          # 화살표 픽셀
    out = np.asarray(pred_img.resize((clean.shape[1], clean.shape[0]), Image.LANCZOS)).copy()
    out[mask] = arrow[mask].astype(np.uint8)
    return Image.fromarray(out), mask.mean() * 100


print("[load] FLUX Kontext + SWEET LoRA", flush=True)
t0 = time.time()
pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda:0",
    model_configs=[
        ModelConfig(path=f"{HG}/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors", **RESIDENT),
        ModelConfig(path=f"{HG}/FLUX.1-dev/text_encoder/model.safetensors", **OFFLOAD),
        ModelConfig(path=[f"{HG}/FLUX.1-dev/text_encoder_2/model-00001-of-00002.safetensors",
                          f"{HG}/FLUX.1-dev/text_encoder_2/model-00002-of-00002.safetensors"], **OFFLOAD),
        ModelConfig(path=f"{HG}/FLUX.1-dev/ae.safetensors", **OFFLOAD),
    ],
    tokenizer_1_config=ModelConfig(path=f"{HG}/FLUX.1-dev/tokenizer"),
    tokenizer_2_config=ModelConfig(path=f"{HG}/FLUX.1-dev/tokenizer_2"),
)
pipe.load_lora(pipe.dit, str(LORA), alpha=1.0)
print(f"[load] {time.time()-t0:.0f}s", flush=True)


def gen(img, step):
    im = snap16(img)
    return pipe(prompt=f"{RULE} {TASK} Step {step}.", kontext_images=im,
                height=im.size[1], width=im.size[0], num_inference_steps=25, seed=42)


def mae(a, b):
    a = np.asarray(a.resize(b.size)).astype(np.int16)
    return float(np.abs(a - np.asarray(b).astype(np.int16)).mean())


results = []

# ── A) teacher forcing: 매 스텝 GT 기반 입력 ──
print("\n[A] teacher forcing", flush=True)
tf_preds = {}
for s in (1, 2, 3):
    t = time.time()
    p = gen(load(BASE / "first_arrow" / f"{EP}_step{s}.png"), s)
    tf_preds[s] = p
    p.save(OUT / f"tf_step{s}.png")
    gt = load(BASE / "final" / f"{EP}_step{s}.png")
    e = mae(p, gt)
    results.append(("teacher_forcing", s, e))
    print(f"  step{s}: GT와 MAE {e:6.2f}  ({time.time()-t:.0f}s)", flush=True)

# ── B) chaining: 이전 예측 위에 다음 화살표를 옮겨 그림 ──
print("\n[B] chaining", flush=True)
cur = load(BASE / "first_arrow" / f"{EP}_step1.png")   # step1 입력은 동일
for s in (1, 2, 3):
    t = time.time()
    p = gen(cur, s)
    p.save(OUT / f"chain_step{s}.png")
    gt = load(BASE / "final" / f"{EP}_step{s}.png")
    e = mae(p, gt)
    results.append(("chaining", s, e))
    print(f"  step{s}: GT와 MAE {e:6.2f}  ({time.time()-t:.0f}s)", flush=True)
    if s < 3:
        cur, frac = transplant_arrow(p, s + 1)
        cur.save(OUT / f"chain_input_step{s+1}.png")
        print(f"    -> step{s+1} 입력 생성 (화살표 픽셀 {frac:.2f}%)", flush=True)

print("\n" + "=" * 52)
print(f"{'방식':16}{'step':>5}{'GT와 MAE':>12}")
for m, s, e in results:
    print(f"{m:16}{s:>5}{e:>12.2f}")
print("=" * 52)
tf = [e for m, _, e in results if m == "teacher_forcing"]
ch = [e for m, _, e in results if m == "chaining"]
print(f"평균  teacher_forcing {np.mean(tf):.2f}  vs  chaining {np.mean(ch):.2f}")
print(f"step3 차이: {ch[2]-tf[2]:+.2f}  (양수면 chaining 이 더 나쁨 = 오차 누적)")
