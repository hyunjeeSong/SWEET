#!/usr/bin/env python3
"""단독 실행용 최소 추론 스크립트 — 우리 world model (bbox+point 마커 LoRA).

레포 없이 이 파일 하나 + LoRA + FLUX 베이스 가중치만 있으면 돈다.

  python infer_minimal.py \
      --image scene.png --bbox 482 445 517 481 --point 272 366 \
      --pick "the orange from the counter" --place "the orange in the trash can" \
      --lora ours_wlm4_step-1485.safetensors --hg /path/to/flux_weights --out after.png

--hg 아래 구조(HuggingFace 에서 받은 그대로):
  FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors
  FLUX.1-dev/{text_encoder/model.safetensors,
              text_encoder_2/model-0000{1,2}-of-00002.safetensors,
              ae.safetensors, tokenizer/, tokenizer_2/}

VRAM 약 26GB, 1280x720 · 25스텝 기준 장당 25초.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

RULE = ("Follow the markers in the image: the green box marks the object to grasp, "
        "the blue dot marks where to place it. Show the scene after the object has been "
        "moved to the marked location.")

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True, help="실행 전 프레임")
ap.add_argument("--bbox", nargs=4, type=int, required=True, metavar=("X0", "Y0", "X1", "Y1"),
                help="집을 물체의 bbox (픽셀)")
ap.add_argument("--point", nargs=2, type=int, required=True, metavar=("X", "Y"),
                help="놓을 위치 (픽셀)")
ap.add_argument("--pick", default="", help='예: "the orange from the counter"')
ap.add_argument("--place", default="", help='예: "the orange in the trash can"')
ap.add_argument("--lora", required=True)
ap.add_argument("--hg", required=True, help="FLUX 가중치 루트")
ap.add_argument("--diffsynth", default=None, help="DiffSynth-Studio 경로 (pip 설치했으면 생략)")
ap.add_argument("--out", default="after.png")
ap.add_argument("--marked-out", default=None, help="마커 그린 입력도 저장")
ap.add_argument("--alpha", type=float, default=0.55)
ap.add_argument("--steps", type=int, default=25)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

if args.diffsynth:
    sys.path.insert(0, args.diffsynth)
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig  # noqa: E402


def draw_markers(clean_bgr, bbox, point, alpha=0.55):
    """학습과 동일한 마커 규약. 색·두께·투명도를 바꾸면 성능이 떨어진다."""
    over = clean_bgr.copy()
    cv2.rectangle(over, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 210, 0), 4)
    cv2.circle(over, tuple(point), 16, (255, 60, 30), -1)
    cv2.circle(over, tuple(point), 16, (255, 255, 255), 2)
    return cv2.addWeighted(over, alpha, clean_bgr, 1 - alpha, 0)


def sentence(pick, place):
    """동사형으로 통일. 이미 동사로 시작하면 그대로 둔다."""
    import re
    VH = re.compile(r"(?i)^(pick|place|put|move|remove|take|grab|stack|hang|transfer|"
                    r"insert|drop|set|push|lift)\b")
    out = []
    for s, verb in ((pick, "Pick up"), (place, "Place")):
        s = (s or "").strip()
        if not s:
            continue
        out.append(s[0].upper() + s[1:] if VH.match(s) else f"{verb} {s}")
    s = ". ".join(out)
    return (s + "." if s and not s.endswith(".") else s)


HG = args.hg
RES = {"offload_dtype": torch.bfloat16, "offload_device": args.device,
       "onload_dtype": torch.bfloat16, "onload_device": args.device,
       "preparing_dtype": torch.bfloat16, "preparing_device": args.device,
       "computation_dtype": torch.bfloat16, "computation_device": args.device}
OFF = {**RES, "offload_device": "cpu", "onload_device": "cpu"}   # 인코더는 CPU 상주

print("[1/3] FLUX 로딩 (약 1분)", flush=True)
pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device=args.device,
    model_configs=[
        ModelConfig(path=f"{HG}/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors", **RES),
        ModelConfig(path=f"{HG}/FLUX.1-dev/text_encoder/model.safetensors", **OFF),
        ModelConfig(path=[f"{HG}/FLUX.1-dev/text_encoder_2/model-0000{i}-of-00002.safetensors"
                          for i in (1, 2)], **OFF),
        ModelConfig(path=f"{HG}/FLUX.1-dev/ae.safetensors", **OFF),
    ],
    tokenizer_1_config=ModelConfig(path=f"{HG}/FLUX.1-dev/tokenizer"),
    tokenizer_2_config=ModelConfig(path=f"{HG}/FLUX.1-dev/tokenizer_2"))

print("[2/3] LoRA 결합", flush=True)
pipe.load_lora(pipe.dit, args.lora, alpha=1.0)

clean = cv2.imread(args.image)
if clean is None:
    sys.exit(f"이미지를 못 읽음: {args.image}")
marked = draw_markers(clean, args.bbox, args.point, args.alpha)
if args.marked_out:
    cv2.imwrite(args.marked_out, marked)
img = Image.fromarray(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
w, h = (img.width // 16) * 16, (img.height // 16) * 16      # 16의 배수 필수

prompt = f"{RULE} {sentence(args.pick, args.place)}".strip()
print(f"[3/3] 생성  |  {prompt[len(RULE):].strip() or '(문장 없음)'}", flush=True)
out = pipe(prompt=prompt, kontext_images=img.resize((w, h)),
           height=h, width=w, num_inference_steps=args.steps, seed=args.seed)
out.save(args.out)
print(f"저장: {args.out}", flush=True)
