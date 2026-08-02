# 우리 world model 추론 가이드 (bbox+point 마커 LoRA)

**한 줄**: 마커(집을 물체 초록 bbox + 놓을 위치 파란 점)를 그린 이미지와 문장을 넣으면,
**조작 실행 후의 장면**을 생성한다. FLUX.1-Kontext 에 LoRA(rank 32)만 얹은 구조다.

## 체크포인트 두 개 — 어느 걸 쓸까

둘 다 같은 구조(FLUX.1-Kontext + LoRA rank 32, 반투명 마커, 마커-전용 가중 손실 λ=4)이고
**학습 데이터 양만 다르다**. 사용법은 완전히 동일하다.

| | `ours_wlm4_step-1485` | `ours_v6_wlm4_step-2213` |
|---|---|---|
| 학습 데이터 | DROID 990쌍 | DROID **1,475쌍** (+49%) |
| 학습 | 3 epoch, 1,485 스텝 | 3 epoch, 2,213 스텝 |
| **물체 지표** (src↓ / dst↓) | 28.67 / 38.50 | **28.02 / 37.90** |
| 이미지 지표 (PSNR↑ / SSIM↑) | **17.52 / 0.791** | 17.17 / 0.783 |
| 배경 보존 (MAE정지↓) | **6.42** | 7.52 |

*(같은 119장 unseen 에서 측정. copy 기준선 = 17.09 / 0.788 / src 56.99 / dst 40.50)*

- **검증기(verifier) 용도라면 v6** — 물체를 지우고 그리는 능력이 우리가 재고 싶은 것이고,
  거기서 v6 가 낫다. 대신 배경을 조금 더 건드린다.
- **이미지 충실도가 중요하면 1485** — PSNR/SSIM/LPIPS 세 지표 모두에서 copy 기준선을
  이긴 유일한 설정이다. 다만 이 지표군은 화면의 97.5%가 배경이라 태스크를 직접 재지 못한다
  (copy 와 최고 모델의 PSNR 차이가 2.4% 뿐).
- 차이는 전반적으로 작다(src 0.65, dst 0.60). 데이터를 49% 늘린 것치고는 수익이 크지 않다.

아래 설명·예시는 `ours_wlm4_step-1485` 기준이지만, `--lora` 경로만 바꾸면 v6 도 그대로 동작한다.
`smoke_test/` 의 기대 출력도 **1485 기준**이라 v6 로 돌리면 md5 가 다르다(정상).

---

## 1. 필요한 것

LoRA 는 **어댑터**라 단독으로 못 돈다. FLUX 베이스 가중치가 있어야 한다.

| 항목 | 크기 | 받는 곳 |
|---|---|---|
| **우리 LoRA** | 293MB | 이 폴더 |
| FLUX.1-Kontext-dev DiT | 23GB | HF `black-forest-labs/FLUX.1-Kontext-dev` → `flux1-kontext-dev.safetensors` |
| T5 텍스트 인코더 | 8.9GB | HF `black-forest-labs/FLUX.1-dev` → `text_encoder_2/model-0000{1,2}-of-00002.safetensors` |
| CLIP 텍스트 인코더 | 235MB | 같은 repo → `text_encoder/model.safetensors` |
| VAE | 320MB | 같은 repo → `ae.safetensors` |
| 토크나이저 2종 | 5MB | 같은 repo → `tokenizer/`, `tokenizer_2/` |

- **GPU**: 추론 시 약 26GB VRAM (DiT bf16 상주 + 인코더 CPU offload). 32GB 카드 1장이면 충분.
- **속도**: 1280×720, 25 스텝 기준 **장당 약 25초**.

### 환경

```bash
conda create -n sweet python=3.10 -y && conda activate sweet
pip install torch --index-url https://download.pytorch.org/whl/cu124   # sm_120(RTX5090)이면 cu128 nightly
git clone https://github.com/modelscope/DiffSynth-Studio
cd DiffSynth-Studio && pip install -e .
pip install opencv-python pillow
```

> 주의: DiffSynth 의 `environment.yml` 을 그대로 쓰면 RTX 5090(sm_120)에서 죽는다. torch 를 직접 맞출 것.

---

## 2. 마커 그리기 — **학습과 똑같이 해야 한다**

모델은 이 규약으로만 학습됐다. 색·두께·투명도가 달라지면 성능이 떨어진다.

```python
import cv2

def draw_markers(clean_bgr, grasp_bbox, place_point, alpha=0.55):
    """clean_bgr: 실행 전 프레임(BGR). grasp_bbox=[x0,y0,x1,y1], place_point=(x,y) — 픽셀 좌표."""
    over = clean_bgr.copy()
    cv2.rectangle(over, (grasp_bbox[0], grasp_bbox[1]), (grasp_bbox[2], grasp_bbox[3]),
                  (0, 210, 0), 4)                      # 집을 물체: 초록 박스, 두께 4
    cv2.circle(over, place_point, 16, (255, 60, 30), -1)    # 놓을 위치: 파란 점, 반경 16 (BGR)
    cv2.circle(over, place_point, 16, (255, 255, 255), 2)   # 흰 테두리 2
    return cv2.addWeighted(over, alpha, clean_bgr, 1 - alpha, 0)   # 반투명 합성 0.55
```

| 요소 | 값 | 비고 |
|---|---|---|
| grasp 박스 | BGR (0,210,0), 두께 4px | 물체를 감싸는 사각형 |
| place 점 | BGR (255,60,30) 채움, 반경 **16px 고정** | 물체 크기와 무관하게 항상 같음 |
| 점 테두리 | 흰색 두께 2 | |
| 합성 | `addWeighted(over, 0.55, clean, 0.45, 0)` | 불투명보다 성능 우세 |

좌표는 **이미지 픽셀 기준**이다. 다른 해상도의 어노테이션이면 `s = W / origin_width` 로 스케일할 것.

---

## 3. 프롬프트 형식

```
Follow the markers in the image: the green box marks the object to grasp, the blue dot
marks where to place it. Show the scene after the object has been moved to the marked
location. <pick 문장>. <place 문장>.
```

앞의 고정 문구(RULE)가 **트리거**다. 이걸 빼면 모델이 편집을 안 하고 마커째 입력을 복사한다(실측).
뒤의 두 문장은 물체의 **외형**을 정하고, 위치는 전적으로 마커가 정한다.

```
Pick up the light green cup from the countertop. Place the light green cup inside the light green bowl.
Pick up the strawberry from the black box. Place the strawberry on the table.
```

- 동사형으로 쓸 것 (`Pick up ...` / `Place ...`).
- 목적지 물체를 명시하는 게 학습 분포와 맞다 (`on the table`, `in the black bowl`).
  `at the marked location` 같은 추상 표현은 학습에 없었다.
- **left/right 는 쓰지 말 것.** 시점에 따라 뒤집혀 학습에서도 제거했다.

---

## 4. 최소 실행 코드

```python
import sys, torch, cv2
from PIL import Image
sys.path.insert(0, "<DiffSynth-Studio 경로>")
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig

HG   = "<FLUX 가중치 루트>"          # 예: /data/hg_models
LORA = "<이 폴더>/ours_wlm4_step-1485.safetensors"

RES = {"offload_dtype": torch.bfloat16, "offload_device": "cuda:0",
       "onload_dtype": torch.bfloat16,  "onload_device": "cuda:0",
       "preparing_dtype": torch.bfloat16, "preparing_device": "cuda:0",
       "computation_dtype": torch.bfloat16, "computation_device": "cuda:0"}
OFF = {**RES, "offload_device": "cpu", "onload_device": "cpu"}   # 인코더는 CPU 상주

pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda:0",
    model_configs=[
        ModelConfig(path=f"{HG}/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors", **RES),
        ModelConfig(path=f"{HG}/FLUX.1-dev/text_encoder/model.safetensors", **OFF),
        ModelConfig(path=[f"{HG}/FLUX.1-dev/text_encoder_2/model-0000{i}-of-00002.safetensors"
                          for i in (1, 2)], **OFF),
        ModelConfig(path=f"{HG}/FLUX.1-dev/ae.safetensors", **OFF),
    ],
    tokenizer_1_config=ModelConfig(path=f"{HG}/FLUX.1-dev/tokenizer"),
    tokenizer_2_config=ModelConfig(path=f"{HG}/FLUX.1-dev/tokenizer_2"))
pipe.load_lora(pipe.dit, LORA, alpha=1.0)          # ← 여기서 LoRA 결합

RULE = ("Follow the markers in the image: the green box marks the object to grasp, "
        "the blue dot marks where to place it. Show the scene after the object has been "
        "moved to the marked location.")

clean = cv2.imread("scene.png")                                   # 실행 전 프레임
marked = draw_markers(clean, [482, 445, 517, 481], (272, 366))     # 위 함수
img = Image.fromarray(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
w, h = (img.width // 16) * 16, (img.height // 16) * 16             # 16의 배수로 맞출 것

out = pipe(prompt=f"{RULE} Pick up the orange from the counter. Place the orange in the trash can.",
           kontext_images=img.resize((w, h)),
           height=h, width=w, num_inference_steps=25, seed=42)
out.save("after.png")
```

`load_lora(..., alpha=1.0)` 이 결합 지점이다. 별도 머지 파일을 만들 필요 없다
(머지하면 23GB 통짜가 되어 오히려 불편).

---

## 5. 여러 스텝 연속 실행 (chaining)

이전 스텝의 **출력**을 다음 스텝의 입력으로 넣는다.

```python
cur = Image.open("scene.png")
for step in plan_steps:
    marked = draw_markers(np.array(cur)[:, :, ::-1], step["src_bbox"], step["dst_point"])
    cur = pipe(prompt=f"{RULE} {step['sentence']}", kontext_images=..., seed=42, ...)
    cur.save(f"step{step['idx']}.png")
```

관측된 한계:
- 스텝이 늘수록 오차가 누적돼 물체 경계가 뭉개진다(4스텝 이상에서 뚜렷).
- 같은 좌표를 여러 스텝의 목적지로 주면 2번째부터 실패한다. **스텝마다 다른 점**을 줄 것.

---

## 6. 알려진 특성

| 관측 | 내용 |
|---|---|
| **위치 제어** | place point 를 옮기면 물체가 **평균 6.4px 오차**로 따라온다. 점이 다른 물체 위에 떨어지면 그 표면 높이에 맞춰 배치한다 |
| **역할 분담** | 마커 = 위치(공간), 문장 = 정체성(외형). 둘이 충돌하면 "지우는 건 마커, 그리는 건 문장"이라 키메라가 나온다 |
| **텍스트 의존** | 물체 이름을 빼도 기하는 정상. 하지만 RULE 문구까지 빼면 편집이 아예 발동하지 않는다 |
| **잘하는 것** | 물체를 원위치에서 지우기(copy 대비 오차 절반), 용기에 넣어 사라지는 조작 |
| **약한 것** | 목적지에 새로 그리기, 로봇 팔 렌더링(학습에서 제외), 긴 chaining |

---

## 7. 참고 파일

| | |
|---|---|
| 실험 전체 정리 | `papers/SWEET/RESULTS_v5a.md` |
| 학습 방법 | `papers/SWEET/TRAIN_OURS.md` |
| 배치 추론 스크립트 | `poc/bench/run_ours_infer.py` (CSV 목록 → 생성) |
| 케이스 chaining | `poc/bench/run_ours_chain.py` |
| 채점 | `poc/bench/eval_ours.py` (PSNR/SSIM/LPIPS + 영역별 MAE) |

---

## 8. 단독 실행 스크립트

`infer_minimal.py` 하나면 위 내용을 다 담고 있다. 레포 없이 이 파일 + LoRA + FLUX 베이스만 있으면 된다.

```bash
python infer_minimal.py \
  --image scene.png --bbox 482 445 517 481 --point 272 366 \
  --pick  "the orange from the counter" \
  --place "the orange in the trash can" \
  --lora  ours_wlm4_step-1485.safetensors \
  --hg    /path/to/flux_weights \
  --diffsynth /path/to/DiffSynth-Studio \   # pip 로 설치했으면 생략
  --out   after.png --marked-out marked.png
```

마커 렌더링 · 문장 동사형 변환 · LoRA 결합 · 16배수 리사이즈가 모두 들어 있다.
**검증됨**: 동일 입력에서 레포 파이프라인(`run_ours_chain.py`) 출력과 md5 일치.
