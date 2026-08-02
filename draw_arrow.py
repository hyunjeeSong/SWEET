"""SWEET 스타일 화살표를 그린다 (analyze_arrow.py 로 역산한 파라미터 기반).

측정된 사양 (1280x720 기준, 36개 샘플):
    샤프트 폭   ≈ 21 px   (20.7~21.2, 거의 고정)
    화살촉 폭   ≈ 40 px   (샤프트의 약 2배)
    alpha       ≈ 0.9     (RULE_PROMPT 는 "semi-transparent" 라 하지만 실측은 거의 불투명)
    색 (스텝 고정, 그리퍼 상태 전이를 인코딩)
        step1 초록  (30,180,20)  open  -> closed  (잡기)
        step2 파랑  (10, 10,160) closed-> closed  (운반)
        step3 노랑  (155,155,5)  closed-> open    (놓기)

우리 벤치에 쓰려면: 시작=grasp bbox 중심, 끝=place bbox 중심.
"""
import numpy as np
from PIL import Image, ImageDraw

# 그리퍼 상태 전이 -> 색 (RULE_PROMPT 규약)
COLORS = {
    "open2closed":   (30, 180, 20),    # green  = 잡기
    "closed2closed": (10, 10, 160),    # blue   = 운반
    "closed2open":   (155, 155, 5),    # yellow = 놓기
    "open2open":     (200, 30, 30),    # red    = 접촉 없음
}
STEP_DEFAULT = {1: "open2closed", 2: "closed2closed", 3: "closed2open"}

SHAFT_W = 21.0
HEAD_W = 40.0
HEAD_LEN = 46.0
ALPHA = 0.9


def draw_arrow(img, start, end, color, shaft_w=SHAFT_W, head_w=HEAD_W,
               head_len=HEAD_LEN, alpha=ALPHA, scale_to=(1280, 720)):
    """start -> end 로 화살표를 그린다. 좌표는 scale_to 기준이며 img 크기에 맞춰 환산한다."""
    img = img.convert("RGB")
    sx, sy = img.width / scale_to[0], img.height / scale_to[1]
    s = np.array([start[0] * sx, start[1] * sy], float)
    e = np.array([end[0] * sx, end[1] * sy], float)
    k = (sx + sy) / 2
    sw, hw, hl = shaft_w * k, head_w * k, head_len * k

    v = e - s
    L = np.hypot(*v)
    if L < 1:
        return img
    u = v / L
    n = np.array([-u[1], u[0]])
    hl = min(hl, L * 0.5)
    j = e - u * hl                       # 샤프트와 화살촉의 경계

    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a = int(round(alpha * 255))
    d.polygon([tuple(s + n * sw / 2), tuple(j + n * sw / 2),
               tuple(j - n * sw / 2), tuple(s - n * sw / 2)], fill=color + (a,))
    d.polygon([tuple(e), tuple(j + n * hw / 2), tuple(j - n * hw / 2)], fill=color + (a,))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def draw_for_step(img, start, end, step):
    return draw_arrow(img, start, end, COLORS[STEP_DEFAULT[step]])


if __name__ == "__main__":
    # 검증: 원본 first 에 역산 파라미터로 다시 그려 first_arrow 와 비교
    import json
    from pathlib import Path

    R = Path(__file__).parent
    BASE = R / "data/RPL_test50_seen"
    EP = "RPL+success+2023-05-25+Thu_May_25_10_50_40_2023+24013089"
    params = {(p["step"]): p for p in json.loads((R / "outputs/artifacts/arrow_params.json").read_text())
              if p.get("ep") and EP.startswith(p["ep"]) and p.get("color_rgb")}

    tiles = []
    for s in (1, 2, 3):
        clean = Image.open(BASE / "first" / f"{EP}_step{s}.png").convert("RGB")
        orig = Image.open(BASE / "first_arrow" / f"{EP}_step{s}.png").convert("RGB")
        p = params.get(s)
        if p is None:
            continue
        redraw = draw_arrow(clean, p["tail"], p["head"], tuple(p["color_rgb"]),
                            shaft_w=p["shaft_width"], head_w=p["head_width"], alpha=p["alpha"] or ALPHA)
        mae = float(np.abs(np.asarray(redraw, float) - np.asarray(orig, float)).mean())
        print(f"step{s}: 재현 vs 원본 MAE {mae:.2f}   색{p['color_rgb']} 폭{p['shaft_width']}")
        tiles.append((f"step{s} 원본", orig))
        tiles.append((f"step{s} 재현 (MAE {mae:.1f})", redraw))

    CW, CH, PAD = 320, 180, 6
    cv = Image.new("RGB", (2 * (CW + PAD) + PAD, len(tiles) // 2 * (CH + PAD + 16) + PAD), (250, 250, 250))
    dr = ImageDraw.Draw(cv)
    for i, (name, im) in enumerate(tiles):
        x = PAD + (i % 2) * (CW + PAD)
        y = PAD + (i // 2) * (CH + PAD + 16)
        cv.paste(im.resize((CW, CH), Image.LANCZOS), (x, y + 16))
        dr.text((x, y + 2), name, fill=(0, 0, 0))
    out = R / "outputs/artifacts/arrow_redraw_check.png"
    cv.save(out)
    print("저장:", out)
