"""SWEET 의 화살표 마커를 기하 파라미터로 역산한다.

목적: 우리 벤치 이미지(GT grasp/place bbox 보유)에 같은 스타일의 화살표를 그리기 위해
      SWEET 이 쓰는 색/두께/투명도/화살촉 형태를 실측한다.

first_arrow 와 first 가 화살표 빼고 동일하므로, 두 이미지의 차이로 화살표 픽셀을 정확히
분리할 수 있고, 알파 블렌딩 계수까지 역산 가능하다:
    observed = alpha*color + (1-alpha)*background
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).parent / "data/RPL_test50_seen"
OUT = Path(__file__).parent / "outputs/artifacts"


def arrow_mask(ep, step, thr=12):
    clean = np.asarray(Image.open(BASE / "first" / f"{ep}_step{step}.png").convert("RGB")).astype(np.float64)
    arrow = np.asarray(Image.open(BASE / "first_arrow" / f"{ep}_step{step}.png").convert("RGB")).astype(np.float64)
    m = np.abs(arrow - clean).max(axis=2) > thr
    return clean, arrow, m


def solve_color_alpha(clean, arrow, m):
    """observed = a*C + (1-a)*bg  를 픽셀들에 대해 풀어 색 C 와 알파 a 를 추정.
    코어(가장 변화가 큰) 픽셀은 알파가 1에 가깝다고 보고 색을 잡은 뒤 알파를 회귀."""
    bg, ob = clean[m], arrow[m]
    d = np.abs(ob - bg).max(axis=1)
    core = d > np.percentile(d, 90)          # 경계 안티에일리어싱 제외
    if core.sum() < 5:
        return None, None
    C = np.median(ob[core], axis=0)
    denom = (C - bg)
    ok = np.abs(denom).max(axis=1) > 20
    if ok.sum() < 5:
        return C, None
    with np.errstate(divide="ignore", invalid="ignore"):
        a = ((ob - bg)[ok] / denom[ok]).mean(axis=1)
    a = a[np.isfinite(a) & (a > 0) & (a <= 1.2)]
    return C, (float(np.median(a)) if a.size else None)


def geometry(m):
    """마스크에서 축/끝점/두께를 뽑는다. 화살촉 쪽이 더 두껍다."""
    ys, xs = np.nonzero(m)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    axis = vt[0]                                     # 주축(화살표 방향)
    t = (pts - c) @ axis                             # 축상 좌표
    p_lo, p_hi = c + axis * t.min(), c + axis * t.max()
    # 양 끝 20% 구간의 폭(축 직교 방향 퍼짐)을 비교 → 넓은 쪽이 화살촉
    perp = (pts - c) @ vt[1]
    lo = perp[t < np.percentile(t, 20)].std()
    hi = perp[t > np.percentile(t, 80)].std()
    tail, head = (p_lo, p_hi) if hi > lo else (p_hi, p_lo)
    shaft = perp[(t > np.percentile(t, 35)) & (t < np.percentile(t, 65))]
    return {"tail": tail.round(1).tolist(), "head": head.round(1).tolist(),
            "length": round(float(np.hypot(*(p_hi - p_lo))), 1),
            "shaft_width": round(float(shaft.std() * 4), 1),   # ±2σ ≈ 폭
            "head_width": round(float(max(lo, hi) * 4), 1),
            "n_pixels": int(m.sum())}


if __name__ == "__main__":
    eps = sorted({p.stem.rsplit("_step", 1)[0] for p in (BASE / "first_arrow").glob("*.png")})
    rows = []
    for ep in eps[:12]:
        for s in (1, 2, 3):
            f = BASE / "first_arrow" / f"{ep}_step{s}.png"
            if not f.exists():
                continue
            clean, arrow, m = arrow_mask(ep, s)
            if m.sum() < 50:
                rows.append({"ep": ep, "step": s, "note": "마커 거의 없음", "n_pixels": int(m.sum())})
                continue
            C, a = solve_color_alpha(clean, arrow, m)
            g = geometry(m)
            g.update({"ep": ep, "step": s,
                      "color_rgb": ([int(x) for x in C] if C is not None else None),
                      "alpha": (round(a, 3) if a is not None else None)})
            rows.append(g)

    print(f"{'step':>4} {'색 RGB':>16} {'alpha':>6} {'길이':>7} {'샤프트폭':>8} {'화살촉폭':>8} {'픽셀%':>7}")
    print("-" * 66)
    for r in rows:
        if "note" in r:
            print(f"{r['step']:>4} {'(마커 없음)':>16} {'':>6} {'':>7} {'':>8} {'':>8} {r['n_pixels']:>7}")
            continue
        a = f"{r['alpha']:.2f}" if r["alpha"] is not None else "n/a"
        print(f"{r['step']:>4} {str(r['color_rgb']):>16} {a:>6} {r['length']:>7.1f} "
              f"{r['shaft_width']:>8.1f} {r['head_width']:>8.1f} {r['n_pixels']/(1280*720)*100:>6.2f}%")

    (OUT / "arrow_params.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT/'arrow_params.json'}")
