import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Base Operation
# -----------------------------
def canny_edges(gray, low=80, high=160):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, low, high)


def detect_circles(gray, dp=1.2, min_dist=40, param1=160, param2=28,
                   min_radius=15, max_radius=0):
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=dp, minDist=min_dist,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius
    )
    if circles is None:
        return []
    return np.squeeze(circles, axis=0).tolist()


def ring_profile(edges, cx, cy, r, ring_width=6, n_theta=720):
    h, w = edges.shape
    thetas = np.linspace(0, 2*np.pi, n_theta, endpoint=False)

    half = max(1, ring_width // 2)
    radii = np.arange(int(r - half), int(r + half) + 1)

    prof = np.zeros(n_theta, dtype=np.float32)

    for rr in radii:
        xs = cx + rr * np.cos(thetas)
        ys = cy + rr * np.sin(thetas)
        xi = np.clip(xs.round().astype(int), 0, w-1)
        yi = np.clip(ys.round().astype(int), 0, h-1)
        prof += edges[yi, xi]

    prof /= (255.0 * len(radii))
    prof -= prof.mean()
    return thetas, prof


def dominant_k_and_score(profile, kmin=1, kmax=60):
    spec = np.fft.rfft(profile)
    amp = np.abs(spec)

    kmax = min(kmax, len(amp)-1)
    band = amp[kmin:kmax+1]

    k = int(np.argmax(band) + kmin)

    # WAVE scoring
    # = dominant peak / range
    score = band.max() / (band.mean() + 1e-6)
    return k, score, amp


# -----------------------------
# visualisation
# -----------------------------
def draw_overlay(img, cx, cy, r, ring_width):
    out = img.copy()
    cv2.circle(out, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
    cv2.circle(out, (int(cx), int(cy)), int(r-ring_width//2), (255, 0, 0), 1)
    cv2.circle(out, (int(cx), int(cy)), int(r+ring_width//2), (255, 0, 0), 1)
    return out


def propose_centers_from_edges(edges, max_centers=40, grid=16):
    """
    エッジ密度が高い領域を粗いグリッドで探し、そのセル中心を候補中心にする。
    円が無くても使える。
    """
    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return []

    h, w = edges.shape
    gx = np.clip((xs / w * grid).astype(int), 0, grid - 1)
    gy = np.clip((ys / h * grid).astype(int), 0, grid - 1)
    idx = gy * grid + gx

    counts = np.bincount(idx, minlength=grid * grid)
    top = np.argsort(counts)[::-1][:max_centers]

    centers = []
    for t in top:
        if counts[t] == 0:
            break
        cy = (t // grid + 0.5) * (h / grid)
        cx = (t % grid + 0.5) * (w / grid)
        centers.append((cx, cy))
    return centers


def wave_search_no_circle(edges, ring_width=6, n_theta=720, kmin=1, kmax=60):
    """
    円検出なしでWAVE（角度方向周期性）を探索する。
    中心候補×半径候補でリングプロファイル→FFT→スコア最大を返す。
    """
    h, w = edges.shape
    centers = propose_centers_from_edges(edges, max_centers=40, grid=16)
    if not centers:
        return None

    # 半径は画像サイズに合わせて広く試す（CIFAR系ならこれでだいたいOK）
    r_list = np.linspace(min(h, w) * 0.08, min(h, w) * 0.48, 22)

    best = None
    for cx, cy in centers:
        for r in r_list:
            _, prof = ring_profile(edges, cx, cy, r, ring_width=ring_width, n_theta=n_theta)
            k, score, amp = dominant_k_and_score(prof, kmin=kmin, kmax=kmax)
            cand = {"cx": cx, "cy": cy, "r": r, "k": k, "score": score, "profile": prof, "amp": amp}
            if best is None or cand["score"] > best["score"]:
                best = cand
    return best



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--outdir", default="wave_out")
    ap.add_argument("--ring_width", type=int, default=3)
    ap.add_argument("--n_theta", type=int, default=60)
    ap.add_argument("--kmin", type=int, default=1)
    ap.add_argument("--kmax", type=int, default=16)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError("image load failed")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = canny_edges(gray)
    cv2.imwrite(str(outdir / "edges.png"), edges)

    # --- B案：円検出なしでWAVE探索 ---
    best = wave_search_no_circle(
        edges,
        ring_width=args.ring_width,
        n_theta=args.n_theta,
        kmin=args.kmin,
        kmax=args.kmax
    )

    if best is None:
        print("No strong WAVE-like angular periodicity found.")
        return

    # save overlay
    overlay = draw_overlay(img, best["cx"], best["cy"], best["r"], args.ring_width)
    cv2.imwrite(str(outdir / "overlay.png"), overlay)

    # save plots
    plt.figure()
    plt.plot(best["profile"])
    plt.title("Ring profile")
    plt.savefig(outdir / "ring_profile.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(best["amp"])
    plt.xlim(0, min(len(best["amp"])-1, 100))
    plt.title(f"Spectrum (k={best['k']})")
    plt.savefig(outdir / "spectrum.png", dpi=200)
    plt.close()

    result = {
        "center": {"x": float(best["cx"]), "y": float(best["cy"])},
        "radius": float(best["r"]),
        "dominant_k": int(best["k"]),
        "wave_score": float(best["score"]),
        "note": "Selected center/radius with strongest angular periodicity (no circle detection)."
    }

    with open(outdir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("DONE")
    print("dominant k:", best["k"])
    print("wave score :", best["score"])


if __name__ == "__main__":
    main()
