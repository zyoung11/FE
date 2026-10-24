import numpy as np
from PIL import Image, ImageFilter

PAIRS = ["input-colorful", "input-colorful1", "input-colorful2", "input-colorful3"]
OLD_DIR = "./old"
NEW_DIR = "./new"

def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)

def blur(img, r=2.0):
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    return np.asarray(pil.filter(ImageFilter.GaussianBlur(r)), dtype=np.float64)

def acorr(res, dx, dy):
    a = res[max(0,dy):res.shape[0]+min(0,dy), max(0,dx):res.shape[1]+min(0,dx)]
    b = res[max(0,-dy):res.shape[0]+min(0,-dy), max(0,-dx):res.shape[1]+min(0,-dx)]
    return float((a * b).mean())

print(f"{'image':<20} {'lowfreq':>9} {'grainRMS':>18} {'ratio':>6} {'lag1':>12} {'lag2':>12} {'xcorr':>7}")
print(f"{'':<20} {'|d blur|':>9} {'old / new':>18} {'n/o':>6} {'old / new':>12} {'old / new':>12} {'':>7}")
for name in PAIRS:
    o = load(f"{OLD_DIR}/{name}.png")
    n = load(f"{NEW_DIR}/{name}.png")
    bo = blur(o)
    bn = blur(n)
    low = np.abs(bo - bn).mean()
    ro = o - bo
    rn = n - bn
    go = ro.mean(axis=2)
    gn = rn.mean(axis=2)
    so = float(np.sqrt((go ** 2).mean()))
    sn = float(np.sqrt((gn ** 2).mean()))
    r1o = acorr(go, 1, 0) / acorr(go, 0, 0)
    r1n = acorr(gn, 1, 0) / acorr(gn, 0, 0)
    r2o = acorr(go, 2, 0) / acorr(go, 0, 0)
    r2n = acorr(gn, 2, 0) / acorr(gn, 0, 0)
    xc = float((go * gn).mean() / (so * sn))
    print(f"{name:<20} {low:>9.3f} {so:>8.2f} /{sn:>8.2f} {sn/so:>6.3f} {r1o:>5.3f}/{r1n:<5.3f} {r2o:>5.3f}/{r2n:<5.3f} {xc:>7.3f}")

print()
print("per-luminance-band grain RMS (old | new | ratio):")
for name in PAIRS[:2]:
    o = load(f"{OLD_DIR}/{name}.png")
    n = load(f"{NEW_DIR}/{name}.png")
    bo = blur(o)
    bn = blur(n)
    go = (o - bo).mean(axis=2)
    gn = (n - bn).mean(axis=2)
    lum = (o + n).mean(axis=2) / 2
    print(f"  {name}:")
    for b in range(8):
        lo, hi = b * 32, (b + 1) * 32
        mask = (lum >= lo) & (lum < hi)
        cnt = int(mask.sum())
        if cnt < 500:
            continue
        so = float(np.sqrt((go[mask] ** 2).mean()))
        sn = float(np.sqrt((gn[mask] ** 2).mean()))
        ratio = sn / so if so > 0.01 else float("nan")
        print(f"    lum {lo:>3}-{hi:>3}: {so:>6.2f} | {sn:>6.2f} | {ratio:>5.3f}  (n={cnt})")

print()
print("per-channel grain RMS ratio (new/old):")
for name in PAIRS:
    o = load(f"{OLD_DIR}/{name}.png")
    n = load(f"{NEW_DIR}/{name}.png")
    ro = (o - blur(o)).reshape(-1, 3)
    rn = (n - blur(n)).reshape(-1, 3)
    ratios = [float(np.sqrt((rn[:, c] ** 2).mean()) / np.sqrt((ro[:, c] ** 2).mean())) for c in range(3)]
    print(f"  {name}: R={ratios[0]:.3f} G={ratios[1]:.3f} B={ratios[2]:.3f}")
