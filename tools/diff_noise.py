import numpy as np
from PIL import Image, ImageFilter

def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)

for name in ["input-colorful", "input-colorful1", "input-colorful2", "input-colorful3"]:
    o = load(f"./old/{name}.png")
    n = load(f"./new/{name}.png")
    d = (n - o).mean(axis=2)
    rmse = float(np.sqrt((d ** 2).mean()))
    bo = np.asarray(Image.fromarray(np.clip(o, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)), dtype=np.float64)
    bn = np.asarray(Image.fromarray(np.clip(n, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)), dtype=np.float64)
    low = float(np.abs(bo - bn).mean())
    lum = (o + n).mean(axis=2) / 2
    bands = []
    for b in range(8):
        lo, hi = b * 32, (b + 1) * 32
        mask = (lum >= lo) & (lum < hi)
        if mask.sum() < 2000:
            continue
        bands.append(f"{lo}-{hi}:{float(np.sqrt((d[mask] ** 2).mean())):.2f}")
    print(f"{name}: diffRMSE={rmse:.2f} lowfreq={low:.2f}  " + " ".join(bands))
