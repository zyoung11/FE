import json
import os
import subprocess
import numpy as np
from PIL import Image

FE_OLD = "./FE-old.exe"
FE_NEW = "C:/zy/Code/Odin/FE/FE.exe"
CFG = os.environ["CAL_CFG"]
DIR = "."
GAMMA = 2.2

def srgb_to_linear(v, gamma):
    a = v / 255.0
    return a / 12.92 if a <= 0.04045 else ((a + 0.055) / 1.055) ** gamma

def std_of(path):
    x = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    return float(x.std())

print(f"{'g':>4} {'u':>7} {'std_old':>8} {'std_new':>8} {'ratio':>7}")
us, rs = [], []
for g in range(8, 256, 16):
    src = f"{DIR}/cal-{g}.png"
    subprocess.run(["magick", "-size", "256x256", f"xc:rgb({g},{g},{g})", "-depth", "8", src], check=True)
    po = f"{DIR}/cal-old-{g}.png"
    pn = f"{DIR}/cal-new-{g}.png"
    subprocess.run([FE_OLD, "--input", src, "--output", po, "--config", CFG], capture_output=True, check=True)
    subprocess.run([FE_NEW, "--input", src, "--output", pn, "--config", CFG], capture_output=True, check=True)
    so = std_of(po)
    sn = std_of(pn)
    u = 1.0 - srgb_to_linear(g, GAMMA)
    us.append(u)
    rs.append(sn / so if so > 0.01 else 1.0)
    print(f"{g:>4} {u:>7.4f} {so:>8.3f} {sn:>8.3f} {rs[-1]:>7.4f}")

json.dump({"u": us, "r": rs}, open(os.environ["CAL_OUT"], "w"))
print()
print("control points (u, r):")
for u, r in zip(us, rs):
    print(f"    {u:.4f}, {r:.4f},")
