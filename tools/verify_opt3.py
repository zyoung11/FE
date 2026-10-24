import math
import time
from fe_helpers import RNG, Params, lam_decay, covered, wang, make_probe_plan, lookup, probe_offsets

def gen_cell(fseed, lam, decay, p, cx, cy):
    exp_lam = math.exp(-lam)
    r_scale = 1.0 / math.sqrt(decay)
    gr2_base = p.r2 * r_scale * r_scale
    cs = RNG(wang((((cy & 0xFFFF) << 16) | (cx & 0xFFFF)) + fseed))
    n = cs.poisson(lam, exp_lam)
    out = []
    for _ in range(n):
        ru = cs.rnd01()
        rv = cs.rnd01()
        gr2 = gr2_base
        if p.sig_px > 0:
            g, _ = cs.gauss2()
            rad = min(math.exp(p.mu_ln + p.sigma_ln * g), p.max_r) * r_scale
            gr2 = rad * rad
        out.append((p.ag * (cx + ru), p.ag * (cy + rv), gr2))
    return out

def estimator_cellcache_img(px, py, img, W, H, p, fseed, k_grid):
    plan = make_probe_plan(px, py, p, fseed, k_grid)
    cache = {}
    hit = 0
    for xS, yS in plan:
        ix = int(math.floor(xS))
        iy = int(math.floor(yS))
        u = lookup(img, W, H, ix, iy)
        lam, decay = lam_decay(u, p)
        if lam < 1e-5:
            continue
        max_r_eff = p.max_r / math.sqrt(decay)
        x0 = int(math.floor((xS - max_r_eff) / p.ag))
        x1 = int(math.floor((xS + max_r_eff) / p.ag))
        y0 = int(math.floor((yS - max_r_eff) / p.ag))
        y1 = int(math.floor((yS + max_r_eff) / p.ag))
        ok = False
        for cx in range(x0, x1 + 1):
            if ok:
                break
            for cy in range(y0, y1 + 1):
                key = (ix, iy, cx, cy)
                gl = cache.get(key)
                if gl is None:
                    gl = gen_cell(fseed, lam, decay, p, cx, cy)
                    cache[key] = gl
                if covered(xS, yS, gl):
                    ok = True
                    break
        if ok:
            hit += 1
    return 1.0 - hit / (2 * k_grid * k_grid), len(cache)

print("== Exp1c: per-cell cache vs current, exactness ==")
W, H = 64, 64
img = [0.05 + 0.9 * (0.5 + 0.5 * math.sin(x * 0.7) * math.cos(y * 0.5)) for y in range(H) for x in range(W)]
from fe_helpers import estimator_current_img
for name, pp in [("fine video", Params(0.0088, 0.002, 0.016)), ("big config", Params(0.04, 0.002, 0.008))]:
    diffs = total = gsum = 0
    for t in range(40):
        fseed = wang(777 + t * 13)
        px, py = 2 + (t * 5) % 58, 2 + (t * 9) % 58
        a, ga = estimator_current_img(px, py, img, W, H, pp, fseed, 7)
        b, nc = estimator_cellcache_img(px, py, img, W, H, pp, fseed, 7)
        total += 1
        gsum += ga
        if abs(a - b) > 1e-9:
            diffs += 1
    print(f"  {name}: differing {diffs}/{total}, cell gens current {gsum*25:.0f} vs cache {nc}")

print()
print("== Exp3c: timing current vs per-cell cache ==")
for name, pp in [("fine video", Params(0.0088, 0.002, 0.016)), ("big config", Params(0.04, 0.002, 0.008))]:
    for label, fn in [("current", estimator_current_img), ("cell-cache", estimator_cellcache_img)]:
        t0 = time.time()
        acc = 0.0
        for t in range(16):
            fseed = wang(500 + t)
            acc += fn(2 + (t * 5) % 58, 2 + (t * 9) % 58, img, W, H, pp, fseed, 7)[0]
        dt = time.time() - t0
        print(f"  {name} / {label}: {dt:.2f}s for 16 pixels")
