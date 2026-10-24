import math
from fe_helpers import RNG, Params, lam_decay, gen_grains, wang, make_probe_plan, lookup

def instrumented(px, py, img, W, H, p, fseed, k_grid, cache_mode):
    plan = make_probe_plan(px, py, p, fseed, k_grid)
    cache = {}
    visits = gens = misses = scanned = covered_cnt = 0
    win_sizes = []
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
        win_sizes.append(((x1 - x0 + 1), (y1 - y0 + 1)))
        hit = False
        scanned += 1
        for cx in range(x0, x1 + 1):
            if hit:
                break
            for cy in range(y0, y1 + 1):
                visits += 1
                key = (ix, iy, cx, cy)
                if cache_mode:
                    gl = cache.get(key)
                    if gl is None:
                        gl = gen_grains(fseed, lam, decay, p, cx, cx, cy, cy)
                        cache[key] = gl
                        misses += 1
                else:
                    gl = gen_grains(fseed, lam, decay, p, cx, cx, cy, cy)
                    gens += 1
                for g in gl:
                    dx = xS - g[0]
                    dy = yS - g[1]
                    if dx * dx + dy * dy < g[2]:
                        hit = True
                        break
        if hit:
            covered_cnt += 1
    if cache_mode:
        gens = misses
    return visits, gens, len(cache), covered_cnt, scanned, win_sizes

W, H = 64, 64
for name, pp in [("fine video", Params(0.0088, 0.002, 0.016)), ("big config", Params(0.04, 0.002, 0.008))]:
    for u in [0.15, 0.5, 0.85]:
        img = [u] * (W * H)
        tv = tg = tu = tc = ts = 0
        wsum = [0, 0]
        wmax = [0, 0]
        n_px = 60
        for t in range(n_px):
            fseed = wang(777 + t * 13)
            px, py = 2 + (t * 5) % 58, 2 + (t * 9) % 58
            v, g, uu, c, s, ws = instrumented(px, py, img, W, H, pp, fseed, 7, True)
            tv += v
            tg += g
            tu += uu
            tc += c
            ts += s
            for a, b in ws:
                wsum[0] += a
                wsum[1] += b
                wmax[0] = max(wmax[0], a)
                wmax[1] = max(wmax[1], b)
        nws = max(1, sum(1 for t in range(n_px) for _ in range(1)))
        print(f"{name} u={u}: visits/px={tv/n_px:.0f} gens/px={tg/n_px:.1f} unique/px={tu/n_px:.1f} "
              f"covered={tc/n_px:.1f}/98 scanned={ts/n_px:.0f} win_avg={wsum[0]/max(1,ts):.1f}x{wsum[1]/max(1,ts):.1f} win_max={wmax[0]}x{wmax[1]}")
