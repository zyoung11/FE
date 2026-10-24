import math
import time

def wang(s):
    s = (s ^ 61) & 0xFFFFFFFF
    s = (s ^ (s >> 16)) & 0xFFFFFFFF
    s = (s * 9) & 0xFFFFFFFF
    s = (s ^ (s >> 4)) & 0xFFFFFFFF
    s = (s * 668265261) & 0xFFFFFFFF
    s = (s ^ (s >> 15)) & 0xFFFFFFFF
    return s

def xor_shift(s):
    s = (s ^ ((s << 13) & 0xFFFFFFFF)) & 0xFFFFFFFF
    s = (s ^ (s >> 17)) & 0xFFFFFFFF
    s = (s ^ ((s << 5) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return s

class RNG:
    __slots__ = ("s",)
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF
    def rnd01(self):
        self.s = xor_shift(self.s)
        return self.s / 4294967295.0
    def gauss2(self):
        r1 = self.rnd01()
        r2 = self.rnd01()
        r = math.sqrt(-2.0 * math.log(r1 + 1e-12))
        return r * math.cos(2 * math.pi * r2), r * math.sin(2 * math.pi * r2)
    def poisson(self, lam, exp_lam):
        u = self.rnd01()
        x = 0
        prod = exp_lam
        summ = exp_lam
        lim = int(math.floor(lam + 6.0 * math.sqrt(lam))) + 1
        while u > summ and x < lim:
            x += 1
            prod *= lam / x
            summ += prod
        return x

class Params:
    def __init__(self, r_px, sig_px, sig_f_px):
        self.r_px, self.sig_px, self.sig_f_px = r_px, sig_px, sig_f_px
        self.r2 = r_px * r_px
        if sig_px == 0:
            self.sigma_ln = self.mu_ln = 0.0
            self.max_r = r_px
        else:
            s2 = math.log((sig_px / r_px) ** 2 + 1)
            self.sigma_ln = math.sqrt(s2)
            self.mu_ln = math.log(r_px) - 0.5 * s2
            self.max_r = math.exp(self.mu_ln + 3.0902 * self.sigma_ln)
        self.ag = 1.0 / math.ceil(1.0 / r_px) if r_px > 0 else 1.0
        self.lambda_fac = self.ag * self.ag / (math.pi * (r_px * r_px + sig_px * sig_px)) if r_px > 0 else 0.0

def lam_decay(u, p):
    lam = -p.lambda_fac * math.log(1.0 - u)
    decay = 1.0 - 0.7 * min(1.0, max(0.0, (u - 0.5) / 0.45))
    return lam * decay, decay

def gen_grains(fseed, lam, decay, p, x0, x1, y0, y1):
    exp_lam = math.exp(-lam)
    r_scale = 1.0 / math.sqrt(decay)
    gr2_base = p.r2 * r_scale * r_scale
    out = []
    for cx in range(x0, x1 + 1):
        for cy in range(y0, y1 + 1):
            cs = RNG(wang((((cy & 0xFFFF) << 16) | (cx & 0xFFFF)) + fseed))
            n = cs.poisson(lam, exp_lam)
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

def covered(xg, yg, grains):
    for gx, gy, gr2 in grains:
        dx = xg - gx
        dy = yg - gy
        if dx * dx + dy * dy < gr2:
            return True
    return False

def probe_offsets(st, k_grid):
    offs = []
    for s in range(2 * k_grid * k_grid):
        pair = s >> 1
        kx = pair % k_grid
        ky = pair // k_grid
        u1 = (kx + st.rnd01()) / k_grid
        u2 = (ky + st.rnd01()) / k_grid
        rr = math.sqrt(-2.0 * math.log(u1 + 1e-12))
        ph = 2 * math.pi * u2
        sgn = -1.0 if (s & 1) else 1.0
        offs.append((rr * math.cos(ph) * sgn, rr * math.sin(ph) * sgn))
    return offs

def lookup(img, W, H, ix, iy):
    return img[max(0, min(H - 1, iy)) * W + max(0, min(W - 1, ix))]

def make_probe_plan(px, py, p, fseed, k_grid):
    st = RNG(wang((py * 73856093) ^ (px * 19349663) ^ fseed))
    plan = []
    for ox, oy in probe_offsets(st, k_grid):
        plan.append((px + p.sig_f_px * ox, py + p.sig_f_px * oy))
    return plan

def estimator_current_img(px, py, img, W, H, p, fseed, k_grid):
    hit = 0
    gens = 0
    for xS, yS in make_probe_plan(px, py, p, fseed, k_grid):
        u = lookup(img, W, H, int(math.floor(xS)), int(math.floor(yS)))
        lam, decay = lam_decay(u, p)
        if lam < 1e-5:
            continue
        max_r_eff = p.max_r / math.sqrt(decay)
        x0 = int(math.floor((xS - max_r_eff) / p.ag))
        x1 = int(math.floor((xS + max_r_eff) / p.ag))
        y0 = int(math.floor((yS - max_r_eff) / p.ag))
        y1 = int(math.floor((yS + max_r_eff) / p.ag))
        grains = gen_grains(fseed, lam, decay, p, x0, x1, y0, y1)
        gens += 1
        if covered(xS, yS, grains):
            hit += 1
    return 1.0 - hit / (2 * k_grid * k_grid), gens

def estimator_4bucket_img(px, py, img, W, H, p, fseed, k_grid):
    plan = make_probe_plan(px, py, p, fseed, k_grid)
    reach = p.sig_f_px * 6.5 + p.max_r * 2.0
    x0 = int(math.floor((px - reach) / p.ag))
    x1 = int(math.floor((px + reach) / p.ag))
    y0 = int(math.floor((py - reach) / p.ag))
    y1 = int(math.floor((py + reach) / p.ag))
    buckets = {}
    for xS, yS in plan:
        ix = int(math.floor(xS))
        iy = int(math.floor(yS))
        if (ix, iy) in buckets:
            continue
        u = lookup(img, W, H, ix, iy)
        lam, decay = lam_decay(u, p)
        buckets[(ix, iy)] = None if lam < 1e-5 else gen_grains(fseed, lam, decay, p, x0, x1, y0, y1)
    hit = 0
    for xS, yS in plan:
        ix = int(math.floor(xS))
        iy = int(math.floor(yS))
        u = lookup(img, W, H, ix, iy)
        lam, _ = lam_decay(u, p)
        if lam < 1e-5:
            continue
        if covered(xS, yS, buckets[(ix, iy)]):
            hit += 1
    return 1.0 - hit / (2 * k_grid * k_grid), len(buckets)

print("== Exp1b: 4-bucket cache vs current on varying-u image ==")
W, H = 64, 64
img = [0.05 + 0.9 * (0.5 + 0.5 * math.sin(x * 0.7) * math.cos(y * 0.5)) for y in range(H) for x in range(W)]
for name, pp in [("fine video", Params(0.0088, 0.002, 0.016)), ("big config", Params(0.04, 0.002, 0.008))]:
    diffs = total = gsum = 0
    for t in range(40):
        fseed = wang(777 + t * 13)
        px, py = 2 + (t * 5) % 58, 2 + (t * 9) % 58
        a, ga = estimator_current_img(px, py, img, W, H, pp, fseed, 7)
        b, nb = estimator_4bucket_img(px, py, img, W, H, pp, fseed, 7)
        total += 1
        gsum += ga
        if abs(a - b) > 1e-9:
            diffs += 1
    print(f"  {name}: differing {diffs}/{total}, grain gens current {gsum/total:.1f} vs cache {nb}")

print()
print("== Exp2b: paired estimator-noise vs field std (fine video params) ==")
pp = Params(0.0088, 0.002, 0.016)
print(f"{'u':>6} {'std_field':>10} {'std_e98':>9} {'std_e50':>9} {'std_e24':>9} {'e98/field':>10} {'e24/field':>10}")
for u in [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]:
    dref_all, de98, de50, de24 = [], [], [], []
    for t in range(400):
        fseed = wang(31337 + t * 29)
        px, py = t * 11 % 700, t * 17 % 700
        lam, decay = lam_decay(u, pp)
        if lam < 1e-5:
            continue
        max_r_eff = pp.max_r / math.sqrt(decay)
        reach = pp.sig_f_px * 7.5 + max_r_eff
        x0 = int(math.floor((px - reach) / pp.ag))
        x1 = int(math.floor((px + reach) / pp.ag))
        y0 = int(math.floor((py - reach) / pp.ag))
        y1 = int(math.floor((py + reach) / pp.ag))
        grains = gen_grains(fseed, lam, decay, pp, x0, x1, y0, y1)
        st_ref = RNG(0x9e3779b9 ^ wang(fseed))
        hit = 0
        for _ in range(2000):
            g1, g2 = st_ref.gauss2()
            if covered(px + pp.sig_f_px * g1, py + pp.sig_f_px * g2, grains):
                hit += 1
        dref = 1.0 - hit / 2000.0
        plan7 = make_probe_plan(px, py, pp, fseed, 7)
        plan5 = make_probe_plan(px, py, pp, fseed, 5)
        plan3 = make_probe_plan(px, py, pp, fseed, 3)
        def est(plan):
            h = 0
            for xS, yS in plan:
                if covered(xS, yS, grains):
                    h += 1
            return 1.0 - h / len(plan)
        dref_all.append(dref)
        de98.append(est(plan7) - dref)
        de50.append(est(plan5) - dref)
        de24.append(est(plan3) - dref)
    def sd(a):
        m = sum(a) / len(a)
        return math.sqrt(sum((x - m) ** 2 for x in a) / len(a))
    sf = sd(dref_all)
    e98, e50, e24 = sd(de98), sd(de50), sd(de24)
    print(f"{u:>6.2f} {sf:>10.5f} {e98:>9.5f} {e50:>9.5f} {e24:>9.5f} {e98/sf:>10.3f} {e24/sf:>10.3f}")

print()
print("== Exp3b: timing current vs 4-bucket cache ==")
pp = Params(0.0088, 0.002, 0.016)
for label, fn in [("current per-probe rescan", estimator_current_img), ("4-bucket cache", estimator_4bucket_img)]:
    t0 = time.time()
    acc = 0.0
    for t in range(16):
        fseed = wang(500 + t)
        acc += fn(2 + (t * 5) % 58, 2 + (t * 9) % 58, img, W, H, pp, fseed, 7)[0]
    dt = time.time() - t0
    print(f"  {label}: {dt:.2f}s for 16 pixels")
