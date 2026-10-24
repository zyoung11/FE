#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

typedef struct { unsigned s; } RNG;
static inline unsigned wang(unsigned s) {
    s = (s ^ 61u); s ^= s >> 16; s *= 9u; s ^= s >> 4; s *= 668265261u; s ^= s >> 15; return s;
}
static inline unsigned xsh(RNG* r) { unsigned s = r->s; s ^= s << 13; s ^= s >> 17; s ^= s << 5; r->s = s; return s; }
static inline float rnd01(RNG* r) { return (float)xsh(r) * (1.0f / 4294967295.0f); }
static inline void rnd_gauss(RNG* r, float* g1, float* g2) {
    float r1 = rnd01(r), r2 = rnd01(r);
    float rr = sqrtf(-2.0f * logf(r1 + 1e-12f));
    *g1 = rr * cosf(6.2831853072f * r2);
    *g2 = rr * sinf(6.2831853072f * r2);
}
static inline unsigned rnd_poisson(RNG* r, float lam, float exp_lam) {
    float u = rnd01(r);
    float prod = exp_lam, summ = exp_lam;
    unsigned x = 0u;
    unsigned lim = (unsigned)floorf(lam + 6.0f * sqrtf(lam)) + 1u;
    while (u > summ && x < lim) { x += 1u; prod *= lam / (float)x; summ += prod; }
    return x;
}

typedef struct {
    float r_px, sig_px, sig_f_px, r2, sigma_ln, mu_ln, max_r, ag, lambda_fac;
} Params;

static Params make_params(float r_px, float sig_px, float sig_f) {
    Params p;
    p.r_px = r_px; p.sig_px = sig_px; p.sig_f_px = sig_f; p.r2 = r_px * r_px;
    if (sig_px == 0) { p.sigma_ln = 0; p.mu_ln = 0; p.max_r = r_px; }
    else {
        float s2 = logf((sig_px / r_px) * (sig_px / r_px) + 1);
        p.sigma_ln = sqrtf(s2); p.mu_ln = logf(r_px) - 0.5f * s2;
        p.max_r = expf(p.mu_ln + 3.0902f * p.sigma_ln);
    }
    p.ag = 1.0f / ceilf(1.0f / r_px);
    p.lambda_fac = p.ag * p.ag / (3.14159265f * (p.r_px * p.r_px + p.sig_px * p.sig_px));
    return p;
}

#define MAXG 8
#define NC 32

typedef struct { float x, y, r2; } Grain;

static float dst_current(int px, int py, float u, Params* p, unsigned fseed, float* acc) {
    RNG st = { wang((unsigned)py * 73856093u ^ (unsigned)px * 19349663u ^ fseed) };
    float hit = 0.0f;
    for (unsigned s = 0; s < 98u; s++) {
        unsigned pair = s >> 1;
        float u1 = ((float)(pair % 7u) + rnd01(&st)) / 7.0f;
        float u2 = ((float)(pair / 7u) + rnd01(&st)) / 7.0f;
        float rr = sqrtf(-2.0f * logf(u1 + 1e-12f));
        float ph = 6.2831853072f * u2;
        float sgn = (s & 1u) ? -1.0f : 1.0f;
        float xS = (float)px + p->sig_f_px * rr * cosf(ph) * sgn;
        float yS = (float)py + p->sig_f_px * rr * sinf(ph) * sgn;
        float lam = -p->lambda_fac * logf(1.0f - u);
        float decay = 1.0f - 0.7f * fminf(1.0f, fmaxf(0.0f, (u - 0.5f) / 0.45f));
        lam *= decay;
        if (lam < 1e-5f) continue;
        float r_scale = 1.0f / sqrtf(decay);
        float max_r_eff = p->max_r * r_scale;
        float exp_lam = expf(-lam);
        float gr2_base = p->r2 * r_scale * r_scale;
        int min_x = (int)floorf((xS - max_r_eff) / p->ag);
        int max_x = (int)floorf((xS + max_r_eff) / p->ag);
        int min_y = (int)floorf((yS - max_r_eff) / p->ag);
        int max_y = (int)floorf((yS + max_r_eff) / p->ag);
        int covered = 0;
        for (int cx = min_x; cx <= max_x && !covered; cx++) {
            for (int cy = min_y; cy <= max_y && !covered; cy++) {
                RNG cs = { wang(((((unsigned)cy & 0xFFFFu) << 16) | ((unsigned)cx & 0xFFFFu)) + fseed) };
                unsigned n = rnd_poisson(&cs, lam, exp_lam);
                for (unsigned z = 0; z < n; z++) {
                    float ru = rnd01(&cs), rv = rnd01(&cs);
                    float xc = p->ag * ((float)cx + ru);
                    float yc = p->ag * ((float)cy + rv);
                    float gr2 = gr2_base;
                    if (p->sig_px > 0) {
                        float g1, g2;
                        rnd_gauss(&cs, &g1, &g2);
                        float rad = fminf(expf(p->mu_ln + p->sigma_ln * g1), p->max_r) * r_scale;
                        gr2 = rad * rad;
                    }
                    float dx = xc - xS, dy = yc - yS;
                    if (dx * dx + dy * dy < gr2) { covered = 1; break; }
                }
            }
        }
        hit += (float)covered;
    }
    *acc += hit;
    return 1.0f - hit / 98.0f;
}

typedef struct { int cx, cy; float lam; unsigned n; Grain g[MAXG]; unsigned used; } Slot;

static float dst_cache(int px, int py, float u, Params* p, unsigned fseed, float* acc) {
    RNG st = { wang((unsigned)py * 73856093u ^ (unsigned)px * 19349663u ^ fseed) };
    Slot slots[NC];
    for (int i = 0; i < NC; i++) slots[i].used = 0;
    float hit = 0.0f;
    for (unsigned s = 0; s < 98u; s++) {
        unsigned pair = s >> 1;
        float u1 = ((float)(pair % 7u) + rnd01(&st)) / 7.0f;
        float u2 = ((float)(pair / 7u) + rnd01(&st)) / 7.0f;
        float rr = sqrtf(-2.0f * logf(u1 + 1e-12f));
        float ph = 6.2831853072f * u2;
        float sgn = (s & 1u) ? -1.0f : 1.0f;
        float xS = (float)px + p->sig_f_px * rr * cosf(ph) * sgn;
        float yS = (float)py + p->sig_f_px * rr * sinf(ph) * sgn;
        float lam = -p->lambda_fac * logf(1.0f - u);
        float decay = 1.0f - 0.7f * fminf(1.0f, fmaxf(0.0f, (u - 0.5f) / 0.45f));
        lam *= decay;
        if (lam < 1e-5f) continue;
        float r_scale = 1.0f / sqrtf(decay);
        float max_r_eff = p->max_r * r_scale;
        float exp_lam = expf(-lam);
        float gr2_base = p->r2 * r_scale * r_scale;
        int min_x = (int)floorf((xS - max_r_eff) / p->ag);
        int max_x = (int)floorf((xS + max_r_eff) / p->ag);
        int min_y = (int)floorf((yS - max_r_eff) / p->ag);
        int max_y = (int)floorf((yS + max_r_eff) / p->ag);
        int covered = 0;
        for (int cx = min_x; cx <= max_x && !covered; cx++) {
            for (int cy = min_y; cy <= max_y && !covered; cy++) {
                unsigned h = wang((unsigned)(cx * 73856093) ^ (unsigned)(cy * 19349663) ^ (unsigned)(int)(lam * 65536.0f));
                Slot* sl = &slots[h & (NC - 1)];
                if (sl->used && sl->cx == cx && sl->cy == cy && sl->lam == lam) {
                    for (unsigned z = 0; z < sl->n; z++) {
                        float dx = sl->g[z].x - xS, dy = sl->g[z].y - yS;
                        if (dx * dx + dy * dy < sl->g[z].r2) { covered = 1; break; }
                    }
                    continue;
                }
                RNG cs = { wang(((((unsigned)cy & 0xFFFFu) << 16) | ((unsigned)cx & 0xFFFFu)) + fseed) };
                unsigned n = rnd_poisson(&cs, lam, exp_lam);
                sl->used = 1; sl->cx = cx; sl->cy = cy; sl->lam = lam; sl->n = n < MAXG ? n : MAXG;
                unsigned zi = 0;
                for (unsigned z = 0; z < n; z++) {
                    float ru = rnd01(&cs), rv = rnd01(&cs);
                    float xc = p->ag * ((float)cx + ru);
                    float yc = p->ag * ((float)cy + rv);
                    float gr2 = gr2_base;
                    if (p->sig_px > 0) {
                        float g1, g2;
                        rnd_gauss(&cs, &g1, &g2);
                        float rad = fminf(expf(p->mu_ln + p->sigma_ln * g1), p->max_r) * r_scale;
                        gr2 = rad * rad;
                    }
                    if (zi < MAXG) { sl->g[zi].x = xc; sl->g[zi].y = yc; sl->g[zi].r2 = gr2; zi++; }
                    float dx = xc - xS, dy = yc - yS;
                    if (dx * dx + dy * dy < gr2) { covered = 1; break; }
                }
                if (n > MAXG) sl->used = 0;
            }
        }
        hit += (float)covered;
    }
    *acc += hit;
    return 1.0f - hit / 98.0f;
}

int main(void) {
    Params pf = make_params(0.0088f, 0.002f, 0.016f);
    Params pb = make_params(0.04f, 0.002f, 0.008f);
    float acc = 0;
    for (int cfg = 0; cfg < 2; cfg++) {
        Params* pp = cfg ? &pb : &pf;
        for (int u_i = 0; u_i < 3; u_i++) {
            float u = u_i == 0 ? 0.15f : (u_i == 1 ? 0.5f : 0.85f);
            for (int mode = 0; mode < 2; mode++) {
                clock_t t0 = clock();
                double sum = 0;
                int N = 20000;
                for (int t = 0; t < N; t++) {
                    unsigned fseed = wang(777 + t * 13);
                    float v = mode ? dst_cache(t % 500, (t * 7) % 500, u, pp, fseed, &acc)
                                   : dst_current(t % 500, (t * 7) % 500, u, pp, fseed, &acc);
                    sum += v;
                }
                double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
                printf("%s u=%.2f %s: %.3fs for %d pixels (%.1f us/px) checksum=%.3f\n",
                       cfg ? "big " : "fine", u, mode ? "cache  " : "current", dt, N, dt / N * 1e6, sum);
            }
        }
    }
    printf("acc=%f\n", acc);
    return 0;
}
