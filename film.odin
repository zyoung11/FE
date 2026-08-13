package main

import "core:math"
import "core:os"
import "core:strconv"
import "core:thread"

EPS :: 1e-5

// Grain smoothing blur radius (px), used to remove white sampling noise in shadows
GRAIN_SMOOTH_SIGMA :: 1.0

// Per-channel penetration depth factors for halation (red penetrates deepest)
halation_chan_depth :: proc(c: int) -> f32 {
	switch c {
	case 0:
		return 1.25
	case 1:
		return 1.0
	}
	return 0.8
}
// Front-interface reflectance for the second halation bounce
HALATION_R_FRONT :: 0.3

Film_Params :: struct {
    r_px:       f32,
    sig_px:     f32,
    sig_f_px:   f32,
    r2:         f32,
    sigma_ln:   f32,
    mu_ln:      f32,
    max_r:      f32,
    ag:         f32,
    lambda_fac: f32,
}

Gauss_Band :: struct {
    src:    []f32,
    dst:    []f32,
    kernel: []f32,
    w:      int,
    h:      int,
    r:      int,
    x0:     int,
    x1:     int,
    y0:     int,
    y1:     int,
}

gauss_h_task :: proc(task: thread.Task) {
    band := cast(^Gauss_Band)task.data
    for y in band.y0 ..< band.y1 {
        row := band.src[y * band.w:(y + 1) * band.w]
        for x in 0 ..< band.w {
            acc: f32
            ksum: f32
            for j in 0 ..< len(band.kernel) {
                sx := x + j - band.r
                if sx >= 0 && sx < band.w {
                    k := band.kernel[j]
                    acc += row[sx] * k
                    ksum += k
                }
            }
            band.dst[y * band.w + x] = acc / ksum
        }
    }
}

gauss_v_task :: proc(task: thread.Task) {
    band := cast(^Gauss_Band)task.data
    for x in band.x0 ..< band.x1 {
        for y in 0 ..< band.h {
            acc: f32
            ksum: f32
            for j in 0 ..< len(band.kernel) {
                sy := y + j - band.r
                if sy >= 0 && sy < band.h {
                    k := band.kernel[j]
                    acc += band.src[sy * band.w + x] * k
                    ksum += k
                }
            }
            band.dst[y * band.w + x] = acc / ksum
        }
    }
}

cpu_thread_count :: proc() -> int {
    n := 8
    if env, found := os.lookup_env_alloc("NUMBER_OF_PROCESSORS", context.allocator); found {
        defer delete(env)
        if v, ok := strconv.parse_int(env); ok {
            n = max(1, v)
        }
    }
    return n
}

prep_physics :: proc(r_px: f32, sig_px: f32, sig_f_px: f32) -> Film_Params {
    params: Film_Params
    params.r_px = r_px
    params.sig_px = sig_px
    params.sig_f_px = sig_f_px
    params.r2 = r_px * r_px

    if sig_px == 0 {
        params.sigma_ln = 0
        params.mu_ln = 0
    } else {
        sigma2_ln := math.log((sig_px / r_px) * (sig_px / r_px) + 1, math.E)
        params.sigma_ln = math.sqrt(sigma2_ln)
        params.mu_ln = math.log(r_px, math.E) - 0.5 * sigma2_ln
    }

    if sig_px == 0 {
        params.max_r = r_px
    } else {
        params.max_r = math.exp(params.mu_ln + 3.0902 * params.sigma_ln)
    }

    if r_px > 0 {
        params.ag = 1.0 / math.ceil(1.0 / r_px)
    } else {
        params.ag = 1.0
    }

    if r_px > 0 {
        params.lambda_fac = params.ag * params.ag / (math.PI * (r_px * r_px + sig_px * sig_px))
    } else {
        params.lambda_fac = 0
    }

    return params
}

srgb_to_linear :: proc(v: f32, gamma: f32) -> f32 {
    a := v / 255.0
    if a <= 0.04045 {
        return a / 12.92
    }
    return math.pow((a + 0.055) / 1.055, gamma)
}

linear_to_srgb :: proc(x: f32, gamma: f32) -> f32 {
    if x <= 0.0031308 {
        return x * 12.92
    }
    return 1.055 * math.pow(x, 1.0 / gamma) - 0.055
}

filmic_curve :: proc(x: f32, strength: f32) -> f32 {
    if strength <= 0 {
        return x
    }
    a := 2.51 * x
    c := 2.43 * x
    y := (x * (a + 0.03)) / (x * (c + 0.59) + 0.14)
    return x * (1.0 - strength) + y * strength
}

print_curve :: proc(x: f32, toe: f32, shoulder: f32) -> f32 {
    if x <= 0 {
        return 0
    }
    if x >= 1 {
        return 1
    }
    t := 0.25 * toe
    s := 0.25 * shoulder
    if x <= t {
        return x * x / (2.0 * t)
    }
    if x >= 1.0 - s {
        u := (1.0 - x) / s
        return 1.0 - u * u / 2.0
    }
    k := (1.0 - s / 2.0 - t / 2.0) / (1.0 - s - t)
    return t / 2.0 + k * (x - t)
}

build_vectors :: proc(dye: [3]u8) -> (absorb, expo_w: [3]f32) {
    comp := [3]f32 {
        1.0 - f32(dye[0]) / 255.0,
        1.0 - f32(dye[1]) / 255.0,
        1.0 - f32(dye[2]) / 255.0,
    }
    if comp == ([3]f32 {1, 1, 1}) {
        return {1, 1, 1}, {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}
    }
    nz := 0
    for c in comp {
        if abs(c) > 1e-6 {
            nz += 1
        }
    }
    if nz == 1 {
        return comp, comp
    }
    s := comp[0] + comp[1] + comp[2]
    if s <= 1e-6 {
        return comp, {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}
    }
    return comp, comp / s
}

inverse_reinhard :: proc(x: f32) -> f32 {
    return x / (1.0 - x + EPS)
}

adaptive_blur :: proc(src: []f32, w: int, h: int, sigma_min: f32, sigma_max: f32) -> []f32 {
    if sigma_max <= sigma_min {
        return gauss_blur(src, w, h, sigma_max)
    }
    lap := make([]f32, len(src))
    defer delete(lap)
    for y in 1 ..< h - 1 {
        for x in 1 ..< w - 1 {
            i := y * w + x
            lap[i] = abs(4.0 * src[i] - src[i - w] - src[i + w] - src[i - 1] - src[i + 1])
        }
    }
    energy := gauss_blur(lap, w, h, max(sigma_min * 2.0, 2.0))
    defer delete(energy)
    mean: f32
    for e in energy {
        mean += e
    }
    mean /= f32(len(energy))
    t := make([]f32, len(src))
    defer delete(t)
    for i in 0 ..< len(src) {
        t[i] = clamp(energy[i] / (2.0 * mean + 1e-6), 0.0, 1.0)
    }
    blur_min := gauss_blur(src, w, h, sigma_min)
    defer delete(blur_min)
    blur_max := gauss_blur(src, w, h, sigma_max)
    defer delete(blur_max)
    out := make([]f32, len(src))
    for i in 0 ..< len(src) {
        out[i] = blur_min[i] * (1.0 - t[i]) + blur_max[i] * t[i]
    }
    return out
}

gauss_pass :: proc(src: []f32, dst: []f32, kernel: []f32, w: int, h: int, r: int, vertical: bool) {
    n_threads := cpu_thread_count()
    pool: thread.Pool
    thread.pool_init(&pool, context.allocator, n_threads)
    defer thread.pool_destroy(&pool)
    thread.pool_start(&pool)

    blocks := n_threads * 4
    bands := make([]Gauss_Band, blocks)
    defer delete(bands)
    if vertical {
        band_w := (w + blocks - 1) / blocks
        for b in 0 ..< blocks {
            x0 := min(b * band_w, w)
            x1 := min(x0 + band_w, w)
            if x0 < x1 {
                bands[b] = Gauss_Band{src = src, dst = dst, kernel = kernel, w = w, h = h, r = r, x0 = x0, x1 = x1}
                thread.pool_add_task(&pool, context.allocator, gauss_v_task, &bands[b])
            }
        }
    } else {
        band_h := (h + blocks - 1) / blocks
        for b in 0 ..< blocks {
            y0 := min(b * band_h, h)
            y1 := min(y0 + band_h, h)
            if y0 < y1 {
                bands[b] = Gauss_Band{src = src, dst = dst, kernel = kernel, w = w, h = h, r = r, y0 = y0, y1 = y1}
                thread.pool_add_task(&pool, context.allocator, gauss_h_task, &bands[b])
            }
        }
    }
    thread.pool_finish(&pool)
}

gauss_blur :: proc(src: []f32, w: int, h: int, sigma: f32) -> []f32 {
    out := make([]f32, len(src))
    if sigma <= 0 {
        copy(out, src)
        return out
    }
    r := max(1, int(math.ceil(3.0 * sigma)))
    kernel := make([]f32, 2 * r + 1)
    defer delete(kernel)
    sum: f32
    for i in 0 ..< 2 * r + 1 {
        x := f32(i - r)
        k := math.exp(-0.5 * (x / sigma) * (x / sigma))
        kernel[i] = k
        sum += k
    }
    for i in 0 ..< len(kernel) {
        kernel[i] /= sum
    }

    tmp := make([]f32, len(src))
    defer delete(tmp)
    gauss_pass(src, tmp, kernel, w, h, r, false)
    gauss_pass(tmp, out, kernel, w, h, r, true)
    return out
}

gauss_blur_dispatch :: proc(ctx: ^Compute_Context, src: []f32, w: int, h: int, sigma: f32, keep_on_device := false) -> ([]f32, bool) {
    if ctx.mode == .Cuda {
        return cuda_gauss_blur(&ctx.cuda, src, u32(w), u32(h), sigma, keep_on_device)
    }
    return gauss_blur(src, w, h, sigma), true
}

adaptive_blur_dispatch :: proc(ctx: ^Compute_Context, src: []f32, w: int, h: int, sigma_min: f32, sigma_max: f32) -> ([]f32, bool) {
    if ctx.mode == .Cuda {
        return cuda_adaptive_blur(&ctx.cuda, src, u32(w), u32(h), sigma_min, sigma_max)
    }
    return adaptive_blur(src, w, h, sigma_min, sigma_max), true
}

// Halation bounce resolution divisor (halation is low-frequency; full-res bounce is wasted work)
HALATION_SCALE :: 4

// Box-filter downsample of an interleaved multi-channel image
box_downsample :: proc(src: []f32, sw: int, sh: int, channels: int, dw: int, dh: int) -> []f32 {
    out := make([]f32, dw * dh * channels)
    for y in 0 ..< dh {
        sy0 := y * sh / dh
        sy1 := max(sy0 + 1, (y + 1) * sh / dh)
        for x in 0 ..< dw {
            sx0 := x * sw / dw
            sx1 := max(sx0 + 1, (x + 1) * sw / dw)
            for c in 0 ..< channels {
                acc: f32
                count := 0
                for sy in sy0 ..< sy1 {
                    for sx in sx0 ..< sx1 {
                        acc += src[(sy * sw + sx) * channels + c]
                        count += 1
                    }
                }
                out[(y * dw + x) * channels + c] = acc / f32(count)
            }
        }
    }
    return out
}

// Box-filter downsample of a plane-major layer stack (layout: layer * sw * sh)
box_downsample_planes :: proc(src: []f32, sw: int, sh: int, planes: int, dw: int, dh: int) -> []f32 {
    out := make([]f32, planes * dw * dh)
    for l in 0 ..< planes {
        src_l := src[l * sw * sh:(l + 1) * sw * sh]
        dst_l := out[l * dw * dh:(l + 1) * dw * dh]
        for y in 0 ..< dh {
            sy0 := y * sh / dh
            sy1 := max(sy0 + 1, (y + 1) * sh / dh)
            for x in 0 ..< dw {
                sx0 := x * sw / dw
                sx1 := max(sx0 + 1, (x + 1) * sw / dw)
                acc: f32
                count := 0
                for sy in sy0 ..< sy1 {
                    for sx in sx0 ..< sx1 {
                        acc += src_l[sy * sw + sx]
                        count += 1
                    }
                }
                dst_l[y * dw + x] = acc / f32(count)
            }
        }
    }
    return out
}

// Bilinear upsample of an interleaved multi-channel image
bilinear_upsample :: proc(src: []f32, sw: int, sh: int, channels: int, dw: int, dh: int) -> []f32 {
    out := make([]f32, dw * dh * channels)
    sx := f32(sw) / f32(dw)
    sy := f32(sh) / f32(dh)
    for y in 0 ..< dh {
        fy := (f32(y) + 0.5) * sy - 0.5
        y0 := clamp(int(fy), 0, sh - 1)
        y1 := min(y0 + 1, sh - 1)
        ty := clamp(fy - f32(y0), 0.0, 1.0)
        for x in 0 ..< dw {
            fx := (f32(x) + 0.5) * sx - 0.5
            x0 := clamp(int(fx), 0, sw - 1)
            x1 := min(x0 + 1, sw - 1)
            tx := clamp(fx - f32(x0), 0.0, 1.0)
            for c in 0 ..< channels {
                v00 := src[(y0 * sw + x0) * channels + c]
                v10 := src[(y0 * sw + x1) * channels + c]
                v01 := src[(y1 * sw + x0) * channels + c]
                v11 := src[(y1 * sw + x1) * channels + c]
                out[(y * dw + x) * channels + c] = (v00 * (1.0 - tx) + v10 * tx) * (1.0 - ty) + (v01 * (1.0 - tx) + v11 * tx) * ty
            }
        }
    }
    return out
}
