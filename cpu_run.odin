package main

import "core:fmt"
import "core:math"
import "core:os"
import "core:strconv"
import "core:sync"
import "core:thread"
import "core:time"

Render_Params :: struct {
	width:      u32,
	height:     u32,
	n_samples:  u32,
	seed:       u32,
	y_offset:   u32,
	sigma_f:    f32,
	sigma:      f32,
	r2:         f32,
	sigma_ln:   f32,
	mu_ln:      f32,
	max_r:      f32,
	ag:         f32,
	lambda_fac: f32,
}

Bounce_Params :: struct {
	width:     u32,
	height:    u32,
	n_layers:  u32,
	n_samples: u32,
	seed:      u32,
	y_offset:  u32,
	film_px:   f32,
}

Compute_Mode :: enum {
	Cpu,
	Cuda,
}

Device_Kind :: enum {
	Auto,
	Cpu,
	Cuda,
}

Device_Choice :: struct {
	kind:    Device_Kind,
	ordinal: int,
}

Compute_Context :: struct {
	mode:    Compute_Mode,
	threads: int,
	cuda:    Cuda_Context,
	pool:    thread.Pool,
}

Render_Band :: struct {
	src:    []f32,
	neg:    []f32,
	params: Render_Params,
	y0:     int,
	y1:     int,
	done:   ^int,
}

Bounce_Band :: struct {
	front:   []f32,
	bounced: []f32,
	dens:    []f32,
	absorb:  []f32,
	depth:   []f32,
	params:  Bounce_Params,
	y0:      int,
	y1:      int,
	done:    ^int,
}

wang :: proc(seed: u32) -> u32 {
	s := (seed ~ 61) ~ (seed >> 16)
	s *= 9
	s ~= s >> 4
	s *= 668265261
	s ~= s >> 15
	return s
}

xor_shift :: proc(s: u32) -> u32 {
	v := s
	v ~= v << 13
	v ~= v >> 17
	v ~= v << 5
	return v
}

rnd01 :: proc(s: ^u32) -> f32 {
	s^ = xor_shift(s^)
	return f32(s^) * (1.0 / 4294967295.0)
}

rnd_gauss :: proc(s: ^u32) -> [2]f32 {
	r1 := rnd01(s)
	r2 := rnd01(s)
	r := math.sqrt(-2.0 * math.ln(r1 + 1e-12))
	return {r * math.cos(2 * math.PI * r2), r * math.sin(2 * math.PI * r2)}
}

rnd_poisson :: proc(s: ^u32, lam: f32, exp_lam: f32) -> u32 {
	u := rnd01(s)
	x: u32 = 0
	prod := exp_lam
	summ := exp_lam
	lim := u32(math.floor(lam + 6.0 * math.sqrt(lam))) + 1
	for u > summ && x < lim {
		x += 1
		prod *= lam / f32(x)
		summ += prod
	}
	return x
}

hemi_cosine :: proc(s: ^u32) -> [3]f32 {
	r1 := rnd01(s)
	r2 := rnd01(s)
	z := math.sqrt(r1)
	r := math.sqrt(1.0 - z * z)
	phi := 2 * math.PI * r2
	return {r * math.cos(phi), r * math.sin(phi), z}
}

render_band :: proc(task: thread.Task) {
	band := cast(^Render_Band)task.data
	p := band.params
	w := int(p.width)
	fseed := wang(p.seed)
	for py in band.y0 ..< band.y1 {
		for px in 0 ..< w {
			st := wang(u32(py) * 73856093 ~ u32(px) * 19349663 ~ fseed)
			hit: f32 = 0
			for _ in 0 ..< int(p.n_samples) {
				g1 := rnd_gauss(&st)
				g2 := rnd_gauss(&st)
				xG := f32(px) + p.sigma_f * g1[0]
				yG := f32(py) + p.sigma_f * g2[0]
				ix := clamp(int(math.floor(xG)), 0, w - 1)
				iy := clamp(int(math.floor(yG)), 0, int(p.height) - 1)
				u := clamp(band.src[iy * w + ix], 0.0, 1.0 - EPS)
				lam := -p.lambda_fac * math.ln(1.0 - u)
				exp_lam := math.exp(-lam)
				min_x := int(math.floor((xG - p.max_r) / p.ag))
				max_x := int(math.floor((xG + p.max_r) / p.ag))
				min_y := int(math.floor((yG - p.max_r) / p.ag))
				max_y := int(math.floor((yG + p.max_r) / p.ag))
				covered := false
				for cx := min_x; cx <= max_x && !covered; cx += 1 {
					for cy := min_y; cy <= max_y && !covered; cy += 1 {
						cs := wang((((u32(cy) & 0xFFFF) << 16) | (u32(cx) & 0xFFFF)) + fseed)
						n_grains := rnd_poisson(&cs, lam, exp_lam)
						for _ in 0 ..< int(n_grains) {
							ru := rnd01(&cs)
							rv := rnd01(&cs)
							xc := p.ag * (f32(cx) + ru)
							yc := p.ag * (f32(cy) + rv)
							gr2 := p.r2
							if p.sigma > 0 {
								rg := rnd_gauss(&cs)
								rad := min(math.exp(p.mu_ln + p.sigma_ln * rg[0]), p.max_r)
								gr2 = rad * rad
							}
							dx := xc - xG
							dy := yc - yG
							if dx * dx + dy * dy < gr2 {
								covered = true
								break
							}
						}
					}
				}
				if covered {
					hit += 1
				}
			}
			band.neg[py * w + px] = 1.0 - hit / f32(p.n_samples)
		}
	}
	sync.atomic_add(band.done, 1)
}

bounce_band :: proc(task: thread.Task) {
	band := cast(^Bounce_Band)task.data
	p := band.params
	w := int(p.width)
	h := int(p.height)
	n_layers := int(p.n_layers)
	fseed := wang(p.seed)
	for py in band.y0 ..< band.y1 {
		for px in 0 ..< w {
			st := wang(u32(py) * 9781 ~ u32(px) * 6271 ~ fseed)
			acc := [3]f32{0, 0, 0}
			x0 := f32(px)
			y0 := f32(py)
			for _ in 0 ..< int(p.n_samples) {
				dir := hemi_cosine(&st)
				if dir[2] < 1e-4 {
					continue
				}
				x_exit := x0 + dir[0] * p.film_px / dir[2]
				y_exit := y0 + dir[1] * p.film_px / dir[2]
				ix := int(math.floor(x_exit))
				iy := int(math.floor(y_exit))
				if ix < 0 || ix >= w || iy < 0 || iy >= h {
					continue
				}
				base := (iy * w + ix) * 3
				col := [3]f32{band.front[base], band.front[base + 1], band.front[base + 2]}
				for l in 0 ..< n_layers {
					z_l := band.depth[l]
					x_l := x0 + dir[0] * z_l / dir[2]
					y_l := y0 + dir[1] * z_l / dir[2]
					ix_l := int(math.floor(x_l))
					iy_l := int(math.floor(y_l))
					dv: f32 = 1.0
					if ix_l >= 0 && ix_l < w && iy_l >= 0 && iy_l < h {
						dv = band.dens[(l * h + iy_l) * w + ix_l]
					}
					ab := l * 3
					col[0] *= 1.0 - band.absorb[ab] + band.absorb[ab] * dv
					col[1] *= 1.0 - band.absorb[ab + 1] + band.absorb[ab + 1] * dv
					col[2] *= 1.0 - band.absorb[ab + 2] + band.absorb[ab + 2] * dv
				}
				acc[0] += col[0]
				acc[1] += col[1]
				acc[2] += col[2]
			}
			inv := 1.0 / f32(max(1, int(p.n_samples)))
			obase := (py * w + px) * 3
			band.bounced[obase] = acc[0] * inv
			band.bounced[obase + 1] = acc[1] * inv
			band.bounced[obase + 2] = acc[2] * inv
		}
	}
	sync.atomic_add(band.done, 1)
}

wait_progress :: proc(done: ^int, total: int, label: string) {
	last_pct := -1
	for {
		d := sync.atomic_load(done)
		pct := d * 100 / total
		if pct != last_pct {
			info(fmt.tprintf("%s: %d%%", label, pct))
			last_pct = pct
		}
		if d >= total {
			break
		}
		time.sleep(250 * time.Millisecond)
	}
}

sim_init :: proc(ctx: ^Compute_Context, w: u32, h: u32, max_layers: u32, choice: Device_Choice) -> bool {
	switch choice.kind {
	case .Cuda:
		if cuda_init(&ctx.cuda, w, h, max_layers, choice.ordinal) {
			ctx.mode = .Cuda
			return true
		}
		fail("CUDA 设备初始化失败")
		return false
	case .Cpu:
		ctx.mode = .Cpu
	case .Auto:
		if cuda_init(&ctx.cuda, w, h, max_layers, 0) {
			ctx.mode = .Cuda
			return true
		}
		ctx.mode = .Cpu
	}
	n := 8
	if env, found := os.lookup_env_alloc("NUMBER_OF_PROCESSORS", context.allocator); found {
		defer delete(env)
		if v, ok := strconv.parse_int(env); ok {
			n = max(1, v)
		}
	}
	ctx.threads = n
	info(fmt.tprintf("CPU 渲染器: %d 线程", n))
	return true
}

sim_cleanup :: proc(ctx: ^Compute_Context) {
	switch ctx.mode {
	case .Cuda:
		cuda_cleanup(&ctx.cuda)
	case .Cpu:
	}
}

cpu_dispatch_render :: proc(ctx: ^Compute_Context, params: Render_Params, src: []f32, neg: []f32) -> bool {
	w := int(params.width)
	h := int(params.height)
	band_h := 16
	total := (h + band_h - 1) / band_h
	bands := make([]Render_Band, total)
	defer delete(bands)
	done: int
	thread.pool_init(&ctx.pool, context.allocator, ctx.threads)
	defer thread.pool_destroy(&ctx.pool)
	thread.pool_start(&ctx.pool)
	for b in 0 ..< total {
		y0 := b * band_h
		y1 := min(y0 + band_h, h)
		bands[b] = Render_Band {
			src    = src,
			neg    = neg,
			params = params,
			y0     = y0,
			y1     = y1,
			done   = &done,
		}
		thread.pool_add_task(&ctx.pool, context.allocator, render_band, &bands[b])
	}
	if total > 8 {
		wait_progress(&done, total, "渲染")
	}
	thread.pool_finish(&ctx.pool)
	return true
}

cpu_dispatch_bounce :: proc(
	ctx: ^Compute_Context,
	params: Bounce_Params,
	front: []f32,
	bounced: []f32,
	dens: []f32,
	absorb: []f32,
	depth: []f32,
) -> bool {
	w := int(params.width)
	h := int(params.height)
	band_h := 16
	total := (h + band_h - 1) / band_h
	bands := make([]Bounce_Band, total)
	defer delete(bands)
	done: int
	thread.pool_init(&ctx.pool, context.allocator, ctx.threads)
	defer thread.pool_destroy(&ctx.pool)
	thread.pool_start(&ctx.pool)
	for b in 0 ..< total {
		y0 := b * band_h
		y1 := min(y0 + band_h, h)
		bands[b] = Bounce_Band {
			front   = front,
			bounced = bounced,
			dens    = dens,
			absorb  = absorb,
			depth   = depth,
			params  = params,
			y0      = y0,
			y1      = y1,
			done    = &done,
		}
		thread.pool_add_task(&ctx.pool, context.allocator, bounce_band, &bands[b])
	}
	if total > 8 {
		wait_progress(&done, total, "回弹")
	}
	thread.pool_finish(&ctx.pool)
	return true
}

dispatch_render :: proc(ctx: ^Compute_Context, params: Render_Params, src: []f32, neg: []f32) -> bool {
	switch ctx.mode {
	case .Cuda:
		return cuda_dispatch_render(&ctx.cuda, params, src, neg)
	case .Cpu:
		return cpu_dispatch_render(ctx, params, src, neg)
	}
	return false
}

dispatch_bounce :: proc(
	ctx: ^Compute_Context,
	params: Bounce_Params,
	front: []f32,
	bounced: []f32,
	dens: []f32,
	absorb: []f32,
	depth: []f32,
) -> bool {
	switch ctx.mode {
	case .Cuda:
		return cuda_dispatch_bounce(&ctx.cuda, params, front, bounced, dens, absorb, depth)
	case .Cpu:
		return cpu_dispatch_bounce(ctx, params, front, bounced, dens, absorb, depth)
	}
	return false
}
