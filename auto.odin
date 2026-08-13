package main

import "core:c"
import "core:fmt"
import stb "vendor:stb/image"

Auto_Result :: struct {
	grain_radius: f32,
	grain_sigma:  f32,
	sigma_filter: f32,
	mtf_abs:      f32,
}

content_sharpness :: proc(pixels: [^]u8, w: int, h: int, target_h: int) -> f32 {
	gray_src := make([]u8, w * h)
	defer delete(gray_src)
	for i in 0 ..< w * h {
		gray_src[i] = u8(
			0.299 * f32(pixels[i * 3]) +
			0.587 * f32(pixels[i * 3 + 1]) +
			0.114 * f32(pixels[i * 3 + 2]),
		)
	}
	scale := min(1.0, f32(target_h) / f32(max(w, h)))
	tw := max(1, int(f32(w) * scale))
	th := max(1, int(f32(h) * scale))
	gray := make([]u8, tw * th)
	defer delete(gray)
	stb.resize_uint8_generic(
		raw_data(gray_src),
		c.int(w),
		c.int(h),
		c.int(w),
		raw_data(gray),
		c.int(tw),
		c.int(th),
		c.int(tw),
		1,
		stb.ALPHA_CHANNEL_NONE,
		0,
		.CLAMP,
		.CATMULLROM,
		.LINEAR,
		nil,
	)
	var_sum: f32
	count := 0
	for y in 0 ..< th {
		for x in 0 ..< tw - 1 {
			var_sum += abs(f32(gray[y * tw + x + 1]) - f32(gray[y * tw + x]))
			count += 1
		}
	}
	for y in 0 ..< th - 1 {
		for x in 0 ..< tw {
			var_sum += abs(f32(gray[(y + 1) * tw + x]) - f32(gray[y * tw + x]))
			count += 1
		}
	}
	return var_sum / f32(count)
}

compute_auto :: proc(sharpness: f32, t_mtf: f32, height: int) -> Auto_Result {
	t := clamp((sharpness - 3.0) / 4.5, 0.0, 1.0)
	g := (0.09 - 0.05 * t) / 10.0
	m_mtf := min(2.89 - 2.39 * t_mtf, max(0.5, 0.003 * f32(height)))
	return Auto_Result{grain_radius = g, grain_sigma = 0.001, sigma_filter = 0.004, mtf_abs = m_mtf}
}

sample_luma :: proc(pixels: []u8, w: int, h: int) -> f32 {
	sum: f32
	n := 0
	for y := 0; y < h; y += 16 {
		for x := 0; x < w; x += 16 {
			i := (y * w + x) * 3
			sum += (0.299 * f32(pixels[i]) + 0.587 * f32(pixels[i + 1]) + 0.114 * f32(pixels[i + 2])) / 255.0
			n += 1
		}
	}
	return sum / f32(max(1, n))
}

image_stats :: proc(pixels: []u8, w: int, h: int) -> (avg: f32, lo: f32, hi: f32) {
	hist: [256]int
	sum: f32
	n := 0
	for y := 0; y < h; y += 16 {
		for x := 0; x < w; x += 16 {
			i := (y * w + x) * 3
			l := u8(0.299 * f32(pixels[i]) + 0.587 * f32(pixels[i + 1]) + 0.114 * f32(pixels[i + 2]))
			hist[l] += 1
			sum += f32(l) / 255.0
			n += 1
		}
	}
	n = max(1, n)
	avg = sum / f32(n)
	cum := 0
	for v in 0 ..< 256 {
		cum += hist[v]
		if cum * 100 >= n * 5 {
			lo = f32(v) / 255.0
			break
		}
	}
	cum = 0
	for v in 0 ..< 256 {
		cum += hist[v]
		if cum * 100 >= n * 95 {
			hi = f32(v) / 255.0
			break
		}
	}
	if hi - lo < 0.1 {
		hi = min(1.0, lo + 0.1)
	}
	return
}

build_auto_config :: proc(
	mode: string,
	thickness: f32,
	reflectance: f32,
	r: f32,
	s: f32,
	f: f32,
	mtf_abs: f32,
) -> Film_Config {
	cfg: Film_Config
	if mode == "bw" {
		append(
			&cfg.emulsions,
			Emulsion_Cfg {
				dye          = {0, 0, 0},
				grain_radius = r,
				grain_sigma  = s,
				sigma_filter = f,
				mtf_blur     = mtf_abs * 0.6,
				mtf_blur_max = mtf_abs * 1.4,
			},
		)
		append(&cfg.order, Config_Item {kind = .Emulsion, index = 0})
	} else {
		dyes := [3][3]u8{{255, 255, 0}, {255, 0, 255}, {0, 255, 255}}
		for i in 0 ..< 3 {
			append(
				&cfg.emulsions,
				Emulsion_Cfg {
					dye          = dyes[i],
					grain_radius = r,
					grain_sigma  = s,
					sigma_filter = f,
					mtf_blur     = mtf_abs * 0.6,
					mtf_blur_max = mtf_abs * 1.4,
				},
			)
			append(&cfg.order, Config_Item {kind = .Emulsion, index = i})
			if i < 2 {
				append(&cfg.filters, Filter_Cfg {color = dyes[i]})
				append(&cfg.order, Config_Item {kind = .Filter, index = i})
			}
		}
	}
	append(&cfg.bases, Film_Base_Cfg {thickness = thickness})
	append(&cfg.order, Config_Item {kind = .Film_Base, index = 0})
	append(&cfg.backs, Back_Cfg {reflectance = reflectance})
	append(&cfg.order, Config_Item {kind = .Back, index = 0})
	return cfg
}

build_auto_config_from_pixels :: proc(opts: ^Options, pixels: []u8, w: int, h: int, for_video: bool) -> (cfg: Film_Config, ok: bool) {
	if opts.supersample == 0 {opts.supersample = 2}
	if opts.gamma == 0 {opts.gamma = 2.4}
	if opts.reflectance < 0 {opts.reflectance = 0.05}
	if opts.thickness < 0 {opts.thickness = 20.0}
	auto_mtf := opts.mtf
	sharpness := content_sharpness(raw_data(pixels), w, h, opts.height)
	w_out := max(1, int(f32(w) * f32(opts.height) / f32(h) + 0.5))
	density := f32(w * h) / f32(w_out * opts.height)
	t_mtf := clamp(-0.1194 * sharpness - 0.05 * density + 1.696, 0.0, 1.0)
	auto_res := compute_auto(sharpness, t_mtf, opts.height)
	gr := auto_res.grain_radius
	sf := auto_res.sigma_filter
	gs := auto_res.grain_sigma
	gr *= 2.2
	sf *= 4.0
	gs *= 2.0
	info(fmt.tprintf("Content sharpness: %.2f", sharpness))
	if opts.grain_radius < 0 {opts.grain_radius = gr}
	if opts.grain_sigma < 0 {opts.grain_sigma = gs}
	if opts.sigma_filter < 0 {opts.sigma_filter = sf}
	if auto_mtf < 0 {
		auto_mtf = auto_res.mtf_abs
		if for_video {
			auto_mtf *= 2.0
		}
	}
	t := clamp((sharpness - 3.0) / 4.5, 0.0, 1.0)
	avg, lo, hi := image_stats(pixels, w, h)
	if opts.film == 0 {opts.film = 0.5 + 0.2 * t}
	if opts.print_toe < 0 {opts.print_toe = clamp(0.15 + (hi - lo) * 0.3, 0.2, 0.5)}
	if opts.print_shoulder < 0 {opts.print_shoulder = clamp(0.15 + (hi - lo) * 0.3, 0.2, 0.5)}
	mtf_boost := max(0.0, 0.686 - t_mtf) * 0.25
	if opts.sat_lo < 0 {opts.sat_lo = max(0.05, 0.12 + 0.08 * (1.0 - avg) - mtf_boost * 0.5)}
	if opts.sat_hi < 0 {opts.sat_hi = max(0.05, 0.15 + 0.1 * t - mtf_boost * 0.5)}
	if opts.cross < 0 {opts.cross = 0.03}
	opts.exposure = clamp(0.08 - (avg - 0.45) * 0.2, -0.2, 0.35)
	opts.contrast = clamp(1.15 - (hi - lo) * 0.3 + mtf_boost, 0.85, 1.25)
	info(
		fmt.tprintf(
			"auto: grain R=%.3f SIG=%.3f F=%.3f MTF=%.3f print T=%.2f S=%.2f EXP=%.3f CON=%.2f",
			opts.grain_radius,
			opts.grain_sigma,
			opts.sigma_filter,
			auto_mtf,
			opts.print_toe,
			opts.print_shoulder,
			opts.exposure,
			opts.contrast,
		),
	)
	cfg = build_auto_config(
		opts.mode,
		opts.thickness,
		opts.reflectance,
		opts.grain_radius,
		opts.grain_sigma,
		opts.sigma_filter,
		auto_mtf,
	)
	return cfg, true
}

