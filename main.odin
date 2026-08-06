package main

import "core:c"
import "core:flags"
import "core:fmt"
import "core:math"
import "core:os"
import "core:path/filepath"
import "core:strconv"
import "core:strings"
import "core:sys/windows"
import "core:time"
import stb "vendor:stb/image"

BLUE :: "\x1b[38;2;138;173;244m"
GREEN :: "\x1b[38;2;166;218;149m"
YELLOW :: "\x1b[38;2;238;212;159m"
RED :: "\x1b[38;2;237;135;150m"
RESET :: "\x1b[0m"

info :: proc(msg: string) {fmt.printfln("%s[i]%s %s", BLUE, RESET, msg)}
success :: proc(msg: string) {fmt.printfln("%s[+]%s %s", GREEN, RESET, msg)}
warn :: proc(msg: string) {fmt.printfln("%s[!]%s %s", YELLOW, RESET, msg)}
fail :: proc(msg: string) {fmt.eprintfln("%s[-]%s %s", RED, RESET, msg)}

stage_time :: proc(label: string, t0: time.Time) -> time.Time {
	info(fmt.tprintf("%s: %.2fs", label, time.duration_seconds(time.since(t0))))
	return time.now()
}

Options :: struct {
	input:          string `usage:"输入图像路径" args:"required"`,
	output:         string `usage:"输出图像路径 (PNG)"`,
	auto:           bool `usage:"一键胶片风格 (内容自适应配置)"`,
	mode:           string `usage:"auto 模式胶片类型: color(默认) | bw"`,
	film_cfg:       string `args:"name=film_cfg" usage:"胶片配置 TOML (auto 模式下忽略)"`,
	height:         int `usage:"输出高度 (像素)"`,
	supersample:    int `usage:"超采样倍数"`,
	samples:        int `usage:"每像素蒙特卡洛采样数"`,
	bounce_samples: int `args:"name=bounce_samples" usage:"halation 回弹采样数"`,
	gamma:          f32 `usage:"sRGB gamma"`,
	mtf:            f32 `usage:"MTF 软化 (px, 绝对值, -1 使用配置)"`,
	exposure:       f32 `usage:"曝光补偿 (999 使用默认)"`,
	contrast:       f32 `usage:"对比度 (0 使用默认)"`,
	reflectance:    f32 `usage:"背面反射率 (-1 使用配置)"`,
	thickness:      f32 `usage:"片基厚度 (-1 使用配置)"`,
	grain_radius:   f32 `args:"name=grain_radius" usage:"覆盖颗粒半径 (像素, -1 使用配置)"`,
	grain_sigma:    f32 `args:"name=grain_sigma" usage:"覆盖颗粒 sigma (-1 使用配置)"`,
	sigma_filter:   f32 `args:"name=sigma_filter" usage:"覆盖采样抖动 sigma (-1 使用配置)"`,
	seed:           u32 `usage:"随机种子"`,
	device:         string `usage:"渲染设备: auto(默认) | cpu | cuda | cuda:N"`,
}

main :: proc() {
	when ODIN_OS == .Windows {
		windows.SetConsoleOutputCP(windows.CODEPAGE(65001))
		windows.SetConsoleCP(windows.CODEPAGE(65001))
	}
	start := time.now()
	opts: Options
	opts.output = "out.png"
	opts.mode = "color"
	opts.film_cfg = "film-config.toml"
	opts.height = 1080
	opts.supersample = 0
	opts.samples = 0
	opts.bounce_samples = 0
	opts.gamma = 0
	opts.mtf = -1.0
	opts.exposure = 999.0
	opts.contrast = 0.0
	opts.reflectance = -1.0
	opts.thickness = -1.0
	opts.grain_radius = -1.0
	opts.grain_sigma = -1.0
	opts.sigma_filter = -1.0
	opts.seed = 12345
	opts.device = "auto"

	if err := flags.parse(&opts, os.args[1:], style = .Unix); err != nil {
		if _, is_help := err.(flags.Help_Request); is_help {
			flags.write_usage(os.to_stream(os.stdout), Options, filepath.base(os.args[0]), .Unix)
			os.exit(0)
		}
		fail(fmt.tprintf("参数解析失败: %v", err))
		os.exit(1)
	}
	if opts.mode != "color" && opts.mode != "bw" {
		fail("--mode 只能是 color 或 bw")
		os.exit(1)
	}
	if opts.height <= 0 {
		fail("height 必须为正数")
		os.exit(1)
	}
	device_choice: Device_Choice
	switch opts.device {
	case "auto":
		device_choice.kind = .Auto
	case "cpu":
		device_choice.kind = .Cpu
	case "cuda":
		device_choice.kind = .Cuda
		device_choice.ordinal = 0
	case:
		if strings.has_prefix(opts.device, "cuda:") {
			n, ok := strconv.parse_int(opts.device[5:])
			if !ok || n < 0 {
				fail("--device 格式: auto | cpu | cuda | cuda:N")
				os.exit(1)
			}
			device_choice.kind = .Cuda
			device_choice.ordinal = n
		} else {
			fail("--device 格式: auto | cpu | cuda | cuda:N")
			os.exit(1)
		}
	}
	if opts.auto {
		if opts.samples == 0 {opts.samples = 400}
		if opts.bounce_samples == 0 {opts.bounce_samples = 400}
		if opts.supersample == 0 {opts.supersample = 2}
		if opts.gamma == 0 {opts.gamma = 2.4}
		if opts.exposure == 999.0 {opts.exposure = 0.08}
		if opts.contrast == 0 {opts.contrast = 0.92}
		if opts.reflectance < 0 {opts.reflectance = 0.03}
		if opts.thickness < 0 {opts.thickness = 20.0}
	} else {
		if opts.samples == 0 {opts.samples = 400}
		if opts.bounce_samples == 0 {opts.bounce_samples = 400}
		if opts.supersample == 0 {opts.supersample = 1}
		if opts.gamma == 0 {opts.gamma = 2.2}
		if opts.exposure == 999.0 {opts.exposure = 0.0}
		if opts.contrast == 0 {opts.contrast = 1.0}
	}

	input_cstr := strings.clone_to_cstring(opts.input)
	defer delete(input_cstr)
	w, h, ch: c.int
	t := time.now()
	pixels := stb.load(input_cstr, &w, &h, &ch, 3)
	if pixels == nil {
		fail(fmt.tprintf("无法读取图像: %s", opts.input))
		os.exit(1)
	}
	defer stb.image_free(pixels)
	info(fmt.tprintf("输入: %s (%dx%d)", opts.input, w, h))
	t = stage_time("图像加载", t)

	auto_mtf := opts.mtf
	if opts.auto {
		sharpness := content_sharpness(pixels, int(w), int(h))
		auto_res := compute_auto(sharpness, opts.height)
		info(fmt.tprintf("内容锐度: %.2f", sharpness))
		if opts.grain_radius < 0 {opts.grain_radius = auto_res.grain_radius}
		if opts.grain_sigma < 0 {opts.grain_sigma = auto_res.grain_sigma}
		if opts.sigma_filter < 0 {opts.sigma_filter = auto_res.sigma_filter}
		if auto_mtf < 0 {auto_mtf = auto_res.mtf_abs}
		info(
			fmt.tprintf(
				"auto: 颗粒 R=%.3f SIG=%.3f F=%.3f MTF=%.3f",
				opts.grain_radius,
				opts.grain_sigma,
				opts.sigma_filter,
				auto_mtf,
			),
		)
	}
	cfg_text := ""
	cfg_text_owned := false
	defer if cfg_text_owned {delete(cfg_text)}
	if opts.auto {
		cfg_text = build_cfg_text(
			opts.mode,
			opts.thickness,
			opts.reflectance,
			opts.grain_radius,
			opts.grain_sigma,
			opts.sigma_filter,
			auto_mtf,
		)
		cfg_text_owned = true
	}
	t = stage_time("auto 分析", t)

	cfg: Film_Config
	cfg_ok: bool
	if opts.auto {
		cfg, cfg_ok = parse_film_config_text(cfg_text)
	} else {
		cfg, cfg_ok = parse_film_config(opts.film_cfg)
	}
	if !cfg_ok {
		fail(fmt.tprintf("无法解析胶片配置: %s", opts.film_cfg))
		os.exit(1)
	}
	defer destroy_film_config(&cfg)
	t = stage_time("配置解析", t)
	if len(cfg.emulsions) == 0 {
		fail("配置中至少需要一个 [[emulsion]] 块")
		os.exit(1)
	}
	if len(cfg.filters) != len(cfg.emulsions) && len(cfg.filters) != len(cfg.emulsions) - 1 {
		fail("#[[filter]] 必须等于 #[[emulsion]] 或少一个")
		os.exit(1)
	}
	film_thick := f32(1.0)
	if len(cfg.bases) > 0 {
		film_thick = cfg.bases[0].thickness
	}
	back_refl := f32(0.0)
	if len(cfg.backs) > 0 {
		back_refl = cfg.backs[0].reflectance
	}
	if opts.thickness >= 0 {
		film_thick = opts.thickness
	}
	if opts.reflectance >= 0 {
		back_refl = opts.reflectance
	}
	if opts.mtf >= 0 {
		for &emu in cfg.emulsions {
			emu.mtf_blur = opts.mtf * 0.6
			emu.mtf_blur_max = opts.mtf * 1.4
		}
	}

	h_sim := opts.height * opts.supersample
	w_sim := max(1, int(f32(w) * f32(h_sim) / f32(h) + 0.5))
	resized := make([]u8, w_sim * h_sim * 3)
	defer delete(resized)
	if int(w) != w_sim || int(h) != h_sim {
		stb.resize_uint8(
			pixels,
			w,
			h,
			w * 3,
			raw_data(resized),
			c.int(w_sim),
			c.int(h_sim),
			c.int(w_sim * 3),
			3,
		)
		info(fmt.tprintf("缩放至: %dx%d", w_sim, h_sim))
	} else {
		copy(resized, pixels[:w * h * 3])
	}

	n := w_sim * h_sim
	ray_front := make([]f32, n * 3)
	defer delete(ray_front)
	for i in 0 ..< n {
		ray_front[i * 3 + 0] = srgb_to_linear(f32(resized[i * 3 + 0]), opts.gamma)
		ray_front[i * 3 + 1] = srgb_to_linear(f32(resized[i * 3 + 1]), opts.gamma)
		ray_front[i * 3 + 2] = srgb_to_linear(f32(resized[i * 3 + 2]), opts.gamma)
	}
	t = stage_time("缩放与线性化", t)

	ctx: Compute_Context
	if !sim_init(&ctx, u32(w_sim), u32(h_sim), u32(len(cfg.emulsions)), device_choice) {
		fail("渲染器初始化失败")
		os.exit(1)
	}
	defer sim_cleanup(&ctx)
	t = stage_time("渲染器初始化", t)

	layer_absorbs: [dynamic][3]f32
	defer delete(layer_absorbs)
	layer_dens: [dynamic][]f32
	defer delete(layer_dens)
	layer_depths: [dynamic]f32
	defer delete(layer_depths)
	defer for d in layer_dens {delete(d)}

	src := make([]f32, n)
	defer delete(src)
	neg := make([]f32, n)
	defer delete(neg)

	current_z: f32 = 0
	for item, idx in cfg.order {
		switch item.kind {
		case .Emulsion:
			emu := cfg.emulsions[item.index]
			if opts.grain_radius >= 0 {
				emu.grain_radius = opts.grain_radius
			}
			if opts.grain_sigma >= 0 {
				emu.grain_sigma = opts.grain_sigma
			}
			if opts.sigma_filter >= 0 {
				emu.sigma_filter = opts.sigma_filter
			}
			absorb, expo_w := build_vectors(emu.dye)
			for i in 0 ..< n {
				lum :=
					ray_front[i * 3] * expo_w[0] +
					ray_front[i * 3 + 1] * expo_w[1] +
					ray_front[i * 3 + 2] * expo_w[2]
				src[i] = 1.0 - clamp(lum, 0.0, 1.0)
			}
			ss := f32(opts.supersample)
			r_px := emu.grain_radius * ss
			sig_px := emu.grain_sigma * ss
			sig_f := emu.sigma_filter * ss
			mtf := emu.mtf_blur
			sigma_mtf := r_px / math.SQRT_TWO
			if mtf != nil {
				sigma_mtf = mtf.? * ss
			}
			mtf_max := emu.mtf_blur_max
			if mtf_max == nil {
				mtf_max = mtf
			}
			if mtf_max == nil || mtf_max.? <= mtf.? {
				if sigma_mtf > 0 {
					blurred, ok := gauss_blur_dispatch(&ctx, src, w_sim, h_sim, sigma_mtf)
					if !ok {
						fail("高斯模糊失败")
						os.exit(1)
					}
					defer delete(blurred)
					copy(src, blurred)
				}
			} else if sigma_mtf > 0 {
				blurred, ok := adaptive_blur_dispatch(&ctx, src, w_sim, h_sim, sigma_mtf, mtf_max.? * ss)
				if !ok {
					fail("自适应模糊失败")
					os.exit(1)
				}
				defer delete(blurred)
				copy(src, blurred)
			}
			params := prep_physics(r_px, sig_px, sig_f)
			info(
				fmt.tprintf(
					"[emu %d] dye=[%d, %d, %d]  R=%.3fpx  mtf=%.3fpx",
					idx,
					emu.dye[0],
					emu.dye[1],
					emu.dye[2],
					params.r_px / ss,
					sigma_mtf / ss,
				),
			)
			render_params := Render_Params {
				width      = u32(w_sim),
				height     = u32(h_sim),
				n_samples  = u32(opts.samples),
				seed       = opts.seed + u32(idx) * 19,
				sigma_f    = params.sig_f_px,
				sigma      = params.sig_px,
				r2         = params.r2,
				sigma_ln   = params.sigma_ln,
				mu_ln      = params.mu_ln,
				max_r      = params.max_r,
				ag         = params.ag,
				lambda_fac = params.lambda_fac,
			}
			t = stage_time(fmt.tprintf("[emu %d] CPU 预处理", idx), t)
			if !dispatch_render(&ctx, render_params, src, neg) {
				fail("调度失败")
				os.exit(1)
			}
			t = stage_time(fmt.tprintf("[emu %d] GPU 渲染", idx), t)
			dens := make([]f32, n)
			copy(dens, neg)
			append(&layer_absorbs, absorb)
			append(&layer_dens, dens)
			append(&layer_depths, current_z)
			current_z += 1
			for i in 0 ..< n {
				d := neg[i]
				ray_front[i * 3 + 0] *= 1.0 - absorb[0] * (1.0 - d)
				ray_front[i * 3 + 1] *= 1.0 - absorb[1] * (1.0 - d)
				ray_front[i * 3 + 2] *= 1.0 - absorb[2] * (1.0 - d)
			}
		case .Filter:
			col := cfg.filters[item.index].color
			info(fmt.tprintf("[filter %d] colour=[%d, %d, %d]", idx, col[0], col[1], col[2]))
			for i in 0 ..< n {
				ray_front[i * 3 + 0] *= f32(col[0]) / 255.0
				ray_front[i * 3 + 1] *= f32(col[1]) / 255.0
				ray_front[i * 3 + 2] *= f32(col[2]) / 255.0
			}
		case .Film_Base:
			info(fmt.tprintf("[base] thickness=%.1f", film_thick))
		case .Back:
			info(fmt.tprintf("[back] reflectance=%.4f", back_refl))
		}
	}

	front := make([]f32, n * 3)
	defer delete(front)
	for i in 0 ..< n * 3 {
		front[i] = 1.0
	}
	for l in 0 ..< len(layer_dens) {
		absorb := layer_absorbs[l]
		dens := layer_dens[l]
		for i in 0 ..< n {
			d := dens[i]
			front[i * 3 + 0] *= 1.0 - absorb[0] + absorb[0] * d
			front[i * 3 + 1] *= 1.0 - absorb[1] + absorb[1] * d
			front[i * 3 + 2] *= 1.0 - absorb[2] + absorb[2] * d
		}
	}
	t = stage_time("front 组装", t)

	final := front
	defer if back_refl > EPS {delete(final)}
	if back_refl > EPS {
		front_hdr := make([]f32, n * 3)
		defer delete(front_hdr)
		for i in 0 ..< n * 3 {
			x := clamp(front[i], 0.0, 1.0 - EPS)
			front_hdr[i] = clamp(inverse_reinhard(x), 0.0, 3.0)
		}
		n_layers := len(layer_dens)
		depth := make([]f32, n_layers)
		defer delete(depth)
		last := max(layer_depths[n_layers - 1], 1.0)
		for l in 0 ..< n_layers {
			depth[l] = layer_depths[l] / last * film_thick
		}
		dens_stack := make([]f32, n_layers * n)
		defer delete(dens_stack)
		for l in 0 ..< n_layers {
			copy(dens_stack[l * n:(l + 1) * n], layer_dens[l])
		}
		absorb_stack := make([]f32, n_layers * 3)
		defer delete(absorb_stack)
		for l in 0 ..< n_layers {
			absorb_stack[l * 3 + 0] = layer_absorbs[l][0]
			absorb_stack[l * 3 + 1] = layer_absorbs[l][1]
			absorb_stack[l * 3 + 2] = layer_absorbs[l][2]
		}
		bounced := make([]f32, n * 3)
		defer delete(bounced)
		info(fmt.tprintf("halation 回弹 %d 采样/像素 ...", opts.bounce_samples))
		bounce_params := Bounce_Params {
			width     = u32(w_sim),
			height    = u32(h_sim),
			n_layers  = u32(n_layers),
			n_samples = u32(opts.bounce_samples),
			seed      = 424242,
			film_px   = film_thick,
		}
		t = stage_time("halation 准备", t)
		if !dispatch_bounce(
			&ctx,
			bounce_params,
			front_hdr,
			bounced,
			dens_stack,
			absorb_stack,
			depth,
		) {
			fail("调度失败")
			os.exit(1)
		}
		t = stage_time("halation GPU 回弹", t)
		final = make([]f32, n * 3)
		for i in 0 ..< n * 3 {
			final[i] = clamp(front[i] + bounced[i] * back_refl, 0.0, 1.0)
		}
	}
	success("仿真完成")
	t = stage_time("合成与输出", t)

	out8 := make([]u8, n * 3)
	defer delete(out8)
	for i in 0 ..< n * 3 {
		v := linear_to_srgb(final[i], opts.gamma)
		v = clamp((v - 0.5 + opts.exposure * 0.5) * opts.contrast + 0.5, 0.0, 1.0)
		out8[i] = u8(clamp(255.0 * v, 0.0, 255.0))
	}

	out_w := w_sim
	out_h := h_sim
	png_data := out8
	defer if opts.supersample > 1 {delete(png_data)}
	if opts.supersample > 1 {
		out_w = w_sim / opts.supersample
		out_h = opts.height
		png_data = make([]u8, out_w * out_h * 3)
		stb.resize_uint8(
			raw_data(out8),
			c.int(w_sim),
			c.int(h_sim),
			c.int(w_sim * 3),
			raw_data(png_data),
			c.int(out_w),
			c.int(out_h),
			c.int(out_w * 3),
			3,
		)
	}

	output_cstr := strings.clone_to_cstring(opts.output)
	defer delete(output_cstr)
	if stb.write_png(
		   output_cstr,
		   c.int(out_w),
		   c.int(out_h),
		   3,
		   raw_data(png_data),
		   c.int(out_w * 3),
	   ) ==
	   0 {
		fail(fmt.tprintf("无法写出图像: %s", opts.output))
		os.exit(1)
	}
	success(fmt.tprintf("已保存 %s", opts.output))
	success(fmt.tprintf("总耗时 %s", time.since(start)))
}

