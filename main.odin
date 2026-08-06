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
import "core:sync/chan"
import "core:thread"
import "core:time"
import stb "vendor:stb/image"

Lin_Data :: struct {
	ray_front: []f32,
	resized:   []u8,
	gamma:     f32,
}

lin_task :: proc(data: rawptr, i: int) {
	d := cast(^Lin_Data)data
	d.ray_front[i * 3 + 0] = srgb_to_linear(f32(d.resized[i * 3 + 0]), d.gamma)
	d.ray_front[i * 3 + 1] = srgb_to_linear(f32(d.resized[i * 3 + 1]), d.gamma)
	d.ray_front[i * 3 + 2] = srgb_to_linear(f32(d.resized[i * 3 + 2]), d.gamma)
}

Src_Data :: struct {
	ray_front: []f32,
	src:       []f32,
	expo_w:    [3]f32,
}

src_task :: proc(data: rawptr, i: int) {
	d := cast(^Src_Data)data
	lum :=
		d.ray_front[i * 3] * d.expo_w[0] +
		d.ray_front[i * 3 + 1] * d.expo_w[1] +
		d.ray_front[i * 3 + 2] * d.expo_w[2]
	d.src[i] = 1.0 - clamp(lum, 0.0, 1.0)
}

Front_Update_Data :: struct {
	neg:       []f32,
	ray_front: []f32,
	absorb:    [3]f32,
}

front_update_task :: proc(data: rawptr, i: int) {
	d := cast(^Front_Update_Data)data
	dd := d.neg[i]
	d.ray_front[i * 3 + 0] *= 1.0 - d.absorb[0] * (1.0 - dd)
	d.ray_front[i * 3 + 1] *= 1.0 - d.absorb[1] * (1.0 - dd)
	d.ray_front[i * 3 + 2] *= 1.0 - d.absorb[2] * (1.0 - dd)
}

Filter_Data :: struct {
	ray_front: []f32,
	col:       [3]u8,
}

filter_task :: proc(data: rawptr, i: int) {
	d := cast(^Filter_Data)data
	d.ray_front[i * 3 + 0] *= f32(d.col[0]) / 255.0
	d.ray_front[i * 3 + 1] *= f32(d.col[1]) / 255.0
	d.ray_front[i * 3 + 2] *= f32(d.col[2]) / 255.0
}

Front_Assemble_Data :: struct {
	front:  []f32,
	dens:   []f32,
	absorb: [3]f32,
}

front_assemble_task :: proc(data: rawptr, i: int) {
	d := cast(^Front_Assemble_Data)data
	dd := d.dens[i]
	d.front[i * 3 + 0] *= 1.0 - d.absorb[0] + d.absorb[0] * dd
	d.front[i * 3 + 1] *= 1.0 - d.absorb[1] + d.absorb[1] * dd
	d.front[i * 3 + 2] *= 1.0 - d.absorb[2] + d.absorb[2] * dd
}

Hdr_Data :: struct {
	front:     []f32,
	front_hdr: []f32,
}

hdr_task :: proc(data: rawptr, i: int) {
	d := cast(^Hdr_Data)data
	x := clamp(d.front[i], 0.0, 1.0 - EPS)
	d.front_hdr[i] = clamp(inverse_reinhard(x), 0.0, 3.0)
}

Final_Data :: struct {
	front:     []f32,
	bounced:   []f32,
	final:     []f32,
	back_refl: f32,
}

final_task :: proc(data: rawptr, i: int) {
	d := cast(^Final_Data)data
	d.final[i] = clamp(d.front[i] + d.bounced[i] * d.back_refl, 0.0, 1.0)
}

Out8_Data :: struct {
	final:    []f32,
	out8:     []u8,
	gamma:    f32,
	exposure: f32,
	contrast: f32,
	film:     f32,
}

out8_task :: proc(data: rawptr, i: int) {
	d := cast(^Out8_Data)data
	v := linear_to_srgb(d.final[i], d.gamma)
	v = clamp((v - 0.5 + d.exposure * 0.5) * d.contrast + 0.5, 0.0, 1.0)
	v = filmic_curve(v, d.film)
	d.out8[i] = u8(clamp(255.0 * v, 0.0, 255.0))
}

BLUE :: "\x1b[38;2;138;173;244m"
GREEN :: "\x1b[38;2;166;218;149m"
YELLOW :: "\x1b[38;2;238;212;159m"
RED :: "\x1b[38;2;237;135;150m"
RESET :: "\x1b[0m"

info :: proc(msg: string) {fmt.printfln("%s[i]%s %s", BLUE, RESET, msg)}
success :: proc(msg: string) {fmt.printfln("%s[+]%s %s", GREEN, RESET, msg)}
warn :: proc(msg: string) {fmt.printfln("%s[!]%s %s", YELLOW, RESET, msg)}
fail :: proc(msg: string) {fmt.eprintfln("%s[-]%s %s", RED, RESET, msg)}

stage_time :: proc(label: string, t0: time.Time, verbose := true) -> time.Time {
	if verbose {
		info(fmt.tprintf("%s: %.2fs", label, time.duration_seconds(time.since(t0))))
	}
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
	video:          bool `usage:"视频模式 (输入为视频文件, 输出视频)"`,
	film:           f32 `usage:"胶片 S 曲线强度 (0=关, 1=满, 视频默认 0.5)"`,
	bitrate:        int `usage:"视频平均码率 Mbps (默认 60)"`,
	maxrate:        int `usage:"视频峰值码率 Mbps (默认 100)"`,
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
		if opts.video {
			opts.height = 2160
		} else {
			opts.height = 1080
		}
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
		if opts.supersample == 0 {opts.supersample = 2}
		if opts.gamma == 0 {opts.gamma = 2.4}
		if opts.exposure == 999.0 {opts.exposure = 0.08}
		if opts.contrast == 0 {opts.contrast = 0.92}
		if opts.reflectance < 0 {opts.reflectance = 0.03}
		if opts.thickness < 0 {opts.thickness = 20.0}
	} else {
		if opts.supersample == 0 {opts.supersample = 1}
		if opts.gamma == 0 {opts.gamma = 2.2}
		if opts.exposure == 999.0 {opts.exposure = 0.0}
		if opts.contrast == 0 {opts.contrast = 1.0}
	}

	if opts.video {
		run_video(&opts, device_choice)
		return
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

	if opts.video {
		run_video(&opts, device_choice)
		return
	}
	cfg, cfg_ok := setup_config(&opts, pixels[:int(w) * int(h) * 3], int(w), int(h), false)
	if !cfg_ok {
		fail(fmt.tprintf("无法解析胶片配置: %s", opts.film_cfg))
		os.exit(1)
	}
	defer destroy_film_config(&cfg)
	t = stage_time("配置解析", t)
	h_sim := opts.height * opts.supersample
	w_sim := max(1, int(f32(w) * f32(h_sim) / f32(h) + 0.5))
	ctx: Compute_Context
	if !sim_init(&ctx, u32(w_sim), u32(h_sim), u32(len(cfg.emulsions)), device_choice) {
		fail("渲染器初始化失败")
		os.exit(1)
	}
	defer sim_cleanup(&ctx)
	t = stage_time("渲染器初始化", t)

	out8, out_w, out_h, rok := render_frame(&ctx, &opts, &cfg, pixels[:int(w) * int(h) * 3], int(w), int(h), opts.seed, &t, true)
	if !rok {
		fail("渲染失败")
		os.exit(1)
	}
	defer delete(out8)

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

setup_config :: proc(opts: ^Options, pixels: []u8, w: int, h: int, for_video: bool) -> (Film_Config, bool) {
	if opts.samples == 0 {opts.samples = 400}
	if opts.bounce_samples == 0 {opts.bounce_samples = 400}
	auto_mtf := opts.mtf
	if opts.auto {
		sharpness := content_sharpness(raw_data(pixels), w, h)
		auto_res := compute_auto(sharpness, opts.height)
		gr := auto_res.grain_radius
		sf := auto_res.sigma_filter
		gs := auto_res.grain_sigma
		if for_video {
			gr *= 2.5
			sf *= 4.0
			gs *= 2.0
		}
		info(fmt.tprintf("内容锐度: %.2f", sharpness))
		if opts.grain_radius < 0 {opts.grain_radius = gr}
		if opts.grain_sigma < 0 {opts.grain_sigma = gs}
		if opts.sigma_filter < 0 {opts.sigma_filter = sf}
		if auto_mtf < 0 {
			auto_mtf = auto_res.mtf_abs
			if for_video {
				auto_mtf *= 2.0
			}
		}
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
	cfg: Film_Config
	cfg_ok: bool
	if opts.auto {
		cfg, cfg_ok = parse_film_config_text(cfg_text)
	} else {
		cfg, cfg_ok = parse_film_config(opts.film_cfg)
	}
	if !cfg_ok {
		fail(fmt.tprintf("无法解析胶片配置: %s", opts.film_cfg))
		return {}, false
	}
	if len(cfg.emulsions) == 0 {
		destroy_film_config(&cfg)
		fail("配置中至少需要一个 [[emulsion]] 块")
		return {}, false
	}
	if len(cfg.filters) != len(cfg.emulsions) && len(cfg.filters) != len(cfg.emulsions) - 1 {
		destroy_film_config(&cfg)
		fail("#[[filter]] 必须等于 #[[emulsion]] 或少一个")
		return {}, false
	}
	if opts.mtf >= 0 {
		for &emu in cfg.emulsions {
			emu.mtf_blur = opts.mtf * 0.6
			emu.mtf_blur_max = opts.mtf * 1.4
		}
	}
	return cfg, true
}

render_frame :: proc(
	ctx: ^Compute_Context,
	opts: ^Options,
	cfg: ^Film_Config,
	input: []u8,
	fw: int,
	fh: int,
	frame_seed: u32,
	t: ^time.Time,
	verbose: bool,
) -> ([]u8, int, int, bool) {
	h_sim := opts.height * opts.supersample
	w_sim := max(1, int(f32(fw) * f32(h_sim) / f32(fh) + 0.5))
	resized := make([]u8, w_sim * h_sim * 3)
	defer delete(resized)
	if fw != w_sim || fh != h_sim {
		stb.resize_uint8(
			raw_data(input),
			c.int(fw),
			c.int(fh),
			c.int(fw * 3),
			raw_data(resized),
			c.int(w_sim),
			c.int(h_sim),
			c.int(w_sim * 3),
			3,
		)
		if verbose {
			info(fmt.tprintf("缩放至: %dx%d", w_sim, h_sim))
		}
	} else {
		copy(resized, input[:fw * fh * 3])
	}

	n := w_sim * h_sim
	ray_front := make([]f32, n * 3)
	defer delete(ray_front)
	parallel_for(n, &Lin_Data{ray_front = ray_front, resized = resized, gamma = opts.gamma}, lin_task)
	t^ = stage_time("缩放与线性化", t^, verbose)

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
			parallel_for(n, &Src_Data{ray_front = ray_front, src = src, expo_w = expo_w}, src_task)
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
					blurred, ok := gauss_blur_dispatch(ctx, src, w_sim, h_sim, sigma_mtf)
					if !ok {
						fail("高斯模糊失败")
						return nil, 0, 0, false
					}
					defer delete(blurred)
					copy(src, blurred)
				}
			} else if sigma_mtf > 0 {
				blurred, ok := adaptive_blur_dispatch(ctx, src, w_sim, h_sim, sigma_mtf, mtf_max.? * ss)
				if !ok {
					fail("自适应模糊失败")
					return nil, 0, 0, false
				}
				defer delete(blurred)
				copy(src, blurred)
			}
			params := prep_physics(r_px, sig_px, sig_f)
			if verbose {
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
			}
			render_params := Render_Params {
				width      = u32(w_sim),
				height     = u32(h_sim),
				n_samples  = u32(opts.samples),
				seed       = frame_seed + u32(idx) * 19,
				sigma_f    = params.sig_f_px,
				sigma      = params.sig_px,
				r2         = params.r2,
				sigma_ln   = params.sigma_ln,
				mu_ln      = params.mu_ln,
				max_r      = params.max_r,
				ag         = params.ag,
				lambda_fac = params.lambda_fac,
			}
			t^ = stage_time(fmt.tprintf("[emu %d] CPU 预处理", idx), t^, verbose)
			if !dispatch_render(ctx, render_params, src, neg) {
				fail("调度失败")
				return nil, 0, 0, false
			}
			t^ = stage_time(fmt.tprintf("[emu %d] GPU 渲染", idx), t^, verbose)
			dens := make([]f32, n)
			copy(dens, neg)
			append(&layer_absorbs, absorb)
			append(&layer_dens, dens)
			append(&layer_depths, current_z)
			current_z += 1
			parallel_for(n, &Front_Update_Data{neg = neg, ray_front = ray_front, absorb = absorb}, front_update_task)
		case .Filter:
			col := cfg.filters[item.index].color
			if verbose {
				info(fmt.tprintf("[filter %d] colour=[%d, %d, %d]", idx, col[0], col[1], col[2]))
			}
			parallel_for(n, &Filter_Data{ray_front = ray_front, col = col}, filter_task)
		case .Film_Base:
			if verbose {
				info(fmt.tprintf("[base] thickness=%.1f", film_thick))
			}
		case .Back:
			if verbose {
				info(fmt.tprintf("[back] reflectance=%.4f", back_refl))
			}
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
		parallel_for(n, &Front_Assemble_Data{front = front, dens = dens, absorb = absorb}, front_assemble_task)
	}
	t^ = stage_time("front 组装", t^, verbose)

	final := front
	defer if back_refl > EPS {delete(final)}
	halation_out: []f32
	if back_refl > EPS {
		front_hdr := make([]f32, n * 3)
		defer delete(front_hdr)
		parallel_for(n * 3, &Hdr_Data{front = front, front_hdr = front_hdr}, hdr_task)
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
		if verbose {
			info(fmt.tprintf("halation 回弹 %d 采样/像素 ...", opts.bounce_samples))
		}
		bounce_params := Bounce_Params {
			width     = u32(w_sim),
			height    = u32(h_sim),
			n_layers  = u32(n_layers),
			n_samples = u32(opts.bounce_samples),
			seed      = 424242,
			film_px   = film_thick,
		}
		t^ = stage_time("halation 准备", t^, verbose)
		if !dispatch_bounce(
			ctx,
			bounce_params,
			front_hdr,
			bounced,
			dens_stack,
			absorb_stack,
			depth,
		) {
			fail("调度失败")
			return nil, 0, 0, false
		}

		t^ = stage_time("halation GPU 回弹", t^, verbose)
		final = make([]f32, n * 3)
		parallel_for(n * 3, &Final_Data{front = front, bounced = bounced, final = final, back_refl = back_refl}, final_task)
	}
	if verbose {
		success("仿真完成")
	}
	t^ = stage_time("合成与输出", t^, verbose)

	out8 := make([]u8, n * 3)
	parallel_for(n * 3, &Out8_Data{final = final, out8 = out8, gamma = opts.gamma, exposure = opts.exposure, contrast = opts.contrast, film = opts.film}, out8_task)
	return out8, w_sim, h_sim, true
}

Render_Task :: struct {
	frame: int,
	data:  []u8,
}

Render_Result :: struct {
	frame: int,
	out8:  []u8,
}

Render_Worker :: struct {
	idx:       int,
	tasks:     chan.Chan(Render_Task),
	results:   chan.Chan(Render_Result),
	thread:    ^thread.Thread,
	opts:      ^Options,
	cfg:       ^Film_Config,
	device:    Device_Choice,
	fw:        int,
	fh:        int,
	seed_base: u32,
	w_sim:     u32,
	h_sim:     u32,
	n_layers:  u32,
	ctx:       Compute_Context,
}

render_worker_main :: proc(t: ^thread.Thread) {
	w := cast(^Render_Worker)t.data
	if !sim_init(&w.ctx, w.w_sim, w.h_sim, w.n_layers, w.device) {
		fail("渲染 worker 初始化失败")
		os.exit(1)
	}
	defer sim_cleanup(&w.ctx)
	for {
		task, ok := chan.recv(w.tasks)
		if !ok {
			break
		}
		tt := time.now()
		out8, _, _, rok := render_frame(&w.ctx, w.opts, w.cfg, task.data, w.fw, w.fh, w.seed_base + u32(task.frame) * 6743, &tt, false)
		delete(task.data)
		if !rok {
			fail("帧渲染失败")
			os.exit(1)
		}
		if !chan.send(w.results, Render_Result{frame = task.frame, out8 = out8}) {
			break
		}
	}
}

collect_one :: proc(res: Render_Result, next_seq: ^int, pending: ^map[int][]u8, vout: ^Video_Out) -> bool {
	if res.frame == next_seq^ {
		if !video_send_encode(vout, res.out8) {
			return false
		}
		next_seq^ += 1
		for {
			if out, ok := pending^[next_seq^]; ok {
				if !video_send_encode(vout, out) {
					return false
				}
				delete_key(pending, next_seq^)
				next_seq^ += 1
			} else {
				break
			}
		}
	} else {
		pending^[res.frame] = res.out8
	}
	return true
}

run_video :: proc(opts: ^Options, device_choice: Device_Choice) {
	vinfo, ok := video_probe(opts.input)
	if !ok {
		fail(fmt.tprintf("无法探测视频: %s", opts.input))
		os.exit(1)
	}
	info(fmt.tprintf("视频: %dx%d @%.2f fps, %d 帧", vinfo.width, vinfo.height, vinfo.fps, vinfo.n_frames))
	opts.supersample = 1
	if opts.samples == 0 {opts.samples = 128}
	if opts.bounce_samples == 0 {opts.bounce_samples = 128}
	if opts.film == 0 {opts.film = 0.5}
	if opts.bitrate <= 0 {opts.bitrate = 60}
	if opts.maxrate <= 0 {opts.maxrate = 100}

	vin, ok2 := video_in_start(opts.input, vinfo)
	if !ok2 {
		fail("视频解码器启动失败 (需要 ffmpeg 在 PATH 中)")
		os.exit(1)
	}
	frame_buf := make([]u8, vinfo.width * vinfo.height * 3)
	defer delete(frame_buf)
	if !video_next_frame(vin, frame_buf) {
		fail("无法解码第一帧")
		os.exit(1)
	}
	cfg, cfg_ok := setup_config(opts, frame_buf, vinfo.width, vinfo.height, true)
	if !cfg_ok {
		os.exit(1)
	}
	defer destroy_film_config(&cfg)

	h_sim := opts.height * opts.supersample
	w_sim := max(1, int(f32(vinfo.width) * f32(h_sim) / f32(vinfo.height) + 0.5))
	info(fmt.tprintf("输出: %dx%d @%.2f fps", w_sim, h_sim, vinfo.fps))

	vout, ok4 := video_out_start(opts.output, w_sim, h_sim, vinfo.fps, opts.input, opts.bitrate, opts.maxrate)
	if !ok4 {
		fail("视频编码器启动失败 (需要 ffmpeg 在 PATH 中, 且显卡支持 NVENC)")
		os.exit(1)
	}

	results, rerr := chan.create_buffered(chan.Chan(Render_Result), 8, context.allocator)
	if rerr != nil {
		fail("创建渲染结果队列失败")
		os.exit(1)
	}
	defer chan.destroy(results)
	n_workers := 3
	workers := make([]^Render_Worker, n_workers)
	defer {
		for w in workers {
			if w != nil {
				chan.close(w.tasks)
				thread.destroy(w.thread)
				chan.destroy(w.tasks)
				free(w)
			}
		}
		delete(workers)
	}
	for i in 0 ..< n_workers {
		w := new(Render_Worker)
		w.idx = i
		w.opts = opts
		w.cfg = &cfg
		w.device = device_choice
		w.fw = vinfo.width
		w.fh = vinfo.height
		w.seed_base = opts.seed
		w.w_sim = u32(w_sim)
		w.h_sim = u32(h_sim)
		w.n_layers = u32(len(cfg.emulsions))
		w.results = results
		tasks, terr := chan.create_buffered(chan.Chan(Render_Task), 4, context.allocator)
		if terr != nil {
			fail("创建渲染任务队列失败")
			os.exit(1)
		}
		w.tasks = tasks
		w.thread = thread.create(render_worker_main)
		w.thread.data = w
		thread.start(w.thread)
		workers[i] = w
	}

	start := time.now()
	total := vinfo.n_frames
	next_seq := 0
	pending: map[int][]u8
	defer delete(pending)
	n_encoded := 0
	actual := 0
	for f := 0; total < 0 || f < total; f += 1 {
		if f > 0 {
			if !video_next_frame(vin, frame_buf) {
				break
			}
		}
		task := Render_Task{frame = f, data = make([]u8, len(frame_buf))}
		copy(task.data, frame_buf)
		if !chan.send(workers[f % n_workers].tasks, task) {
			fail("渲染队列关闭")
			os.exit(1)
		}
		actual = f + 1
		for {
			res, ok := chan.try_recv(results)
			if !ok {
				break
			}
			if !collect_one(res, &next_seq, &pending, vout) {
				fail("编码输出失败")
				os.exit(1)
			}
			n_encoded = next_seq
			if n_encoded % 10 == 0 {
				elapsed := time.duration_seconds(time.since(start))
				avg := f32(elapsed) / f32(max(1, n_encoded))
				eta := avg * f32(max(0, actual - n_encoded))
				info(fmt.tprintf("帧 %d/%d (%.2fs/帧, 已用 %.1fs, 预计剩余 %.1fs)", n_encoded, actual, avg, elapsed, eta))
			}
		}
	}
	for next_seq < actual {
		res, ok := chan.recv(results)
		if !ok {
			fail("渲染结果队列关闭")
			os.exit(1)
		}
		if !collect_one(res, &next_seq, &pending, vout) {
			fail("编码输出失败")
			os.exit(1)
		}
		n_encoded = next_seq
		if n_encoded % 10 == 0 || next_seq == actual {
			elapsed := time.duration_seconds(time.since(start))
			avg := f32(elapsed) / f32(max(1, n_encoded))
			eta := avg * f32(max(0, actual - n_encoded))
			info(fmt.tprintf("帧 %d/%d (%.2fs/帧, 已用 %.1fs, 预计剩余 %.1fs)", n_encoded, actual, avg, elapsed, eta))
		}
	}

	video_in_finish(vin)
	video_out_finish(vout)
	success(fmt.tprintf("视频处理完成: %d 帧, 总耗时 %s", n_encoded, time.since(start)))
}
