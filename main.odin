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

wstring_to_utf8_own :: proc(s: []u16) -> string {
	n := 0
	for i := 0; i < len(s); i += 1 {
		c := s[i]
		if c < 0x80 {
			n += 1
		} else if c < 0x800 {
			n += 2
		} else if c >= 0xD800 && c < 0xDC00 && i + 1 < len(s) && s[i + 1] >= 0xDC00 && s[i + 1] < 0xE000 {
			n += 4
			i += 1
		} else {
			n += 3
		}
	}
	buf := make([]u8, n)
	k := 0
	for i := 0; i < len(s); i += 1 {
		c := s[i]
		if c < 0x80 {
			buf[k] = u8(c)
			k += 1
		} else if c < 0x800 {
			buf[k] = u8(0xC0 | (c >> 6))
			buf[k + 1] = u8(0x80 | (c & 0x3F))
			k += 2
		} else if c >= 0xD800 && c < 0xDC00 && i + 1 < len(s) && s[i + 1] >= 0xDC00 && s[i + 1] < 0xE000 {
			cp := u32(c - 0xD800) << 10 | u32(s[i + 1] - 0xDC00) + 0x10000
			buf[k] = u8(0xF0 | (cp >> 18))
			buf[k + 1] = u8(0x80 | ((cp >> 12) & 0x3F))
			buf[k + 2] = u8(0x80 | ((cp >> 6) & 0x3F))
			buf[k + 3] = u8(0x80 | (cp & 0x3F))
			k += 4
			i += 1
		} else {
			buf[k] = u8(0xE0 | (c >> 12))
			buf[k + 1] = u8(0x80 | ((c >> 6) & 0x3F))
			buf[k + 2] = u8(0x80 | (c & 0x3F))
			k += 3
		}
	}
	return string(buf)
}

parse_cmdline_wide :: proc(cmd: [^]u16) -> []string {
	args := make([dynamic]string)
	cur: [dynamic]u16
	in_quotes := false
	i := 0
	for {
		c := cmd[i]
		if c == 0 {
			break
		}
		if c == '"' {
			in_quotes = !in_quotes
			i += 1
			continue
		}
		if (c == ' ' || c == '\t') && !in_quotes {
			if len(cur) > 0 {
				append(&args, wstring_to_utf8_own(cur[:]))
				clear(&cur)
			}
			i += 1
			continue
		}
		append(&cur, c)
		i += 1
	}
	if len(cur) > 0 {
		append(&args, wstring_to_utf8_own(cur[:]))
	}
	return args[:]
}

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

// Schwarzschild reciprocity failure: layer sensitivity loss factors (B, G, R layers)
recip_diff_for :: proc(layer: int) -> f32 {
	switch layer % 3 {
	case 0:
		return 0.8
	case 1:
		return 0.9
	}
	return 1.0
}

Recip_Data :: struct {
	src: []f32,
	p:   f32,
}

recip_task :: proc(data: rawptr, i: int) {
	d := cast(^Recip_Data)data
	base := 1.0 - d.src[i]
	if base > 0.001 && base < 0.999 {
		expo := math.pow(base, 1.0 - d.p)
		d.src[i] = 1.0 - math.pow(base, expo)
	}
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

Min_Data :: struct {
	front:  []f32,
	smooth: []f32,
}

min_task :: proc(data: rawptr, i: int) {
	d := cast(^Min_Data)data
	if d.smooth[i] < d.front[i] {
		d.front[i] = d.smooth[i]
	}
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
	final:         []f32,
	out8:          []u8,
	gamma:         f32,
	exposure:      f32,
	contrast:      f32,
	film:          f32,
	print_toe:     f32,
	print_shoulder: f32,
	sat_lo:         f32,
	sat_hi:         f32,
	cross:          f32,
	negative:       bool,
}

out8_task :: proc(data: rawptr, i: int) {
	d := cast(^Out8_Data)data
	r := linear_to_srgb(d.final[i * 3 + 0], d.gamma)
	g := linear_to_srgb(d.final[i * 3 + 1], d.gamma)
	b := linear_to_srgb(d.final[i * 3 + 2], d.gamma)
	r = clamp((r - 0.5 + d.exposure * 0.5) * d.contrast + 0.5, 0.0, 1.0)
	g = clamp((g - 0.5 + d.exposure * 0.5) * d.contrast + 0.5, 0.0, 1.0)
	b = clamp((b - 0.5 + d.exposure * 0.5) * d.contrast + 0.5, 0.0, 1.0)
	luma := 0.2126 * r + 0.7152 * g + 0.0722 * b
	if d.negative {
		toe := d.print_toe
		if toe < 0 {toe = 0.3}
		shoulder := d.print_shoulder
		if shoulder < 0 {shoulder = 0.3}
		y := print_curve(luma, toe, shoulder)
		r = y + (r - luma)
		g = y + (g - luma)
		b = y + (b - luma)
		mask_w := (1.0 - y) * (1.0 - y) * 0.5
		r = r + mask_w * 0.12
		b = b - mask_w * 0.08
		nr := r * 1.02 - g * 0.02
		nb := b * 0.98 + r * 0.02
		r = nr
		b = nb
	}
	sat_w := 1.0 - d.sat_lo * (1.0 - luma) * (1.0 - luma) - d.sat_hi * luma * luma
	r = luma + (r - luma) * sat_w
	g = luma + (g - luma) * sat_w
	b = luma + (b - luma) * sat_w
	if d.film > 0 {
		luma := 0.2126 * r + 0.7152 * g + 0.0722 * b
		y := filmic_curve(luma, d.film)
		tint := d.film * 0.04
		r = y + (r - luma) + (y - luma) * tint
		g = y + (g - luma)
		b = y + (b - luma) - (y - luma) * tint * 0.8
		cr := 1.0 - d.cross
		rr := r * cr + (g + b) * d.cross * 0.5
		gg := g * cr + (r + b) * d.cross * 0.5
		bb := b * cr + (r + g) * d.cross * 0.5
		r = rr
		g = gg
		b = bb
	}
	d.out8[i * 3 + 0] = u8(clamp(255.0 * r, 0.0, 255.0))
	d.out8[i * 3 + 1] = u8(clamp(255.0 * g, 0.0, 255.0))
	d.out8[i * 3 + 2] = u8(clamp(255.0 * b, 0.0, 255.0))
}

BLUE :: "\x1b[38;2;138;173;244m"
GREEN :: "\x1b[38;2;166;218;149m"
YELLOW :: "\x1b[38;2;238;212;159m"
RED :: "\x1b[38;2;237;135;150m"
RESET :: "\x1b[0m"

info :: proc(msg: string) {fmt.printfln("%s[i]%s %s", BLUE, RESET, msg)}
success :: proc(msg: string) {fmt.printfln("%s[+]%s %s", GREEN, RESET, msg)}
warn :: proc(msg: string) {fmt.printfln("%s[!]%s %s", YELLOW, RESET, msg)}

progress_line :: proc(n: int, total: int, start: time.Time) {
	elapsed := f32(time.duration_seconds(time.since(start)))
	avg := elapsed / f32(max(1, n))
	fps: f32 = 0
	if avg > 0 {
		fps = 1.0 / avg
	}
	if total > 0 {
		eta := avg * f32(max(0, total - n))
		fmt.printf("\r\x1b[2K%s[i]%s frame %d/%d (%.2ffps, elapsed %.1fs, ETA %.1fs)", BLUE, RESET, n, total, fps, elapsed, eta)
	} else {
		fmt.printf("\r\x1b[2K%s[i]%s frame %d (%.2ffps, elapsed %.1fs)", BLUE, RESET, n, fps, elapsed)
	}
}
fail :: proc(msg: string) {fmt.eprintfln("%s[-]%s %s", RED, RESET, msg)}

stage_time :: proc(label: string, t0: time.Time, verbose := true) -> time.Time {
	if verbose {
		info(fmt.tprintf("%s: %.2fs", label, time.duration_seconds(time.since(t0))))
	}
	return time.now()
}

Options :: struct {
	height:         int,
	supersample:    int,
	samples:        int,
	bounce_samples: int,
	gamma:          f32,
	mtf:            f32,
	exposure:       f32,
	contrast:       f32,
	reflectance:    f32,
	thickness:      f32,
	grain_radius:   f32,
	grain_sigma:    f32,
	sigma_filter:   f32,
	seed:           u32,
	device:         string,
	film:           f32,
	print_toe:      f32,
	print_shoulder: f32,
	sat_lo:         f32,
	sat_hi:         f32,
	cross:          f32,
	qp:             int,
	reciprocity:    f32,
	negative:       bool,
	mode:           string,
}

Cli_Options :: struct {
	input:  string `usage:"Input file path (photo or video)" args:"required"`,
	output: string `usage:"Output file path; extension decides format: photo .png/.jpg, video .mp4/.mov/.mkv" args:"required"`,
	auto:   bool   `usage:"Auto mode (content-adaptive, default)"`,
	config: string `usage:"JSON config file path (custom mode)"`,
	mode:   string `usage:"Auto mode film type: color (default) | bw"`,
}

File_Type :: enum {
	Photo,
	Video,
	Unknown,
}

detect_file_type :: proc(path: string) -> File_Type {
	ext := strings.to_lower(filepath.ext(path))
	defer delete(ext)
	switch ext {
	case ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif":
		return .Photo
	case ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".ts", ".m2ts", ".flv", ".wmv", ".mpg", ".mpeg":
		return .Video
	}
	return .Unknown
}

read_file_wide :: proc(path: string) -> ([]u8, bool) {
	wpath := utf8_to_wstring_own(path)
	defer delete(wpath)
	handle := windows.CreateFileW(
		cast(windows.LPCWSTR)raw_data(wpath),
		windows.GENERIC_READ,
		windows.FILE_SHARE_READ,
		nil,
		windows.OPEN_EXISTING,
		0,
		nil,
	)
	if handle == windows.INVALID_HANDLE_VALUE {
		return {}, false
	}
	defer windows.CloseHandle(handle)
	size: windows.LARGE_INTEGER
	if !windows.GetFileSizeEx(handle, &size) {
		return {}, false
	}
	buf := make([]u8, int(size))
	read: u32
	if !windows.ReadFile(handle, raw_data(buf), u32(len(buf)), &read, nil) || read != u32(len(buf)) {
		delete(buf)
		return {}, false
	}
	return buf, true
}

write_image :: proc(path: string, data: []u8, w: int, h: int) -> bool {
	wpath := utf8_to_wstring_own(path)
	defer delete(wpath)
	handle := windows.CreateFileW(
		cast(windows.LPCWSTR)raw_data(wpath),
		windows.GENERIC_WRITE,
		windows.FILE_SHARE_READ,
		nil,
		windows.CREATE_ALWAYS,
		0,
		nil,
	)
	if handle == windows.INVALID_HANDLE_VALUE {
		return false
	}
	defer windows.CloseHandle(handle)
	write_file_cb :: proc "c" (ctx: rawptr, data: rawptr, size: c.int) {
		h := cast(windows.HANDLE)ctx
		written: u32
		windows.WriteFile(h, data, u32(size), &written, nil)
	}
	ext := strings.to_lower(filepath.ext(path))
	defer delete(ext)
	switch ext {
	case ".png":
		return stb.write_png_to_func(write_file_cb, handle, c.int(w), c.int(h), 3, raw_data(data), c.int(w * 3)) != 0
	case ".jpg", ".jpeg":
		return stb.write_jpg_to_func(write_file_cb, handle, c.int(w), c.int(h), 3, raw_data(data), 90) != 0
	case ".bmp":
		return stb.write_bmp_to_func(write_file_cb, handle, c.int(w), c.int(h), 3, raw_data(data)) != 0
	}
	return false
}

parse_device_choice :: proc(device: string) -> (Device_Choice, bool) {
	choice: Device_Choice
	switch device {
	case "auto":
		choice.kind = .Auto
	case "cpu":
		choice.kind = .Cpu
	case "cuda":
		choice.kind = .Cuda
		choice.ordinal = 0
	case:
		if strings.has_prefix(device, "cuda:") {
			n, ok := strconv.parse_int(device[5:])
			if !ok || n < 0 {
				return {}, false
			}
			choice.kind = .Cuda
			choice.ordinal = n
		} else {
			return {}, false
		}
	}
	return choice, true
}

main :: proc() {
	when ODIN_OS == .Windows {
		windows.SetConsoleOutputCP(windows.CODEPAGE(65001))
		windows.SetConsoleCP(windows.CODEPAGE(65001))
	}
	start := time.now()
	cli: Cli_Options
	unicode_args := parse_cmdline_wide(cast([^]u16)windows.GetCommandLineW())
	defer {
		for a in unicode_args {
			delete(a)
		}
		delete(unicode_args)
	}
	if err := flags.parse(&cli, unicode_args[1:], style = .Unix); err != nil {
		if _, is_help := err.(flags.Help_Request); is_help {
			flags.write_usage(os.to_stream(os.stdout), Cli_Options, filepath.base(os.args[0]), .Unix)
			os.exit(0)
		}
		fail(fmt.tprintf("Failed to parse arguments: %v", err))
		os.exit(1)
	}
	if cli.auto && cli.config != "" {
		fail("--auto and --config cannot be used together")
		os.exit(1)
	}
	if cli.mode != "" && cli.mode != "color" && cli.mode != "bw" {
		fail("--mode must be \"color\" or \"bw\"")
		os.exit(1)
	}
	in_type := detect_file_type(cli.input)
	out_type := detect_file_type(cli.output)
	if in_type == .Unknown {
		fail(fmt.tprintf("Unrecognized input file type: %s (photos .png/.jpg/.bmp, videos .mp4/.mov/.mkv/.webm etc.)", cli.input))
		os.exit(1)
	}
	if out_type == .Unknown {
		fail(fmt.tprintf("Unrecognized output file extension: %s (photos .png/.jpg/.bmp, videos .mp4/.mov/.mkv/.webm)", cli.output))
		os.exit(1)
	}
	if in_type == .Photo && out_type == .Video {
		fail(fmt.tprintf("Input is a photo but output has a video extension: %s", cli.output))
		os.exit(1)
	}
	if in_type == .Video && out_type == .Photo {
		fail(fmt.tprintf("Input is a video but output has a photo extension: %s", cli.output))
		os.exit(1)
	}
	opts := default_options()
	if cli.mode != "" {
		opts.mode = cli.mode
	}

	if in_type == .Video {
		run_video(&opts, &cli)
		return
	}
	if opts.height <= 0 {
		opts.height = 1080
	}
	input_bytes, iok := read_file_wide(cli.input)
	if !iok {
		fail(fmt.tprintf("Failed to read image: %s", cli.input))
		os.exit(1)
	}
	defer delete(input_bytes)
	w, h, ch: c.int
	t := time.now()
	pixels := stb.load_from_memory(raw_data(input_bytes), c.int(len(input_bytes)), &w, &h, &ch, 3)
	if pixels == nil {
		fail(fmt.tprintf("Failed to read image: %s", cli.input))
		os.exit(1)
	}
	defer stb.image_free(pixels)
	info(fmt.tprintf("Input: %s (%dx%d)", cli.input, w, h))
	t = stage_time("Image load", t)

	cfg, cfg_ok := resolve_config(&opts, cli.config != "", pixels[:int(w) * int(h) * 3], int(w), int(h), false, cli.config, len(input_bytes))
	if !cfg_ok {
		os.exit(1)
	}
	defer destroy_film_config(&cfg)
	t = stage_time("Config parse", t)
	device_choice, dok := parse_device_choice(opts.device)
	if !dok {
		fail("--device format: auto | cpu | cuda | cuda:N (config \"device\" field)")
		os.exit(1)
	}
	h_sim := opts.height * opts.supersample
	w_sim := max(1, int(f32(w) * f32(h_sim) / f32(h) + 0.5))
	ctx: Compute_Context
	if !sim_init(&ctx, u32(w_sim), u32(h_sim), u32(len(cfg.emulsions)), device_choice) {
		fail("Renderer initialization failed")
		os.exit(1)
	}
	defer sim_cleanup(&ctx)
	t = stage_time("Renderer init", t)

	out8, out_w, out_h, rok := render_frame(&ctx, &opts, &cfg, pixels[:int(w) * int(h) * 3], int(w), int(h), opts.seed, 0, &t, true)
	if !rok {
		fail("Render failed")
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

	if !write_image(cli.output, png_data, out_w, out_h) {
		fail(fmt.tprintf("Failed to write image: %s", cli.output))
		os.exit(1)
	}
	success(fmt.tprintf("Saved %s", cli.output))
	success(fmt.tprintf("Total time %s", time.since(start)))
}

Emu_Prep :: struct {
	dye:       [3]u8,
	absorb:    [3]f32,
	expo_w:    [3]f32,
	params:    Render_Params,
	sigma_mtf: f32,
	mtf_max:   f32,
	r_px:      f32,
}

prep_emulsion :: proc(opts: ^Options, emu_in: Emulsion_Cfg, ss: f32, frame_seed: u32, idx: int, frame_idx: int, w_sim: int, h_sim: int) -> Emu_Prep {
	emu := emu_in
	if opts.mtf >= 0 {
		emu.mtf_blur = opts.mtf * 0.6
		emu.mtf_blur_max = opts.mtf * 1.4
	}
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
	max_v := f32(-1)
	if mtf_max != nil {
		max_v = mtf_max.? * ss
	}
	params := prep_physics(r_px, sig_px, sig_f)
	return Emu_Prep {
		dye       = emu.dye,
		absorb    = absorb,
		expo_w    = expo_w,
		sigma_mtf = sigma_mtf,
		mtf_max   = max_v,
		r_px      = r_px,
		params    = Render_Params {
			width       = u32(w_sim),
			height      = u32(h_sim),
			n_samples   = u32(opts.samples),
			seed        = frame_seed + u32(idx) * 19,
			sigma_f     = params.sig_f_px,
			sigma       = params.sig_px,
			r2          = params.r2,
			sigma_ln    = params.sigma_ln,
			mu_ln       = params.mu_ln,
			max_r       = params.max_r,
			ag          = params.ag,
			lambda_fac  = params.lambda_fac,
			frame_off_x = math.sin(f32(frame_idx) * 0.7) * 0.12,
			frame_off_y = math.cos(f32(frame_idx) * 0.9) * 0.12,
		},
	}
}

resolve_config :: proc(
	opts: ^Options,
	config_mode: bool,
	pixels: []u8,
	w: int,
	h: int,
	for_video: bool,
	config_path: string,
	file_size: int,
) -> (cfg: Film_Config, ok: bool) {
	if config_mode {
		parsed, parsed_opts, pok := parse_config_file(config_path)
		if !pok {
			fail(fmt.tprintf("Failed to parse config file: %s (JSON format required)", config_path))
			return {}, false
		}
		opts^ = parsed_opts
		cfg = parsed
	} else {
		auto_cfg, aok := build_auto_config_from_pixels(opts, pixels, w, h, for_video, file_size)
		if !aok {
			return {}, false
		}
		cfg = auto_cfg
	}
	if for_video {
		if opts.samples == 0 {opts.samples = 128}
		if opts.bounce_samples == 0 {opts.bounce_samples = 128}
		if opts.film == 0 {opts.film = 0.5}
	} else {
		if opts.samples == 0 {opts.samples = 400}
		if opts.bounce_samples == 0 {opts.bounce_samples = 400}
	}
	if opts.supersample == 0 {opts.supersample = 1}
	if opts.gamma == 0 {opts.gamma = 2.2}
	if !validate_film_config(opts, &cfg) {
		destroy_film_config(&cfg)
		return {}, false
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
	frame_idx: int,
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
			info(fmt.tprintf("Resized to: %dx%d", w_sim, h_sim))
		}
	} else {
		copy(resized, input[:fw * fh * 3])
	}
	if ctx.mode == .Cuda {
		return render_frame_cuda(ctx, opts, cfg, resized, w_sim, h_sim, frame_seed, frame_idx, t, verbose)
	}

	n := w_sim * h_sim
	ray_front := make([]f32, n * 3)
	defer delete(ray_front)
	parallel_for(n, &Lin_Data{ray_front = ray_front, resized = resized, gamma = opts.gamma}, lin_task)
	t^ = stage_time("Resize & linearize", t^, verbose)

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
			prep := prep_emulsion(opts, cfg.emulsions[item.index], f32(opts.supersample), frame_seed, idx, frame_idx, w_sim, h_sim)
			parallel_for(n, &Src_Data{ray_front = ray_front, src = src, expo_w = prep.expo_w}, src_task)
			if opts.reciprocity > 0 {
				p := 1.0 - opts.reciprocity * 0.2 * recip_diff_for(idx)
				parallel_for(n, &Recip_Data{src = src, p = p}, recip_task)
			}
			if prep.mtf_max < 0 || prep.mtf_max <= prep.sigma_mtf {
				if prep.sigma_mtf > 0 {
					blurred := gauss_blur(src, w_sim, h_sim, prep.sigma_mtf)
					defer delete(blurred)
					copy(src, blurred)
				}
			} else if prep.sigma_mtf > 0 {
				blurred := adaptive_blur(src, w_sim, h_sim, prep.sigma_mtf, prep.mtf_max)
				defer delete(blurred)
				copy(src, blurred)
			}
			if verbose {
				info(
					fmt.tprintf(
						"[emu %d] dye=[%d, %d, %d]  R=%.3fpx  mtf=%.3fpx",
						idx,
						prep.dye[0],
						prep.dye[1],
						prep.dye[2],
						prep.r_px / f32(opts.supersample),
						prep.sigma_mtf / f32(opts.supersample),
					),
				)
			}
			t^ = stage_time(fmt.tprintf("[emu %d] CPU preprocess", idx), t^, verbose)
			if !cpu_dispatch_render(ctx, prep.params, src, neg) {
				fail("Dispatch failed")
				return nil, 0, 0, false
			}
			t^ = stage_time(fmt.tprintf("[emu %d] GPU render", idx), t^, verbose)
			dens := make([]f32, n)
			copy(dens, neg)
			append(&layer_absorbs, prep.absorb)
			append(&layer_dens, dens)
			append(&layer_depths, current_z)
			current_z += 1
			parallel_for(n, &Front_Update_Data{neg = neg, ray_front = ray_front, absorb = prep.absorb}, front_update_task)
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
	front_smooth := make([]f32, n * 3)
	defer delete(front_smooth)
	for i in 0 ..< n * 3 {
		front[i] = 1.0
		front_smooth[i] = 1.0
	}
	for l in 0 ..< len(layer_dens) {
		absorb := layer_absorbs[l]
		dens := layer_dens[l]
		parallel_for(n, &Front_Assemble_Data{front = front, dens = dens, absorb = absorb}, front_assemble_task)
		dens_smooth := gauss_blur(dens, w_sim, h_sim, GRAIN_SMOOTH_SIGMA)
		defer delete(dens_smooth)
		parallel_for(n, &Front_Assemble_Data{front = front_smooth, dens = dens_smooth, absorb = absorb}, front_assemble_task)
	}
	parallel_for(n * 3, &Min_Data{front = front, smooth = front_smooth}, min_task)
	t^ = stage_time("Front assembly", t^, verbose)

	final := front
	defer if back_refl > EPS {delete(final)}
	if back_refl > EPS {
		front_hdr := make([]f32, n * 3)
		defer delete(front_hdr)
		parallel_for(n * 3, &Hdr_Data{front = front, front_hdr = front_hdr}, hdr_task)
		n_layers := len(layer_dens)
		depth := make([]f32, n_layers)
		defer delete(depth)
		last := max(layer_depths[n_layers - 1], 1.0)
		for l in 0 ..< n_layers {
			depth[l] = layer_depths[l] / last * film_thick / HALATION_SCALE
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
		bw := max(1, w_sim / HALATION_SCALE)
		bh := max(1, h_sim / HALATION_SCALE)
		front_low := box_downsample(front_hdr, w_sim, h_sim, 3, bw, bh)
		defer delete(front_low)
		dens_low := box_downsample_planes(dens_stack, w_sim, h_sim, n_layers, bw, bh)
		defer delete(dens_low)
		bounced_low := make([]f32, bw * bh * 3)
		defer delete(bounced_low)
		if verbose {
			info(fmt.tprintf("Halation bounce %d samples/pixel @%dx%d ...", opts.bounce_samples, bw, bh))
		}
		bounce_params := Bounce_Params {
			width     = u32(bw),
			height    = u32(bh),
			n_layers  = u32(n_layers),
			n_samples = u32(opts.bounce_samples),
			seed      = 424242,
			film_px   = film_thick / HALATION_SCALE,
		}
		t^ = stage_time("Halation prep", t^, verbose)
		if !dispatch_bounce(
			ctx,
			bounce_params,
			front_low,
			bounced_low,
			dens_low,
			absorb_stack,
			depth,
		) {
			fail("Dispatch failed")
			return nil, 0, 0, false
		}

		t^ = stage_time("Halation GPU bounce", t^, verbose)
		bounced := bilinear_upsample(bounced_low, bw, bh, 3, w_sim, h_sim)
		defer delete(bounced)
		final = make([]f32, n * 3)
		parallel_for(n * 3, &Final_Data{front = front, bounced = bounced, final = final, back_refl = back_refl}, final_task)
	}
	if verbose {
		success("Simulation complete")
	}
	t^ = stage_time("Compose & output", t^, verbose)

	out8 := make([]u8, n * 3)
	pt := opts.print_toe
	if pt < 0 {pt = 0.35}
	ps := opts.print_shoulder
	if ps < 0 {ps = 0.35}
	sl := opts.sat_lo
	if sl < 0 {sl = 0.15}
	sh := opts.sat_hi
	if sh < 0 {sh = 0.2}
	cr := opts.cross
	if cr < 0 {cr = 0.05}
	parallel_for(n, &Out8_Data{final = final, out8 = out8, gamma = opts.gamma, exposure = opts.exposure, contrast = opts.contrast, film = opts.film, print_toe = pt, print_shoulder = ps, sat_lo = sl, sat_hi = sh, cross = cr, negative = opts.negative}, out8_task)
	return out8, w_sim, h_sim, true
}

Render_Task :: struct {
	frame:  int,
	data:   []u8,
	auto_p: Auto_Params,
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
	if !sim_init(&w.ctx, w.w_sim, w.h_sim, w.n_layers, w.device, w.idx == 0) {
		fail("Render worker initialization failed")
		os.exit(1)
	}
	defer sim_cleanup(&w.ctx)
	for {
		task, ok := chan.recv(w.tasks)
		if !ok {
			break
		}
		tt := time.now()
		opts_copy := w.opts^
		apply_auto_params(&opts_copy, task.auto_p)
		out8, _, _, rok := render_frame(&w.ctx, &opts_copy, w.cfg, task.data, w.fw, w.fh, w.seed_base, task.frame, &tt, false)
		delete(task.data)
		if !rok {
			fail("Frame render failed")
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

run_video :: proc(opts: ^Options, cli: ^Cli_Options) {
	vinfo, ok := video_probe(cli.input)
	if !ok {
		fail(fmt.tprintf("Failed to probe video: %s", cli.input))
		os.exit(1)
	}
	info(fmt.tprintf("Video: %dx%d @%.2f fps, %d frames", vinfo.width, vinfo.height, vinfo.fps, vinfo.n_frames))
	opts.supersample = 1
	if opts.height <= 0 {opts.height = 2160}
	if opts.qp <= 0 {opts.qp = 28}

	vin, ok2 := video_in_start(cli.input, vinfo)
	if !ok2 {
		fail("Failed to start video decoder (ffmpeg required in PATH)")
		os.exit(1)
	}
	frame_buf := make([]u8, vinfo.width * vinfo.height * 3)
	defer delete(frame_buf)
	if !video_next_frame(vin, frame_buf) {
		fail("Failed to decode first frame")
		os.exit(1)
	}
	cfg, cfg_ok := resolve_config(opts, cli.config != "", frame_buf, vinfo.width, vinfo.height, true, cli.config, 0)
	if !cfg_ok {
		os.exit(1)
	}
	defer destroy_film_config(&cfg)
	auto_adapt := cli.config == ""
	ap_cur, ap_tgt: Auto_Params
	ap_prog: f32 = 1.0
	bl_avg, bl_lo, bl_hi: f32
	persist := 0
	if auto_adapt {
		ap_cur = Auto_Params {
			grain_radius   = opts.grain_radius,
			grain_sigma    = opts.grain_sigma,
			sigma_filter   = opts.sigma_filter,
			film           = opts.film,
			print_toe      = opts.print_toe,
			print_shoulder = opts.print_shoulder,
			sat_lo         = opts.sat_lo,
			sat_hi         = opts.sat_hi,
			cross          = opts.cross,
			exposure       = opts.exposure,
			contrast       = opts.contrast,
			valid          = true,
		}
		if len(cfg.emulsions) > 0 && cfg.emulsions[0].mtf_blur != nil {
			ap_cur.mtf = cfg.emulsions[0].mtf_blur.? / 0.6
		}
		ap_tgt = ap_cur
		bl_avg, bl_lo, bl_hi = image_stats(frame_buf, vinfo.width, vinfo.height)
	}
	trans_frames := max(3, int(vinfo.fps * 0.5))
	device_choice, dok := parse_device_choice(opts.device)
	if !dok {
		fail("--device format: auto | cpu | cuda | cuda:N (config \"device\" field)")
		os.exit(1)
	}

	h_sim := opts.height * opts.supersample
	w_sim := max(1, int(f32(vinfo.width) * f32(h_sim) / f32(vinfo.height) + 0.5))
	info(fmt.tprintf("Output: %dx%d @%.2f fps", w_sim, h_sim, vinfo.fps))

	vout, ok4 := video_out_start(cli.output, w_sim, h_sim, vinfo.fps, cli.input, opts.qp)
	if !ok4 {
		fail("Failed to start video encoder (ffmpeg in PATH and NVENC-capable GPU required)")
		os.exit(1)
	}

	results, rerr := chan.create_buffered(chan.Chan(Render_Result), 8, context.allocator)
	if rerr != nil {
		fail("Failed to create render result queue")
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
			fail("Failed to create render task queue")
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
	last_progress := 0
	actual := 0
	for f := 0; total < 0 || f < total; f += 1 {
		if f > 0 {
			if !video_next_frame(vin, frame_buf) {
				break
			}
		}
		ap_frame := ap_tgt
		if auto_adapt {
			avg, lo, hi := image_stats(frame_buf, vinfo.width, vinfo.height)
			if abs(avg - bl_avg) > 0.10 || abs(lo - bl_lo) > 0.12 || abs(hi - bl_hi) > 0.12 {
				persist += 1
				if persist >= 3 {
					ap_cur = lerp_auto_params(ap_cur, ap_tgt, ap_prog)
					ap_tgt = compute_auto_values(opts, frame_buf, vinfo.width, vinfo.height, true)
					ap_prog = 0.0
					bl_avg, bl_lo, bl_hi = avg, lo, hi
					persist = 0
					info(fmt.tprintf("scene change at frame %d: R=%.3f MTF=%.3f EXP=%.3f", f, ap_tgt.grain_radius, ap_tgt.mtf, ap_tgt.exposure))
				}
			} else {
				persist = 0
			}
			if ap_prog < 1.0 {
				ap_prog = min(1.0, ap_prog + 1.0 / f32(trans_frames))
				t := ap_prog * ap_prog * (3.0 - 2.0 * ap_prog)
				ap_frame = lerp_auto_params(ap_cur, ap_tgt, t)
			}
		}
		task := Render_Task{frame = f, data = make([]u8, len(frame_buf)), auto_p = ap_frame}
		copy(task.data, frame_buf)
		if !chan.send(workers[f % n_workers].tasks, task) {
			fail("Render queue closed")
			os.exit(1)
		}
		actual = f + 1
		for {
			res, ok := chan.try_recv(results)
			if !ok {
				break
			}
			if !collect_one(res, &next_seq, &pending, vout) {
				fail("Failed to encode output")
				os.exit(1)
			}
			n_encoded = next_seq
			if n_encoded - last_progress >= 5 {
				last_progress = n_encoded
				progress_line(n_encoded, vinfo.n_frames, start)
			}
		}
	}
	for next_seq < actual {
		res, ok := chan.recv(results)
		if !ok {
			fail("Render result queue closed")
			os.exit(1)
		}
		if !collect_one(res, &next_seq, &pending, vout) {
			fail("Failed to encode output")
			os.exit(1)
		}
		n_encoded = next_seq
		if n_encoded - last_progress >= 5 || next_seq == actual {
			last_progress = n_encoded
			progress_line(n_encoded, vinfo.n_frames, start)
		}
	}

	video_in_finish(vin)
	video_out_finish(vout)
	fmt.println()
	success(fmt.tprintf("Video processed: %d frames, total time %s", n_encoded, time.since(start)))
}
