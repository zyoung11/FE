package main

import "core:fmt"
import "core:time"

Grading_Params :: struct {
	gamma:          f32,
	exposure:       f32,
	contrast:       f32,
	film:           f32,
	print_toe:      f32,
	print_shoulder: f32,
	sat_lo:         f32,
	sat_hi:         f32,
	cross:          f32,
	negative:       i32,
}

resolve_grading :: proc(opts: ^Options) -> Grading_Params {
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
	return Grading_Params {
		gamma          = opts.gamma,
		exposure       = opts.exposure,
		contrast       = opts.contrast,
		film           = opts.film,
		print_toe      = pt,
		print_shoulder = ps,
		sat_lo         = sl,
		sat_hi         = sh,
		cross          = cr,
		negative       = opts.negative ? 1 : 0,
	}
}

render_frame_cuda :: proc(
	ctx: ^Compute_Context,
	opts: ^Options,
	cfg: ^Film_Config,
	resized: []u8,
	w_sim: int,
	h_sim: int,
	frame_seed: u32,
	frame_idx: int,
	t: ^time.Time,
	verbose: bool,
) -> ([]u8, int, int, bool) {
	c := &ctx.cuda
	n := u32(w_sim * h_sim)
	if res := cu_memcpy_htod(c.d_input, raw_data(resized), u64(len(resized))); res != 0 {
		fail(fmt.tprintf("cuMemcpyHtoD: %s", cuda_error_string(res)))
		return nil, 0, 0, false
	}
	t^ = stage_time("Resize & linearize", t^, verbose)
	gpu_start := time.now()

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

	gamma_v := opts.gamma
	lin_args := [4]rawptr{&c.d_input, &c.d_ray_front, &gamma_v, &n}
	if !cuda_launch_1d(c, c.linearize_fn, n, &lin_args[0]) {
		return nil, 0, 0, false
	}

	layer_absorbs: [dynamic][3]f32
	defer delete(layer_absorbs)
	layer_depths: [dynamic]f32
	defer delete(layer_depths)
	emu_l := 0
	current_z: f32 = 0
	for item, idx in cfg.order {
		switch item.kind {
		case .Emulsion:
			prep := prep_emulsion(opts, cfg.emulsions[item.index], f32(opts.supersample), frame_seed, idx, frame_idx, w_sim, h_sim)
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
			expo_v := prep.expo_w
			src_args := [4]rawptr{&c.d_ray_front, &c.d_src, &expo_v, &n}
			if !cuda_launch_1d(c, c.src_fn, n, &src_args[0]) {
				return nil, 0, 0, false
			}
			if opts.reciprocity > 0 {
				p := 1.0 - opts.reciprocity * 0.2 * recip_diff_for(idx)
				rec_args := [3]rawptr{&c.d_src, &p, &n}
				if !cuda_launch_1d(c, c.recip_fn, n, &rec_args[0]) {
					return nil, 0, 0, false
				}
			}
			if prep.mtf_max < 0 || prep.mtf_max <= prep.sigma_mtf {
				if prep.sigma_mtf > 0 {
					if !cuda_gauss_blur_device(c, c.d_src, c.d_src, u32(w_sim), u32(h_sim), prep.sigma_mtf) {
						return nil, 0, 0, false
					}
				}
			} else if prep.sigma_mtf > 0 {
				if !cuda_adaptive_blur_device(c, c.d_src, c.d_src, u32(w_sim), u32(h_sim), prep.sigma_mtf, prep.mtf_max) {
					return nil, 0, 0, false
				}
			}
			dens_dst := c.d_dens + u64(emu_l) * c.img_f32
			if !cuda_render_kernel(c, prep.params, c.d_src, dens_dst) {
				return nil, 0, 0, false
			}
			absorb_v := prep.absorb
			fu_args := [4]rawptr{&c.d_ray_front, &dens_dst, &absorb_v, &n}
			if !cuda_launch_1d(c, c.front_update_fn, n, &fu_args[0]) {
				return nil, 0, 0, false
			}
			append(&layer_absorbs, prep.absorb)
			append(&layer_depths, current_z)
			current_z += 1
			emu_l += 1
		case .Filter:
			col := cfg.filters[item.index].color
			if verbose {
				info(fmt.tprintf("[filter %d] colour=[%d, %d, %d]", idx, col[0], col[1], col[2]))
			}
			col_v := [3]f32{f32(col[0]), f32(col[1]), f32(col[2])}
			fil_args := [3]rawptr{&c.d_ray_front, &col_v, &n}
			if !cuda_launch_1d(c, c.filter_fn, n, &fil_args[0]) {
				return nil, 0, 0, false
			}
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

	one := u32(0x3F800000)
	if res := cu_memset_d32(c.d_front, one, u64(n) * 3); res != 0 {
		fail(fmt.tprintf("cuMemsetD32: %s", cuda_error_string(res)))
		return nil, 0, 0, false
	}
	if res := cu_memset_d32(c.d_front_smooth, one, u64(n) * 3); res != 0 {
		fail(fmt.tprintf("cuMemsetD32: %s", cuda_error_string(res)))
		return nil, 0, 0, false
	}
	for l in 0 ..< emu_l {
		absorb_v := layer_absorbs[l]
		dens_l := c.d_dens + u64(l) * c.img_f32
		as_args := [4]rawptr{&c.d_front, &dens_l, &absorb_v, &n}
		if !cuda_launch_1d(c, c.assemble_fn, n, &as_args[0]) {
			return nil, 0, 0, false
		}
		if !cuda_gauss_blur_device(c, dens_l, c.d_blur_src, u32(w_sim), u32(h_sim), GRAIN_SMOOTH_SIGMA) {
			return nil, 0, 0, false
		}
		as2_args := [4]rawptr{&c.d_front_smooth, &c.d_blur_src, &absorb_v, &n}
		if !cuda_launch_1d(c, c.assemble_fn, n, &as2_args[0]) {
			return nil, 0, 0, false
		}
	}
	n3 := n * 3
	min_args := [3]rawptr{&c.d_front, &c.d_front_smooth, &n3}
	if !cuda_launch_1d(c, c.min_fn, n3, &min_args[0]) {
		return nil, 0, 0, false
	}

	if back_refl > EPS {
		hdr_args := [3]rawptr{&c.d_front, &c.d_front_hdr, &n3}
		if !cuda_launch_1d(c, c.hdr_fn, n3, &hdr_args[0]) {
			return nil, 0, 0, false
		}
		n_layers := emu_l
		depth := make([]f32, n_layers)
		defer delete(depth)
		last := max(layer_depths[n_layers - 1], 1.0)
		for l in 0 ..< n_layers {
			depth[l] = layer_depths[l] / last * film_thick / HALATION_SCALE
		}
		absorb_stack := make([]f32, n_layers * 3)
		defer delete(absorb_stack)
		for l in 0 ..< n_layers {
			absorb_stack[l * 3 + 0] = layer_absorbs[l][0]
			absorb_stack[l * 3 + 1] = layer_absorbs[l][1]
			absorb_stack[l * 3 + 2] = layer_absorbs[l][2]
		}
		bw := u32(max(1, w_sim / HALATION_SCALE))
		bh := u32(max(1, h_sim / HALATION_SCALE))
		if verbose {
			info(fmt.tprintf("Halation bounce %d samples/pixel @%dx%d ...", opts.bounce_samples, bw, bh))
		}
		sw_v := u32(w_sim)
		sh_v := u32(h_sim)
		gx := (bw + 15) / 16
		gy := (bh + 15) / 16
		bd_args := [6]rawptr{&c.d_front_hdr, &c.d_front_low, &sw_v, &sh_v, &bw, &bh}
		if res := cu_launch_kernel(
			c.box_down3_fn,
			gx, gy, 1,
			16, 16, 1,
			0, nil,
			raw_data(bd_args[:]), nil,
		); res != 0 {
			cu_ctx_synchronize()
			fail(fmt.tprintf("cuLaunchKernel: %s", cuda_error_string(res)))
			return nil, 0, 0, false
		}
		planes_v := u32(n_layers)
		bp_args := [7]rawptr{&c.d_dens, &c.d_dens_low, &sw_v, &sh_v, &bw, &bh, &planes_v}
		if res := cu_launch_kernel(
			c.box_down_planes_fn,
			gx, gy, 1,
			16, 16, 1,
			0, nil,
			raw_data(bp_args[:]), nil,
		); res != 0 {
			cu_ctx_synchronize()
			fail(fmt.tprintf("cuLaunchKernel: %s", cuda_error_string(res)))
			return nil, 0, 0, false
		}
		if res := cu_memcpy_htod(c.d_absorb, raw_data(absorb_stack), u64(len(absorb_stack)) * size_of(f32)); res != 0 {
			fail(fmt.tprintf("cuMemcpyHtoD: %s", cuda_error_string(res)))
			return nil, 0, 0, false
		}
		if res := cu_memcpy_htod(c.d_depth, raw_data(depth), u64(len(depth)) * size_of(f32)); res != 0 {
			fail(fmt.tprintf("cuMemcpyHtoD: %s", cuda_error_string(res)))
			return nil, 0, 0, false
		}
		bounce_params := Bounce_Params {
			width     = bw,
			height    = bh,
			n_layers  = u32(n_layers),
			n_samples = u32(opts.bounce_samples),
			seed      = 424242,
			film_px   = film_thick / HALATION_SCALE,
		}
		if !cuda_bounce_kernel(c, bounce_params, c.d_front_low, c.d_bounced_low, c.d_dens_low, c.d_absorb, c.d_depth) {
			return nil, 0, 0, false
		}
		gx_full := (sw_v + 15) / 16
		gy_full := (sh_v + 15) / 16
		up_args := [6]rawptr{&c.d_bounced_low, &c.d_bounced, &bw, &bh, &sw_v, &sh_v}
		if res := cu_launch_kernel(
			c.bilinear_up3_fn,
			gx_full, gy_full, 1,
			16, 16, 1,
			0, nil,
			raw_data(up_args[:]), nil,
		); res != 0 {
			cu_ctx_synchronize()
			fail(fmt.tprintf("cuLaunchKernel: %s", cuda_error_string(res)))
			return nil, 0, 0, false
		}
		refl_v := back_refl
		fin_args := [4]rawptr{&c.d_front, &c.d_bounced, &refl_v, &n3}
		if !cuda_launch_1d(c, c.final_fn, n3, &fin_args[0]) {
			return nil, 0, 0, false
		}
	}

	grading := resolve_grading(opts)
	out_args := [4]rawptr{&c.d_front, &c.d_out8, &grading, &n}
	if !cuda_launch_1d(c, c.out8_fn, n, &out_args[0]) {
		return nil, 0, 0, false
	}

	out8 := make([]u8, n * 3)
	if res := cu_memcpy_dtoh(raw_data(out8), c.d_out8, u64(len(out8))); res != 0 {
		delete(out8)
		fail(fmt.tprintf("cuMemcpyDtoH: %s", cuda_error_string(res)))
		return nil, 0, 0, false
	}
	if verbose {
		success("Simulation complete")
	}
	stage_time("GPU pipeline", gpu_start, verbose)
	return out8, w_sim, h_sim, true
}
