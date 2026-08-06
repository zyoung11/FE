package main

import "core:c"
import "core:fmt"
import "core:strings"
import stb "vendor:stb/image"

Auto_Result :: struct {
	grain_radius: f32,
	grain_sigma:  f32,
	sigma_filter: f32,
	mtf_abs:      f32,
}

content_sharpness :: proc(pixels: [^]u8, w: int, h: int) -> f32 {
	gray_src := make([]u8, w * h)
	defer delete(gray_src)
	for i in 0 ..< w * h {
		gray_src[i] = u8(
			0.299 * f32(pixels[i * 3]) +
			0.587 * f32(pixels[i * 3 + 1]) +
			0.114 * f32(pixels[i * 3 + 2]),
		)
	}
	scale := min(1.0, 256.0 / f32(max(w, h)))
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

compute_auto :: proc(sharpness: f32, height: int) -> Auto_Result {
	t := clamp((sharpness - 4.0) / 8.0, 0.0, 1.0)
	g := 0.09 - 0.05 * t
	m_mtf := min(1.8 - 0.8 * t, max(0.5, 0.0015 * f32(height)))
	return Auto_Result{grain_radius = g, grain_sigma = 0.01, sigma_filter = 0.04, mtf_abs = m_mtf}
}

build_cfg_text :: proc(
	mode: string,
	thickness: f32,
	reflectance: f32,
	r: f32,
	s: f32,
	f: f32,
	mtf_abs: f32,
) -> string {
	sb := strings.builder_make()
	defer strings.builder_destroy(&sb)
	if mode == "bw" {
		strings.write_string(&sb, "[[emulsion]]\n")
		strings.write_string(&sb, "sensitising_dye_color = [0, 0, 0]\n")
		strings.write_string(&sb, fmt.tprintf("grain_radius = %.4f\n", r))
		strings.write_string(&sb, fmt.tprintf("grain_sigma = %.4f\n", s))
		strings.write_string(&sb, fmt.tprintf("sigma_filter = %.4f\n", f))
		strings.write_string(&sb, fmt.tprintf("mtf_blur = %.4f\n", mtf_abs * 0.6))
		strings.write_string(&sb, fmt.tprintf("mtf_blur_max = %.4f\n", mtf_abs * 1.4))
		strings.write_string(&sb, "\n")
	} else {
		dyes := [3][3]u8{{255, 255, 0}, {255, 0, 255}, {0, 255, 255}}
		for i in 0 ..< 3 {
			strings.write_string(&sb, "[[emulsion]]\n")
			strings.write_string(
				&sb,
				fmt.tprintf(
					"sensitising_dye_color = [%d, %d, %d]\n",
					dyes[i][0],
					dyes[i][1],
					dyes[i][2],
				),
			)
			strings.write_string(&sb, fmt.tprintf("grain_radius = %.4f\n", r))
			strings.write_string(&sb, fmt.tprintf("grain_sigma = %.4f\n", s))
			strings.write_string(&sb, fmt.tprintf("sigma_filter = %.4f\n", f))
			strings.write_string(&sb, fmt.tprintf("mtf_blur = %.4f\n", mtf_abs * 0.6))
			strings.write_string(&sb, fmt.tprintf("mtf_blur_max = %.4f\n", mtf_abs * 1.4))
			if i < 2 {
				strings.write_string(&sb, "\n[[filter]]\n")
				strings.write_string(
					&sb,
					fmt.tprintf("color = [%d, %d, %d]\n", dyes[i][0], dyes[i][1], dyes[i][2]),
				)
			}
			strings.write_string(&sb, "\n")
		}
	}
	strings.write_string(&sb, "[[film_base]]\n")
	strings.write_string(&sb, fmt.tprintf("thickness = %.1f\n", thickness))
	strings.write_string(&sb, "\n[[back]]\n")
	strings.write_string(&sb, fmt.tprintf("reflectance = %.4f\n", reflectance))
	return strings.clone(strings.to_string(sb))
}

