package main

import "core:dynlib"
import "core:fmt"
import "core:math"

CUDA_LIB :: "nvcuda.dll" when ODIN_OS == .Windows else "libcuda.so.1"

KERNEL_PTX_120 :: #load("kernel_120.ptx")
KERNEL_PTX_BASE :: #load("kernel_61.ptx")

CUresult :: i32
CUdevice :: i32
CUcontext :: rawptr
CUmodule :: rawptr
CUfunction :: rawptr
CUdeviceptr :: u64

cu_init: proc "c" (flags: u32) -> CUresult
cu_device_get_count: proc "c" (count: ^i32) -> CUresult
cu_device_get: proc "c" (device: ^CUdevice, ordinal: i32) -> CUresult
cu_device_get_name: proc "c" (name: cstring, len: i32, dev: CUdevice) -> CUresult
cu_device_compute_capability: proc "c" (major: ^i32, minor: ^i32, dev: CUdevice) -> CUresult
cu_ctx_create: proc "c" (pctx: ^CUcontext, flags: u32, dev: CUdevice) -> CUresult
cu_ctx_destroy: proc "c" (ctx: CUcontext) -> CUresult
cu_module_load_data: proc "c" (module: ^CUmodule, image: rawptr) -> CUresult
cu_module_unload: proc "c" (module: CUmodule) -> CUresult
cu_module_get_function: proc "c" (hfunc: ^CUfunction, hmod: CUmodule, name: cstring) -> CUresult
cu_launch_kernel: proc "c" (
	f: CUfunction,
	grid_x, grid_y, grid_z: u32,
	block_x, block_y, block_z: u32,
	shared_bytes: u32,
	stream: rawptr,
	kernel_params: ^rawptr,
	extra: rawptr,
) -> CUresult
cu_mem_alloc: proc "c" (dptr: ^CUdeviceptr, bytes: u64) -> CUresult
cu_mem_free: proc "c" (dptr: CUdeviceptr) -> CUresult
cu_memcpy_htod: proc "c" (dst: CUdeviceptr, src: rawptr, bytes: u64) -> CUresult
cu_memcpy_dtoh: proc "c" (dst: rawptr, src: CUdeviceptr, bytes: u64) -> CUresult
cu_ctx_synchronize: proc "c" () -> CUresult
cu_get_error_string: proc "c" (error: CUresult, pstr: ^cstring) -> CUresult
cu_memset_d32: proc "c" (dptr: CUdeviceptr, value: u32, count: u64) -> CUresult

Cuda_Context :: struct {
	lib:         dynlib.Library,
	dev:         CUdevice,
	ctx:         CUcontext,
	module:      CUmodule,
	render_fn:   CUfunction,
	bounce_fn:   CUfunction,
	lap_fn:      CUfunction,
	gauss_h_fn:  CUfunction,
	gauss_v_fn:  CUfunction,
	tmap_fn:     CUfunction,
	sum_fn:      CUfunction,
	sum_final_fn: CUfunction,
	blend_fn:    CUfunction,
	linearize_fn: CUfunction,
	src_fn:      CUfunction,
	recip_fn:    CUfunction,
	front_update_fn: CUfunction,
	filter_fn:   CUfunction,
	assemble_fn: CUfunction,
	min_fn:      CUfunction,
	hdr_fn:      CUfunction,
	final_fn:    CUfunction,
	out8_fn:     CUfunction,
	box_down3_fn: CUfunction,
	box_down_planes_fn: CUfunction,
	bilinear_up3_fn: CUfunction,
	img_f32:     u64,
	img_u8:      u64,
	bw:          u32,
	bh:          u32,
	d_src:       CUdeviceptr,
	d_front:     CUdeviceptr,
	d_bounced:   CUdeviceptr,
	d_dens:      CUdeviceptr,
	d_absorb:    CUdeviceptr,
	d_depth:     CUdeviceptr,
	d_blur_src:  CUdeviceptr,
	d_blur_tmp:  CUdeviceptr,
	d_lap:       CUdeviceptr,
	d_blur_min:  CUdeviceptr,
	d_blur_max:  CUdeviceptr,
	d_sum:       CUdeviceptr,
	d_blur_kernel: CUdeviceptr,
	d_ray_front: CUdeviceptr,
	d_front_smooth: CUdeviceptr,
	d_front_hdr: CUdeviceptr,
	d_input:     CUdeviceptr,
	d_out8:      CUdeviceptr,
	d_front_low: CUdeviceptr,
	d_bounced_low: CUdeviceptr,
	d_dens_low:  CUdeviceptr,
}

cuda_error_string :: proc(result: CUresult) -> string {
	if cu_get_error_string != nil {
		s: cstring
		if cu_get_error_string(result, &s) == 0 && s != nil {
			return string(s)
		}
	}
	return fmt.tprintf("CUDA error %d", result)
}

load_sym :: proc(lib: dynlib.Library, name: string, dst: ^rawptr) -> bool {
	addr, ok := dynlib.symbol_address(lib, name)
	if !ok {
		return false
	}
	dst^ = addr
	return true
}

cuda_load_symbols :: proc(lib: dynlib.Library) -> bool {
	if !load_sym(lib, "cuInit", cast(^rawptr)&cu_init) {return false}
	if !load_sym(lib, "cuDeviceGetCount", cast(^rawptr)&cu_device_get_count) {return false}
	if !load_sym(lib, "cuDeviceGet", cast(^rawptr)&cu_device_get) {return false}
	if !load_sym(lib, "cuDeviceGetName", cast(^rawptr)&cu_device_get_name) {return false}
	if !load_sym(lib, "cuDeviceComputeCapability", cast(^rawptr)&cu_device_compute_capability) {return false}
	if !load_sym(lib, "cuCtxCreate_v2", cast(^rawptr)&cu_ctx_create) {return false}
	if !load_sym(lib, "cuCtxDestroy_v2", cast(^rawptr)&cu_ctx_destroy) {return false}
	if !load_sym(lib, "cuModuleLoadData", cast(^rawptr)&cu_module_load_data) {return false}
	if !load_sym(lib, "cuModuleUnload", cast(^rawptr)&cu_module_unload) {return false}
	if !load_sym(lib, "cuModuleGetFunction", cast(^rawptr)&cu_module_get_function) {return false}
	if !load_sym(lib, "cuLaunchKernel", cast(^rawptr)&cu_launch_kernel) {return false}
	if !load_sym(lib, "cuMemAlloc_v2", cast(^rawptr)&cu_mem_alloc) {return false}
	if !load_sym(lib, "cuMemFree_v2", cast(^rawptr)&cu_mem_free) {return false}
	if !load_sym(lib, "cuMemcpyHtoD_v2", cast(^rawptr)&cu_memcpy_htod) {return false}
	if !load_sym(lib, "cuMemcpyDtoH_v2", cast(^rawptr)&cu_memcpy_dtoh) {return false}
	if !load_sym(lib, "cuCtxSynchronize", cast(^rawptr)&cu_ctx_synchronize) {return false}
	if !load_sym(lib, "cuGetErrorString", cast(^rawptr)&cu_get_error_string) {return false}
	if !load_sym(lib, "cuMemsetD32_v2", cast(^rawptr)&cu_memset_d32) {return false}
	return true
}

cuda_init :: proc(c: ^Cuda_Context, w: u32, h: u32, max_layers: u32, ordinal: int, verbose := true) -> bool {
	lib, ok := dynlib.load_library(CUDA_LIB, global_symbols = true)
	if !ok {
		warn(fmt.tprintf("%s not found", CUDA_LIB))
		return false
	}
	c.lib = lib
	if !cuda_load_symbols(lib) {
		warn("Failed to load CUDA driver symbols")
		return false
	}
	if cu_init(0) != 0 {
		warn("cuInit failed")
		return false
	}
	count: i32
	if cu_device_get_count(&count) != 0 || count <= 0 {
		warn("No CUDA device detected")
		return false
	}
	for i in 0 ..< int(count) {
		d: CUdevice
		if cu_device_get(&d, i32(i)) != 0 {
			continue
		}
		name_buf: [256]u8
		if cu_device_get_name(cstring(&name_buf[0]), 256, d) == 0 && verbose {
			info(fmt.tprintf("CUDA device %d: %s", i, string(cstring(&name_buf[0]))))
		}
	}
	if ordinal >= int(count) {
		warn(fmt.tprintf("CUDA device %d does not exist (%d available)", ordinal, count))
		return false
	}
	if cu_device_get(&c.dev, i32(ordinal)) != 0 {
		warn(fmt.tprintf("cuDeviceGet(%d) failed", ordinal))
		return false
	}
	name_buf: [256]u8
	if cu_device_get_name(cstring(&name_buf[0]), 256, c.dev) != 0 {
		warn("cuDeviceGetName failed")
		return false
	}
	if verbose {
		info(fmt.tprintf("GPU: %s", string(cstring(&name_buf[0]))))
	}

	major, minor: i32
	if cu_device_compute_capability(&major, &minor, c.dev) != 0 {
		warn("cuDeviceComputeCapability failed, falling back to CPU")
		return false
	}
	if verbose {
		info(fmt.tprintf("CUDA arch: %d.%d", major, minor))
	}
	ptx_data := KERNEL_PTX_BASE
	if major >= 12 {
		ptx_data = KERNEL_PTX_120
	}
	if cu_ctx_create(&c.ctx, 0, c.dev) != 0 {
		warn("cuCtxCreate failed, falling back to CPU")
		return false
	}
	if cu_module_load_data(&c.module, raw_data(ptx_data)) != 0 {
		if cu_module_load_data(&c.module, raw_data(KERNEL_PTX_BASE)) != 0 {
			warn("CUDA module load failed, falling back to CPU")
			return false
		}
	}
	if cu_module_get_function(&c.render_fn, c.module, "render_kernel") != 0 ||
	   cu_module_get_function(&c.bounce_fn, c.module, "bounce_kernel") != 0 ||
	   cu_module_get_function(&c.lap_fn, c.module, "lap_kernel") != 0 ||
	   cu_module_get_function(&c.gauss_h_fn, c.module, "gauss_h_kernel") != 0 ||
	   cu_module_get_function(&c.gauss_v_fn, c.module, "gauss_v_kernel") != 0 ||
	   cu_module_get_function(&c.tmap_fn, c.module, "tmap_kernel") != 0 ||
	   cu_module_get_function(&c.sum_fn, c.module, "sum_kernel") != 0 ||
	   cu_module_get_function(&c.sum_final_fn, c.module, "sum_final_kernel") != 0 ||
	   cu_module_get_function(&c.blend_fn, c.module, "blend_kernel") != 0 ||
	   cu_module_get_function(&c.linearize_fn, c.module, "linearize_kernel") != 0 ||
	   cu_module_get_function(&c.src_fn, c.module, "src_kernel") != 0 ||
	   cu_module_get_function(&c.recip_fn, c.module, "recip_kernel") != 0 ||
	   cu_module_get_function(&c.front_update_fn, c.module, "front_update_kernel") != 0 ||
	   cu_module_get_function(&c.filter_fn, c.module, "filter_kernel") != 0 ||
	   cu_module_get_function(&c.assemble_fn, c.module, "assemble_kernel") != 0 ||
	   cu_module_get_function(&c.min_fn, c.module, "min_kernel") != 0 ||
	   cu_module_get_function(&c.hdr_fn, c.module, "hdr_kernel") != 0 ||
	   cu_module_get_function(&c.final_fn, c.module, "final_kernel") != 0 ||
	   cu_module_get_function(&c.out8_fn, c.module, "out8_kernel") != 0 ||
	   cu_module_get_function(&c.box_down3_fn, c.module, "box_down3_kernel") != 0 ||
	   cu_module_get_function(&c.box_down_planes_fn, c.module, "box_down_planes_kernel") != 0 ||
	   cu_module_get_function(&c.bilinear_up3_fn, c.module, "bilinear_up3_kernel") != 0 {
		warn("CUDA kernel lookup failed, falling back to CPU")
		return false
	}

	img_size := u64(w) * u64(h) * size_of(f32)
	rgb_size := img_size * 3
	layer_size := u64(max_layers) * img_size
	c.img_f32 = img_size
	c.img_u8 = u64(w) * u64(h) * 3
	c.bw = max(u32(1), w / 4)
	c.bh = max(u32(1), h / 4)
	low_rgb := u64(c.bw) * u64(c.bh) * 3 * size_of(f32)
	low_layer := u64(max_layers) * u64(c.bw) * u64(c.bh) * size_of(f32)
	if cu_mem_alloc(&c.d_src, img_size) != 0 ||
	   cu_mem_alloc(&c.d_front, rgb_size) != 0 ||
	   cu_mem_alloc(&c.d_bounced, rgb_size) != 0 ||
	   cu_mem_alloc(&c.d_dens, layer_size) != 0 ||
	   cu_mem_alloc(&c.d_absorb, u64(max_layers) * 3 * size_of(f32)) != 0 ||
	   cu_mem_alloc(&c.d_depth, u64(max_layers) * size_of(f32)) != 0 ||
	   cu_mem_alloc(&c.d_blur_src, img_size) != 0 ||
	   cu_mem_alloc(&c.d_blur_tmp, img_size) != 0 ||
	   cu_mem_alloc(&c.d_lap, img_size) != 0 ||
	   cu_mem_alloc(&c.d_blur_min, img_size) != 0 ||
	   cu_mem_alloc(&c.d_blur_max, img_size) != 0 ||
	   cu_mem_alloc(&c.d_sum, u64((w * h + 255) / 256) * 4 + 4) != 0 ||
	   cu_mem_alloc(&c.d_blur_kernel, 256) != 0 ||
	   cu_mem_alloc(&c.d_ray_front, rgb_size) != 0 ||
	   cu_mem_alloc(&c.d_front_smooth, rgb_size) != 0 ||
	   cu_mem_alloc(&c.d_front_hdr, rgb_size) != 0 ||
	   cu_mem_alloc(&c.d_input, c.img_u8) != 0 ||
	   cu_mem_alloc(&c.d_out8, c.img_u8) != 0 ||
	   cu_mem_alloc(&c.d_front_low, low_rgb) != 0 ||
	   cu_mem_alloc(&c.d_bounced_low, low_rgb) != 0 ||
	   cu_mem_alloc(&c.d_dens_low, low_layer) != 0 {
		warn("CUDA memory allocation failed, falling back to CPU")
		cuda_cleanup(c)
		return false
	}
	return true
}

cuda_cleanup :: proc(c: ^Cuda_Context) {
	if c.d_dens_low != 0 {cu_mem_free(c.d_dens_low)}
	if c.d_bounced_low != 0 {cu_mem_free(c.d_bounced_low)}
	if c.d_front_low != 0 {cu_mem_free(c.d_front_low)}
	if c.d_out8 != 0 {cu_mem_free(c.d_out8)}
	if c.d_input != 0 {cu_mem_free(c.d_input)}
	if c.d_front_hdr != 0 {cu_mem_free(c.d_front_hdr)}
	if c.d_front_smooth != 0 {cu_mem_free(c.d_front_smooth)}
	if c.d_ray_front != 0 {cu_mem_free(c.d_ray_front)}
	if c.d_blur_kernel != 0 {cu_mem_free(c.d_blur_kernel)}
	if c.d_sum != 0 {cu_mem_free(c.d_sum)}
	if c.d_blur_max != 0 {cu_mem_free(c.d_blur_max)}
	if c.d_blur_min != 0 {cu_mem_free(c.d_blur_min)}
	if c.d_lap != 0 {cu_mem_free(c.d_lap)}
	if c.d_blur_tmp != 0 {cu_mem_free(c.d_blur_tmp)}
	if c.d_blur_src != 0 {cu_mem_free(c.d_blur_src)}
	if c.d_depth != 0 {cu_mem_free(c.d_depth)}
	if c.d_absorb != 0 {cu_mem_free(c.d_absorb)}
	if c.d_dens != 0 {cu_mem_free(c.d_dens)}
	if c.d_bounced != 0 {cu_mem_free(c.d_bounced)}
	if c.d_front != 0 {cu_mem_free(c.d_front)}
	if c.d_src != 0 {cu_mem_free(c.d_src)}
	if c.module != nil {cu_module_unload(c.module)}
	if c.ctx != nil {cu_ctx_destroy(c.ctx)}
	c.d_dens_low = 0
	c.d_bounced_low = 0
	c.d_front_low = 0
	c.d_out8 = 0
	c.d_input = 0
	c.d_front_hdr = 0
	c.d_front_smooth = 0
	c.d_ray_front = 0
	c.d_blur_kernel = 0
	c.d_sum = 0
	c.d_blur_max = 0
	c.d_blur_min = 0
	c.d_lap = 0
	c.d_blur_tmp = 0
	c.d_blur_src = 0
	c.d_depth = 0
	c.d_absorb = 0
	c.d_dens = 0
	c.d_bounced = 0
	c.d_front = 0
	c.d_src = 0
	c.module = nil
	c.ctx = nil
}

cuda_render_kernel :: proc(c: ^Cuda_Context, params: Render_Params, src: CUdeviceptr, dst: CUdeviceptr) -> bool {
	p := params
	p.y_offset = 0
	src_v := src
	dst_v := dst
	grid_x := (params.width + 15) / 16
	grid_y := (params.height + 15) / 16
	args := [3]rawptr{&src_v, &dst_v, &p}
	if res := cu_launch_kernel(
		c.render_fn,
		grid_x, grid_y, 1,
		16, 16, 1,
		0, nil,
		raw_data(args[:]), nil,
	); res != 0 {
		cu_ctx_synchronize()
		fail(fmt.tprintf("cuLaunchKernel: %s", cuda_error_string(res)))
		return false
	}
	return true
}

cuda_bounce_kernel :: proc(
	c: ^Cuda_Context,
	params: Bounce_Params,
	front: CUdeviceptr,
	bounced: CUdeviceptr,
	dens: CUdeviceptr,
	absorb: CUdeviceptr,
	depth: CUdeviceptr,
) -> bool {
	p := params
	p.y_offset = 0
	grid_x := (params.width + 15) / 16
	grid_y := (params.height + 15) / 16
	front_v := front
	bounced_v := bounced
	dens_v := dens
	absorb_v := absorb
	depth_v := depth
	args := [6]rawptr{&front_v, &bounced_v, &dens_v, &absorb_v, &depth_v, &p}
	if res := cu_launch_kernel(
		c.bounce_fn,
		grid_x, grid_y, 1,
		16, 16, 1,
		0, nil,
		raw_data(args[:]), nil,
	); res != 0 {
		cu_ctx_synchronize()
		fail(fmt.tprintf("cuLaunchKernel: %s", cuda_error_string(res)))
		return false
	}
	return true
}

gauss_kernel_weights :: proc(sigma: f32) -> ([]f32, int) {
	r := max(1, int(math.ceil(3.0 * sigma)))
	kernel := make([]f32, 2 * r + 1)
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
	return kernel, r
}

cuda_launch_1d :: proc(c: ^Cuda_Context, f: CUfunction, n: u32, args: ^rawptr) -> bool {
	grid := (n + 255) / 256
	if res := cu_launch_kernel(
		f,
		grid, 1, 1,
		256, 1, 1,
		0, nil,
		args, nil,
	); res != 0 {
		cu_ctx_synchronize()
		fail(fmt.tprintf("cuLaunchKernel: %s", cuda_error_string(res)))
		return false
	}
	return true
}

cuda_upload_gauss_kernel :: proc(c: ^Cuda_Context, sigma: f32) -> (i32, bool) {
	kernel, r := gauss_kernel_weights(sigma)
	defer delete(kernel)
	if res := cu_memcpy_htod(c.d_blur_kernel, raw_data(kernel), u64(len(kernel)) * size_of(f32)); res != 0 {
		fail(fmt.tprintf("cuMemcpyHtoD: %s", cuda_error_string(res)))
		return 0, false
	}
	return i32(r), true
}

cuda_launch_gauss :: proc(c: ^Cuda_Context, src: CUdeviceptr, dst: CUdeviceptr, w: u32, h: u32, r: i32) -> bool {
	n := w * h
	src_v := src
	dst_v := dst
	w_v := w
	h_v := h
	r_v := r
	args := [6]rawptr{&src_v, &c.d_blur_tmp, &w_v, &h_v, &c.d_blur_kernel, &r_v}
	if !cuda_launch_1d(c, c.gauss_h_fn, n, &args[0]) {
		return false
	}
	args2 := [6]rawptr{&c.d_blur_tmp, &dst_v, &w_v, &h_v, &c.d_blur_kernel, &r_v}
	if !cuda_launch_1d(c, c.gauss_v_fn, n, &args2[0]) {
		return false
	}
	return true
}

cuda_gauss_blur_device :: proc(c: ^Cuda_Context, src: CUdeviceptr, dst: CUdeviceptr, w: u32, h: u32, sigma: f32) -> bool {
	r, ok := cuda_upload_gauss_kernel(c, sigma)
	if !ok {
		return false
	}
	return cuda_launch_gauss(c, src, dst, w, h, r)
}

cuda_adaptive_blur_device :: proc(c: ^Cuda_Context, src: CUdeviceptr, dst: CUdeviceptr, w: u32, h: u32, sigma_min: f32, sigma_max: f32) -> bool {
	if sigma_max <= sigma_min {
		return cuda_gauss_blur_device(c, src, dst, w, h, sigma_max)
	}
	n := w * h
	w_v := w
	h_v := h
	src_v := src
	args := [4]rawptr{&src_v, &c.d_lap, &w_v, &h_v}
	if !cuda_launch_1d(c, c.lap_fn, n, &args[0]) {
		return false
	}
	r, ok := cuda_upload_gauss_kernel(c, max(sigma_min * 2.0, 2.0))
	if !ok {
		return false
	}
	if !cuda_launch_gauss(c, c.d_lap, c.d_lap, w, h, r) {
		return false
	}
	blocks := (n + 255) / 256
	args2 := [3]rawptr{&c.d_lap, &c.d_sum, &n}
	if !cuda_launch_1d(c, c.sum_fn, n, &args2[0]) {
		return false
	}
	sf_args := [3]rawptr{&c.d_sum, &c.d_sum, &blocks}
	if res := cu_launch_kernel(
		c.sum_final_fn,
		1, 1, 1,
		256, 1, 1,
		0, nil,
		&sf_args[0], nil,
	); res != 0 {
		cu_ctx_synchronize()
		fail(fmt.tprintf("cuLaunchKernel: %s", cuda_error_string(res)))
		return false
	}
	args3 := [4]rawptr{&c.d_lap, &c.d_lap, &c.d_sum, &n}
	if !cuda_launch_1d(c, c.tmap_fn, n, &args3[0]) {
		return false
	}
	r2, ok2 := cuda_upload_gauss_kernel(c, sigma_min)
	if !ok2 {
		return false
	}
	if !cuda_launch_gauss(c, src, c.d_blur_min, w, h, r2) {
		return false
	}
	r3, ok3 := cuda_upload_gauss_kernel(c, sigma_max)
	if !ok3 {
		return false
	}
	if !cuda_launch_gauss(c, src, c.d_blur_max, w, h, r3) {
		return false
	}
	dst_v := dst
	args4 := [5]rawptr{&c.d_blur_min, &c.d_blur_max, &c.d_lap, &dst_v, &n}
	return cuda_launch_1d(c, c.blend_fn, n, &args4[0])
}
