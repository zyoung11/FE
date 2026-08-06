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
	d_src:       CUdeviceptr,
	d_neg:       CUdeviceptr,
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

cuda_init :: proc(c: ^Cuda_Context, w: u32, h: u32, max_layers: u32, ordinal: int) -> bool {
	lib, ok := dynlib.load_library(CUDA_LIB, global_symbols = true)
	if !ok {
		warn(fmt.tprintf("未找到 %s", CUDA_LIB))
		return false
	}
	c.lib = lib
	if !cuda_load_symbols(lib) {
		warn("CUDA 驱动符号加载失败")
		return false
	}
	if cu_init(0) != 0 {
		warn("cuInit 失败")
		return false
	}
	count: i32
	if cu_device_get_count(&count) != 0 || count <= 0 {
		warn("未检测到 CUDA 设备")
		return false
	}
	for i in 0 ..< int(count) {
		d: CUdevice
		if cu_device_get(&d, i32(i)) != 0 {
			continue
		}
		name_buf: [256]u8
		if cu_device_get_name(cstring(&name_buf[0]), 256, d) == 0 {
			info(fmt.tprintf("CUDA 设备 %d: %s", i, string(cstring(&name_buf[0]))))
		}
	}
	if ordinal >= int(count) {
		warn(fmt.tprintf("CUDA 设备 %d 不存在（共 %d 个）", ordinal, count))
		return false
	}
	if cu_device_get(&c.dev, i32(ordinal)) != 0 {
		warn(fmt.tprintf("cuDeviceGet(%d) 失败", ordinal))
		return false
	}
	name_buf: [256]u8
	if cu_device_get_name(cstring(&name_buf[0]), 256, c.dev) != 0 {
		warn("cuDeviceGetName 失败")
		return false
	}
	info(fmt.tprintf("GPU: %s", string(cstring(&name_buf[0]))))

	major, minor: i32
	if cu_device_compute_capability(&major, &minor, c.dev) != 0 {
		warn("cuDeviceComputeCapability 失败，回退 CPU")
		return false
	}
	info(fmt.tprintf("CUDA 架构: %d.%d", major, minor))
	ptx_data := KERNEL_PTX_BASE
	if major >= 12 {
		ptx_data = KERNEL_PTX_120
	}
	if cu_ctx_create(&c.ctx, 0, c.dev) != 0 {
		warn("cuCtxCreate 失败，回退 CPU")
		return false
	}
	if cu_module_load_data(&c.module, raw_data(ptx_data)) != 0 {
		if cu_module_load_data(&c.module, raw_data(KERNEL_PTX_BASE)) != 0 {
			warn("CUDA 模块加载失败，回退 CPU")
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
	   cu_module_get_function(&c.blend_fn, c.module, "blend_kernel") != 0 {
		warn("CUDA 内核查找失败，回退 CPU")
		return false
	}

	img_size := u64(w) * u64(h) * size_of(f32)
	rgb_size := img_size * 3
	layer_size := u64(max_layers) * img_size
	if cu_mem_alloc(&c.d_src, img_size) != 0 ||
	   cu_mem_alloc(&c.d_neg, img_size) != 0 ||
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
	   cu_mem_alloc(&c.d_blur_kernel, 256) != 0 {
		warn("CUDA 显存分配失败，回退 CPU")
		cuda_cleanup(c)
		return false
	}
	return true
}

cuda_cleanup :: proc(c: ^Cuda_Context) {
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
	if c.d_neg != 0 {cu_mem_free(c.d_neg)}
	if c.d_src != 0 {cu_mem_free(c.d_src)}
	if c.module != nil {cu_module_unload(c.module)}
	if c.ctx != nil {cu_ctx_destroy(c.ctx)}
	c.d_src = 0
	c.d_neg = 0
	c.d_front = 0
	c.d_bounced = 0
	c.d_dens = 0
	c.d_absorb = 0
	c.d_depth = 0
	c.d_blur_src = 0
	c.d_blur_tmp = 0
	c.d_lap = 0
	c.d_blur_min = 0
	c.d_blur_max = 0
	c.d_sum = 0
	c.d_blur_kernel = 0
	c.module = nil
	c.ctx = nil
}

cuda_dispatch_render :: proc(c: ^Cuda_Context, params: Render_Params, src: []f32, neg: []f32) -> bool {
	if res := cu_memcpy_htod(c.d_src, raw_data(src), u64(len(src)) * size_of(f32)); res != 0 {
		fail(fmt.tprintf("cuMemcpyHtoD: %s", cuda_error_string(res)))
		return false
	}
	p := params
	p.y_offset = 0
	grid_x := (params.width + 15) / 16
	grid_y := (params.height + 15) / 16
	args := [3]rawptr{&c.d_src, &c.d_neg, &p}
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
	if res := cu_ctx_synchronize(); res != 0 {
		fail(fmt.tprintf("cuCtxSynchronize: %s", cuda_error_string(res)))
		return false
	}
	if res := cu_memcpy_dtoh(raw_data(neg), c.d_neg, u64(len(neg)) * size_of(f32)); res != 0 {
		fail(fmt.tprintf("cuMemcpyDtoH: %s", cuda_error_string(res)))
		return false
	}
	return true
}

cuda_dispatch_bounce :: proc(
	c: ^Cuda_Context,
	params: Bounce_Params,
	front: []f32,
	bounced: []f32,
	dens: []f32,
	absorb: []f32,
	depth: []f32,
) -> bool {
	if res := cu_memcpy_htod(c.d_front, raw_data(front), u64(len(front)) * size_of(f32)); res != 0 {
		fail(fmt.tprintf("cuMemcpyHtoD: %s", cuda_error_string(res)))
		return false
	}
	if res := cu_memcpy_htod(c.d_dens, raw_data(dens), u64(len(dens)) * size_of(f32)); res != 0 {
		fail(fmt.tprintf("cuMemcpyHtoD: %s", cuda_error_string(res)))
		return false
	}
	if res := cu_memcpy_htod(c.d_absorb, raw_data(absorb), u64(len(absorb)) * size_of(f32)); res != 0 {
		fail(fmt.tprintf("cuMemcpyHtoD: %s", cuda_error_string(res)))
		return false
	}
	if res := cu_memcpy_htod(c.d_depth, raw_data(depth), u64(len(depth)) * size_of(f32)); res != 0 {
		fail(fmt.tprintf("cuMemcpyHtoD: %s", cuda_error_string(res)))
		return false
	}
	p := params
	p.y_offset = 0
	grid_x := (params.width + 15) / 16
	grid_y := (params.height + 15) / 16
	args := [6]rawptr{&c.d_front, &c.d_bounced, &c.d_dens, &c.d_absorb, &c.d_depth, &p}
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
	if res := cu_ctx_synchronize(); res != 0 {
		fail(fmt.tprintf("cuCtxSynchronize: %s", cuda_error_string(res)))
		return false
	}
	if res := cu_memcpy_dtoh(raw_data(bounced), c.d_bounced, u64(len(bounced)) * size_of(f32)); res != 0 {
		fail(fmt.tprintf("cuMemcpyDtoH: %s", cuda_error_string(res)))
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

cuda_gauss_blur :: proc(c: ^Cuda_Context, src: []f32, w: u32, h: u32, sigma: f32) -> ([]f32, bool) {
	if sigma <= 0 {
		out := make([]f32, len(src))
		copy(out, src)
		return out, true
	}
	if res := cu_memcpy_htod(c.d_blur_src, raw_data(src), u64(len(src)) * size_of(f32)); res != 0 {
		fail(fmt.tprintf("cuMemcpyHtoD: %s", cuda_error_string(res)))
		return nil, false
	}
	r, ok := cuda_upload_gauss_kernel(c, sigma)
	if !ok {
		return nil, false
	}
	if !cuda_launch_gauss(c, c.d_blur_src, c.d_blur_src, w, h, r) {
		return nil, false
	}
	if res := cu_ctx_synchronize(); res != 0 {
		fail(fmt.tprintf("cuCtxSynchronize: %s", cuda_error_string(res)))
		return nil, false
	}
	out := make([]f32, len(src))
	if res := cu_memcpy_dtoh(raw_data(out), c.d_blur_src, u64(len(out)) * size_of(f32)); res != 0 {
		delete(out)
		fail(fmt.tprintf("cuMemcpyDtoH: %s", cuda_error_string(res)))
		return nil, false
	}
	return out, true
}

cuda_adaptive_blur :: proc(c: ^Cuda_Context, src: []f32, w: u32, h: u32, sigma_min: f32, sigma_max: f32) -> ([]f32, bool) {
	if sigma_max <= sigma_min {
		return cuda_gauss_blur(c, src, w, h, sigma_max)
	}
	n := w * h
	w_v := w
	h_v := h
	if res := cu_memcpy_htod(c.d_blur_src, raw_data(src), u64(len(src)) * size_of(f32)); res != 0 {
		fail(fmt.tprintf("cuMemcpyHtoD: %s", cuda_error_string(res)))
		return nil, false
	}
	args := [4]rawptr{&c.d_blur_src, &c.d_lap, &w_v, &h_v}
	if !cuda_launch_1d(c, c.lap_fn, n, &args[0]) {
		return nil, false
	}
	r, ok := cuda_upload_gauss_kernel(c, max(sigma_min * 2.0, 2.0))
	if !ok {
		return nil, false
	}
	if !cuda_launch_gauss(c, c.d_lap, c.d_lap, w, h, r) {
		return nil, false
	}
	if res := cu_memset_d32(c.d_sum, 0, 1); res != 0 {
		fail(fmt.tprintf("cuMemsetD32: %s", cuda_error_string(res)))
		return nil, false
	}
	blocks := (n + 255) / 256
	args2 := [3]rawptr{&c.d_lap, &c.d_sum, &n}
	if !cuda_launch_1d(c, c.sum_fn, n, &args2[0]) {
		return nil, false
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
		return nil, false
	}
	mean_host: f32
	if res := cu_memcpy_dtoh(&mean_host, c.d_sum, 4); res != 0 {
		fail(fmt.tprintf("cuMemcpyDtoH: %s", cuda_error_string(res)))
		return nil, false
	}
	mean := mean_host / f32(n)
	args3 := [4]rawptr{&c.d_lap, &c.d_lap, &mean, &n}
	if !cuda_launch_1d(c, c.tmap_fn, n, &args3[0]) {
		return nil, false
	}
	r2, ok2 := cuda_upload_gauss_kernel(c, sigma_min)
	if !ok2 {
		return nil, false
	}
	if !cuda_launch_gauss(c, c.d_blur_src, c.d_blur_min, w, h, r2) {
		return nil, false
	}
	r3, ok3 := cuda_upload_gauss_kernel(c, sigma_max)
	if !ok3 {
		return nil, false
	}
	if !cuda_launch_gauss(c, c.d_blur_src, c.d_blur_max, w, h, r3) {
		return nil, false
	}
	args4 := [5]rawptr{&c.d_blur_min, &c.d_blur_max, &c.d_lap, &c.d_blur_src, &n}
	if !cuda_launch_1d(c, c.blend_fn, n, &args4[0]) {
		return nil, false
	}
	if res := cu_ctx_synchronize(); res != 0 {
		fail(fmt.tprintf("cuCtxSynchronize: %s", cuda_error_string(res)))
		return nil, false
	}
	out := make([]f32, len(src))
	if res := cu_memcpy_dtoh(raw_data(out), c.d_blur_src, u64(len(out)) * size_of(f32)); res != 0 {
		delete(out)
		fail(fmt.tprintf("cuMemcpyDtoH: %s", cuda_error_string(res)))
		return nil, false
	}
	return out, true
}
