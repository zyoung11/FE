package main

import "core:dynlib"
import "core:fmt"

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

Cuda_Context :: struct {
	lib:       dynlib.Library,
	dev:       CUdevice,
	ctx:       CUcontext,
	module:    CUmodule,
	render_fn: CUfunction,
	bounce_fn: CUfunction,
	d_src:     CUdeviceptr,
	d_neg:     CUdeviceptr,
	d_front:   CUdeviceptr,
	d_bounced: CUdeviceptr,
	d_dens:    CUdeviceptr,
	d_absorb:  CUdeviceptr,
	d_depth:   CUdeviceptr,
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
	   cu_module_get_function(&c.bounce_fn, c.module, "bounce_kernel") != 0 {
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
	   cu_mem_alloc(&c.d_depth, u64(max_layers) * size_of(f32)) != 0 {
		warn("CUDA 显存分配失败，回退 CPU")
		cuda_cleanup(c)
		return false
	}
	return true
}

cuda_cleanup :: proc(c: ^Cuda_Context) {
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
