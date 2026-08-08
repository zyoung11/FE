package main

import "core:encoding/json"
import "core:fmt"
import "core:os"
import "core:strconv"
import "core:strings"

Config_Kind :: enum {
	Emulsion,
	Filter,
	Film_Base,
	Back,
}

Config_Item :: struct {
	kind:  Config_Kind,
	index: int,
}

Emulsion_Cfg :: struct {
	dye:          [3]u8,
	grain_radius: f32,
	grain_sigma:  f32,
	sigma_filter: f32,
	mtf_blur:     Maybe(f32),
	mtf_blur_max: Maybe(f32),
}

Filter_Cfg :: struct {
	color: [3]u8,
}

Film_Base_Cfg :: struct {
	thickness: f32,
}

Back_Cfg :: struct {
	reflectance: f32,
}

Film_Config :: struct {
	emulsions: [dynamic]Emulsion_Cfg,
	filters:   [dynamic]Filter_Cfg,
	bases:     [dynamic]Film_Base_Cfg,
	backs:     [dynamic]Back_Cfg,
	order:     [dynamic]Config_Item,
}

destroy_film_config :: proc(cfg: ^Film_Config) {
	delete(cfg.emulsions)
	delete(cfg.filters)
	delete(cfg.bases)
	delete(cfg.backs)
	delete(cfg.order)
}

default_options :: proc() -> Options {
	return Options {
		height         = 0,
		supersample    = 0,
		samples        = 0,
		bounce_samples = 0,
		gamma          = 0,
		mtf            = -1.0,
		exposure       = 0.0,
		contrast       = 1.0,
		reflectance    = -1.0,
		thickness      = -1.0,
		grain_radius   = -1.0,
		grain_sigma    = -1.0,
		sigma_filter   = -1.0,
		seed           = 12345,
		device         = "auto",
		film           = 0.0,
		print_toe      = -1,
		print_shoulder = -1,
		sat_lo         = -1,
		sat_hi         = -1,
		cross          = -1,
		bitrate        = 60,
		maxrate        = 100,
		reciprocity    = 0.0,
		negative       = false,
		mode           = "color",
	}
}

json_num_f32 :: proc(v: json.Value) -> (f32, bool) {
	#partial switch x in v {
	case json.Integer:
		return f32(x), true
	case json.Float:
		return f32(x), true
	}
	return 0, false
}

json_num_int :: proc(v: json.Value) -> (int, bool) {
	#partial switch x in v {
	case json.Integer:
		return int(x), true
	case json.Float:
		return int(x), true
	}
	return 0, false
}

set_f32_field :: proc(o: json.Object, key: string, dst: ^f32) {
	if v, ok := o[key]; ok {
		if x, xok := json_num_f32(v); xok {
			dst^ = x
		}
	}
}

set_int_field :: proc(o: json.Object, key: string, dst: ^int) {
	if v, ok := o[key]; ok {
		if x, xok := json_num_int(v); xok {
			dst^ = x
		}
	}
}

set_u32_field :: proc(o: json.Object, key: string, dst: ^u32) {
	if v, ok := o[key]; ok {
		if x, xok := json_num_int(v); xok && x >= 0 {
			dst^ = u32(x)
		}
	}
}

set_bool_field :: proc(o: json.Object, key: string, dst: ^bool) {
	if v, ok := o[key]; ok {
		if b, bok := v.(json.Boolean); bok {
			dst^ = b
		}
	}
}

set_str_field :: proc(o: json.Object, key: string, dst: ^string) {
	if v, ok := o[key]; ok {
		if s, sok := v.(json.String); sok {
			dst^ = strings.clone(s)
		}
	}
}

parse_u8_arr3 :: proc(v: json.Value) -> ([3]u8, bool) {
	arr, ok := v.(json.Array)
	if !ok || len(arr) != 3 {
		return {}, false
	}
	out: [3]u8
	for i in 0 ..< 3 {
		x, xok := json_num_int(arr[i])
		if !xok || x < 0 || x > 255 {
			return {}, false
		}
		out[i] = u8(x)
	}
	return out, true
}

parse_maybe_f32 :: proc(v: json.Value) -> Maybe(f32) {
	if _, is_null := v.(json.Null); is_null {
		return nil
	}
	if x, ok := json_num_f32(v); ok {
		return x
	}
	return nil
}

parse_order_item :: proc(s: string) -> (kind: Config_Kind, idx: int, ok: bool) {
	parts := strings.split(s, ":")
	defer delete(parts)
	if len(parts) != 2 {
		return
	}
	switch strings.trim_space(parts[0]) {
	case "emulsion":
		kind = .Emulsion
	case "filter":
		kind = .Filter
	case "film_base":
		kind = .Film_Base
	case "back":
		kind = .Back
	case:
		return
	}
	i, iok := strconv.parse_int(strings.trim_space(parts[1]))
	if !iok {
		return
	}
	return kind, i, true
}

validate_film_config :: proc(opts: ^Options, cfg: ^Film_Config) -> bool {
	if len(cfg.emulsions) == 0 {
		fail("Config requires at least one emulsion")
		return false
	}
	if len(cfg.filters) != len(cfg.emulsions) && len(cfg.filters) != len(cfg.emulsions) - 1 {
		fail("filters count must equal emulsions count or be one less")
		return false
	}
	if opts.mtf >= 0 {
		for &emu in cfg.emulsions {
			emu.mtf_blur = opts.mtf * 0.6
			emu.mtf_blur_max = opts.mtf * 1.4
		}
	}
	return true
}

parse_config_file :: proc(path: string) -> (cfg: Film_Config, opts: Options, ok: bool) {
	opts = default_options()
	data, err := os.read_entire_file_from_path(path, context.allocator)
	if err != nil {
		return cfg, opts, false
	}
	defer delete(data)
	value, perr := json.parse_string(string(data))
	if perr != .None {
		return cfg, opts, false
	}
	defer json.destroy_value(value)
	obj, is_obj := value.(json.Object)
	if !is_obj {
		return cfg, opts, false
	}
	set_int_field(obj, "height", &opts.height)
	set_int_field(obj, "supersample", &opts.supersample)
	set_int_field(obj, "samples", &opts.samples)
	set_int_field(obj, "bounce_samples", &opts.bounce_samples)
	set_int_field(obj, "bitrate", &opts.bitrate)
	set_int_field(obj, "maxrate", &opts.maxrate)
	set_u32_field(obj, "seed", &opts.seed)
	set_f32_field(obj, "gamma", &opts.gamma)
	set_f32_field(obj, "mtf", &opts.mtf)
	set_f32_field(obj, "exposure", &opts.exposure)
	set_f32_field(obj, "contrast", &opts.contrast)
	set_f32_field(obj, "reflectance", &opts.reflectance)
	set_f32_field(obj, "thickness", &opts.thickness)
	set_f32_field(obj, "grain_radius", &opts.grain_radius)
	set_f32_field(obj, "grain_sigma", &opts.grain_sigma)
	set_f32_field(obj, "sigma_filter", &opts.sigma_filter)
	set_f32_field(obj, "film", &opts.film)
	set_f32_field(obj, "print_toe", &opts.print_toe)
	set_f32_field(obj, "print_shoulder", &opts.print_shoulder)
	set_f32_field(obj, "sat_lo", &opts.sat_lo)
	set_f32_field(obj, "sat_hi", &opts.sat_hi)
	set_f32_field(obj, "cross", &opts.cross)
	set_f32_field(obj, "reciprocity", &opts.reciprocity)
	set_bool_field(obj, "negative", &opts.negative)
	set_str_field(obj, "device", &opts.device)

	if v, ok2 := obj["emulsions"]; ok2 {
		arr, is_arr := v.(json.Array)
		if is_arr {
			for item in arr {
				eo, eok := item.(json.Object)
				if !eok {
					continue
				}
				emu := Emulsion_Cfg {
					grain_radius = 0.02,
					grain_sigma  = 0.0,
					sigma_filter = 0.08,
				}
				if d, dk := eo["dye"]; dk {
					if rgb, rok := parse_u8_arr3(d); rok {
						emu.dye = rgb
					}
				}
				set_f32_field(eo, "grain_radius", &emu.grain_radius)
				set_f32_field(eo, "grain_sigma", &emu.grain_sigma)
				set_f32_field(eo, "sigma_filter", &emu.sigma_filter)
				if mv, mk := eo["mtf_blur"]; mk {
					emu.mtf_blur = parse_maybe_f32(mv)
				}
				if mv, mk := eo["mtf_blur_max"]; mk {
					emu.mtf_blur_max = parse_maybe_f32(mv)
				}
				append(&cfg.emulsions, emu)
			}
		}
	}
	if v, ok2 := obj["filters"]; ok2 {
		arr, is_arr := v.(json.Array)
		if is_arr {
			for item in arr {
				fo, fok := item.(json.Object)
				if !fok {
					continue
				}
				flt: Filter_Cfg
				if c, ck := fo["color"]; ck {
					if rgb, rok := parse_u8_arr3(c); rok {
						flt.color = rgb
					}
				}
				append(&cfg.filters, flt)
			}
		}
	}
	if v, ok2 := obj["film_bases"]; ok2 {
		arr, is_arr := v.(json.Array)
		if is_arr {
			for item in arr {
				bo, bok := item.(json.Object)
				if !bok {
					continue
				}
				base := Film_Base_Cfg {thickness = 1.0}
				set_f32_field(bo, "thickness", &base.thickness)
				append(&cfg.bases, base)
			}
		}
	}
	if v, ok2 := obj["backs"]; ok2 {
		arr, is_arr := v.(json.Array)
		if is_arr {
			for item in arr {
				ko, kok := item.(json.Object)
				if !kok {
					continue
				}
				back: Back_Cfg
				set_f32_field(ko, "reflectance", &back.reflectance)
				append(&cfg.backs, back)
			}
		}
	}
	if v, ok2 := obj["order"]; ok2 {
		arr, is_arr := v.(json.Array)
		if is_arr {
			for item in arr {
				s, sok := item.(json.String)
				if !sok {
					continue
				}
				if kind, idx, iok := parse_order_item(s); iok {
					append(&cfg.order, Config_Item {kind = kind, index = idx})
				}
			}
		}
	} else {
		for e, i in cfg.emulsions {
			append(&cfg.order, Config_Item {kind = .Emulsion, index = i})
		}
		for f, i in cfg.filters {
			append(&cfg.order, Config_Item {kind = .Filter, index = i})
		}
		for b, i in cfg.bases {
			append(&cfg.order, Config_Item {kind = .Film_Base, index = i})
		}
		for b, i in cfg.backs {
			append(&cfg.order, Config_Item {kind = .Back, index = i})
		}
	}
	for item in cfg.order {
		switch item.kind {
		case .Emulsion:
			if item.index < 0 || item.index >= len(cfg.emulsions) {
				fail(fmt.tprintf("order item emulsion index out of range: %d", item.index))
				destroy_film_config(&cfg)
				return {}, opts, false
			}
		case .Filter:
			if item.index < 0 || item.index >= len(cfg.filters) {
				fail(fmt.tprintf("order item filter index out of range: %d", item.index))
				destroy_film_config(&cfg)
				return {}, opts, false
			}
		case .Film_Base:
			if item.index < 0 || item.index >= len(cfg.bases) {
				fail(fmt.tprintf("order item film_base index out of range: %d", item.index))
				destroy_film_config(&cfg)
				return {}, opts, false
			}
		case .Back:
			if item.index < 0 || item.index >= len(cfg.backs) {
				fail(fmt.tprintf("order item back index out of range: %d", item.index))
				destroy_film_config(&cfg)
				return {}, opts, false
			}
		}
	}
	return cfg, opts, true
}
