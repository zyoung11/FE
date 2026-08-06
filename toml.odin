package main

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

parse_float_array :: proc(s: string, out: []f32) -> bool {
    inner := strings.trim(s, "[]")
    parts := strings.split(inner, ",")
    defer delete(parts)
    if len(parts) != len(out) {
        return false
    }
    for p, i in parts {
        v, ok := strconv.parse_f32(strings.trim_space(p))
        if !ok {
            return false
        }
        out[i] = v
    }
    return true
}

parse_film_config :: proc(path: string) -> (cfg: Film_Config, ok: bool) {
    data, err := os.read_entire_file_from_path(path, context.allocator)
    if err != nil {
        return cfg, false
    }
    defer delete(data)
    return parse_film_config_text(string(data))
}

parse_film_config_text :: proc(text: string) -> (cfg: Film_Config, ok: bool) {
    current_kind := Config_Kind.Emulsion
    current_index := -1
    it := text
    for line in strings.split_lines_iterator(&it) {
        s := strings.trim_space(line)
        if s == "" || s[0] == '#' {
            continue
        }
        if hash := strings.index(s, "#"); hash >= 0 {
            s = strings.trim_space(s[:hash])
            if s == "" {
                continue
            }
        }
        if strings.has_prefix(s, "[[") {
            end := strings.index(s, "]]")
            if end < 0 {
                continue
            }
            name := strings.trim_space(s[2:end])
            switch name {
            case "emulsion":
                append(&cfg.emulsions, Emulsion_Cfg {
                    grain_radius = 0.02,
                    grain_sigma  = 0.0,
                    sigma_filter = 0.08,
                })
                current_kind = .Emulsion
                current_index = len(cfg.emulsions) - 1
                append(&cfg.order, Config_Item {kind = .Emulsion, index = current_index})
            case "filter":
                append(&cfg.filters, Filter_Cfg {})
                current_kind = .Filter
                current_index = len(cfg.filters) - 1
                append(&cfg.order, Config_Item {kind = .Filter, index = current_index})
            case "film_base":
                append(&cfg.bases, Film_Base_Cfg {thickness = 1.0})
                current_kind = .Film_Base
                current_index = len(cfg.bases) - 1
                append(&cfg.order, Config_Item {kind = .Film_Base, index = current_index})
            case "back":
                append(&cfg.backs, Back_Cfg {reflectance = 0.0})
                current_kind = .Back
                current_index = len(cfg.backs) - 1
                append(&cfg.order, Config_Item {kind = .Back, index = current_index})
            }
            continue
        }
        eq := strings.index(s, "=")
        if eq < 0 || current_index < 0 {
            continue
        }
        key := strings.trim_space(s[:eq])
        val := strings.trim_space(s[eq + 1:])
        if strings.has_prefix(val, "[") {
            arr: [3]f32
            if !parse_float_array(val, arr[:]) {
                continue
            }
            switch current_kind {
            case .Emulsion:
                emu := &cfg.emulsions[current_index]
                switch key {
                case "sensitising_dye_color":
                    emu.dye = [3]u8 {u8(arr[0]), u8(arr[1]), u8(arr[2])}
                case "mtf_blur":
                    emu.mtf_blur = arr[0]
                case "mtf_blur_max":
                    emu.mtf_blur_max = arr[0]
                }
            case .Filter:
                flt := &cfg.filters[current_index]
                switch key {
                case "color":
                    flt.color = [3]u8 {u8(arr[0]), u8(arr[1]), u8(arr[2])}
                }
            case .Film_Base:
                base := &cfg.bases[current_index]
                switch key {
                case "thickness":
                    base.thickness = arr[0]
                }
            case .Back:
                back := &cfg.backs[current_index]
                switch key {
                case "reflectance":
                    back.reflectance = arr[0]
                }
            }
            continue
        }
        v, vok := strconv.parse_f32(val)
        if !vok {
            continue
        }
        switch current_kind {
        case .Emulsion:
            emu := &cfg.emulsions[current_index]
            switch key {
            case "grain_radius":
                emu.grain_radius = v
            case "grain_sigma":
                emu.grain_sigma = v
            case "sigma_filter":
                emu.sigma_filter = v
            case "mtf_blur":
                emu.mtf_blur = v
            case "mtf_blur_max":
                emu.mtf_blur_max = v
            }
        case .Filter:
        case .Film_Base:
            base := &cfg.bases[current_index]
            switch key {
            case "thickness":
                base.thickness = v
            }
        case .Back:
            back := &cfg.backs[current_index]
            switch key {
            case "reflectance":
                back.reflectance = v
            }
        }
    }
    return cfg, true
}
