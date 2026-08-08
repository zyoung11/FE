# FE

A physically-based film simulation renderer that converts photos and videos into film-like output. It simulates silver-halide grain, layered dye emulsions, halation (light bouncing inside the film base), MTF softening, and print-curve color grading.

Written in [Odin](https://odin-lang.org/), with CUDA acceleration via the driver API and optional CPU fallback.

<p align="center">
  <img src="input.jpg" width="49%" alt="Input"/>&nbsp;
  <img src="output.jpg" width="49%" alt="Output"/>
</p>
<p align="center"><em>Input &rarr; Output (auto mode)</em></p>

---

## Features

- **Physically-based pipeline**
  - Multi-layer emulsion stack with sensitizing dyes (Y/M/C color separation)
  - Monte-Carlo simulation of silver-halide grain coverage (Poisson-distributed, log-normal grain size)
  - Halation: cosine-weighted light bounce through the film base with per-layer absorption
  - MTF softening (fixed or edge-adaptive Gaussian blur)
  - Film S-curve, print toe/shoulder, shadow/highlight desaturation, color cross, exposure/contrast grading
  - Optional negative-film mode (orange mask, H&D curve) and Schwarzschild reciprocity failure (long-exposure shadow fog & warm cast)
- **Two modes**
  - `auto` — content-adaptive configuration (analyzes sharpness and histogram, tunes grain, MTF, exposure, contrast automatically)
  - `config` — full manual control via a JSON configuration file
- **Photo & video, auto-detected** — the input file extension decides whether it is treated as a photo or a video; the output file extension decides the container/format, with mismatches rejected
- **Dual backends** — CUDA (driver API, PTX embedded at compile time) with automatic CPU fallback
- **Parallel video pipeline** — 3 render workers + ordered frame encoding via ffmpeg/NVENC

---

## Dependencies

### Runtime

| Dependency | Required for | Notes |
|---|---|---|
| `ffmpeg` / `ffprobe` | Video mode | Must be in `PATH`; not needed for photos |
| NVIDIA driver | CUDA rendering + NVENC encoding | Loaded dynamically via `nvcuda.dll`; CPU fallback for photo rendering without a GPU |

### Build time

| Dependency | Notes |
|---|---|
| Odin compiler | `odin build` |
| CUDA Toolkit (`nvcc`) | Compiles `kernel.cu` → `kernel_61.ptx` / `kernel_120.ptx` (embedded into the executable at compile time) |

Everything else (stb image codec, JSON parser, thread pools, ffmpeg pipe handling) is statically linked from the Odin core and vendor libraries.

---

## Download (recommended)

Pre-built Windows binaries are published in the [Releases](https://github.com/zyoung11/FE/releases/latest) section — download `FE.exe` from the latest release and you're ready to go. No installation needed; just put `ffmpeg`/`ffprobe` in `PATH` if you want video support.

## Building

### Windows

```
build.bat
```

This compiles the CUDA kernels (`kernel.cu` → PTX) and then builds `FE.exe` (`-o:speed`). Requires a Visual Studio environment for `nvcc` (see `build-kernel.bat`).

### Cross-compile from Linux/macOS (optional)

```
build-win.sh
```

Requires mingw-w64, clang, lld-link and (optionally) nvcc.

---

## Usage

```
FE.exe --input <file> --output <file> [--auto] [--config <file.json>] [--mode color|bw]
```

| Flag | Description |
|---|---|
| `--input` | Input file (photo or video, auto-detected by extension) |
| `--output` | Output file. The **extension decides the format**: photos `.png` / `.jpg` / `.bmp`, videos `.mp4` / `.mov` / `.mkv` / etc. |
| `--auto` | Auto mode, content-adaptive configuration *(default when `--config` is not given)* |
| `--config` | Manual mode: path to a JSON configuration file |
| `--mode` | Auto mode only: `color` (default) or `bw` |

### Validation rules

- Photo input + video output extension → error
- Video input + photo output extension → error
- Unknown input/output extension → error
- `--auto` and `--config` together → error

### Examples

```bash
# Auto mode, photo → PNG (1080p default)
FE.exe --input photo.jpg --output result.png

# Auto mode, black & white, JPEG output
FE.exe --input photo.jpg --output result.jpg --mode bw

# Manual mode with a JSON config, video → MP4 (HEVC)
FE.exe --input clip.mp4 --output result.mp4 --config film.json

# Manual mode, photo
FE.exe --input photo.jpg --output result.png --config film.json
```

---

## Configuration File (JSON)

All render settings live in the JSON config file. A complete example is provided in [`film-config.example.json`](film-config.example.json).

### Render settings (top level)

| Field | Type | Default | Description |
|---|---|---|---|
| `height` | int | `0` (auto: 1080 photo / 2160 video) | Output height in pixels |
| `supersample` | int | `1` (`2` in auto mode) | Supersampling factor |
| `samples` | int | `400` (photo) / `128` (video) | Monte-Carlo samples per pixel (grain) |
| `bounce_samples` | int | `400` (photo) / `128` (video) | Monte-Carlo samples per pixel (halation) |
| `gamma` | float | `2.2` (`2.4` in auto mode) | sRGB gamma |
| `mtf` | float | `-1` (use emulsion config) | MTF softening in px (overrides `mtf_blur`/`mtf_blur_max`) |
| `exposure` | float | `0.0` | Exposure compensation |
| `contrast` | float | `1.0` | Contrast |
| `reflectance` | float | `-1` (use `backs`) | Back-side reflectance (overrides `backs`) |
| `thickness` | float | `-1` (use `film_bases`) | Film base thickness in px (overrides `film_bases`) |
| `grain_radius` | float | `-1` (use emulsion) | Grain radius in px (overrides all emulsions) |
| `grain_sigma` | float | `-1` (use emulsion) | Grain radius log-normal sigma (overrides all emulsions) |
| `sigma_filter` | float | `-1` (use emulsion) | Sample jitter sigma (overrides all emulsions) |
| `seed` | int | `12345` | Random seed |
| `device` | string | `"auto"` | `auto` \| `cpu` \| `cuda` \| `cuda:N` |
| `film` | float | `0.0` (`0.5` video, `0.5+` auto) | Film S-curve strength (0 = off, 1 = full) |
| `print_toe` | float | `-1` (auto) | Print toe strength |
| `print_shoulder` | float | `-1` (auto) | Print shoulder strength |
| `sat_lo` | float | `-1` (auto) | Shadow desaturation |
| `sat_hi` | float | `-1` (auto) | Highlight desaturation |
| `cross` | float | `-1` (auto) | Color cross coefficient |
| `reciprocity` | float | `0` | Schwarzschild reciprocity failure strength (0 = off, 1 = max). Simulates long-exposure sensitivity loss: shadows gain fog and a warm cast (per-layer differences) |
| `negative` | bool | `false` | Negative film mode: print toe/shoulder curve, warm orange mask residue in shadows, and a correction matrix for the color mask |
| `bitrate` | int | `60` | Video average bitrate (Mbps) |
| `maxrate` | int | `100` | Video peak bitrate (Mbps) |

### Film structure

```json
{
  "emulsions": [
    { "dye": [255, 255, 0], "grain_radius": 0.02, "grain_sigma": 0.001, "sigma_filter": 0.004, "mtf_blur": 0.5, "mtf_blur_max": 1.2 }
  ],
  "filters": [
    { "color": [255, 255, 0] }
  ],
  "film_bases": [
    { "thickness": 20.0 }
  ],
  "backs": [
    { "reflectance": 0.03 }
  ],
  "order": ["emulsion:0", "filter:0", "film_base:0", "back:0"]
}
```

| Section | Fields | Notes |
|---|---|---|
| `emulsions` | `dye` (RGB, sensitizing dye), `grain_radius`, `grain_sigma`, `sigma_filter`, `mtf_blur` (or `null`), `mtf_blur_max` (or `null`) | At least one required |
| `filters` | `color` (RGB) | Count must equal `emulsions` or `emulsions - 1` |
| `film_bases` | `thickness` | Film base thickness in px |
| `backs` | `reflectance` | Back-side reflectance |
| `order` | array of `"kind:index"` strings | Layer stacking order; defaults to all emulsions, then filters, then bases, then backs |

---

## Output

- **Photos**: PNG, JPEG (quality 90), BMP via stb
- **Videos**: any container supported by the local ffmpeg build (`.mp4` recommended). Encoded with HEVC (NVENC, `preset p7`, VBR, spatial/temporal AQ), audio copied/encoded as AAC 192k

## Notes

- Video encoding requires a GPU with NVENC support
- Without a CUDA-capable GPU, photo rendering falls back to the CPU backend automatically (slower)
- Grain is intentionally subtle at 4K output; increase `grain_radius` in a config file if more visible grain is desired

## Credits & Inspiration

This project is inspired by and builds upon [Retraced](https://github.com/tr-nc/retraced) by Ruitian Yang (杨瑞天) — an open-source, Taichi-accelerated simulator that recreates the look & feel of silver-halide emulsions purely in software. The physical model (layered dye emulsions, silver-halide grain, halation, MTF softening) and the overall film rendering pipeline follow its approach, reimplemented in Odin with CUDA and CPU backends.

Retraced is released under the [MIT License](https://github.com/tr-nc/retraced/blob/main/LICENSE).
