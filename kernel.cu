extern "C" {

// Per-channel penetration depth factors (red penetrates deepest)
__constant__ float CHAN_DEPTH[3] = {1.25f, 1.0f, 0.8f};
// Front-interface reflectance for the second bounce
#define HALATION_R_FRONT 0.3f

struct RenderParams {
	unsigned width, height, n_samples, seed, y_offset;
	float sigma_f, sigma, r2, sigma_ln, mu_ln, max_r, ag, lambda_fac;
	float frame_off_x, frame_off_y;
};

struct BounceParams {
	unsigned width, height, n_layers, n_samples, seed, y_offset;
	float film_px;
};

__device__ unsigned wang(unsigned s) {
	s = (s ^ 61u) ^ (s >> 16);
	s *= 9u;
	s ^= s >> 4;
	s *= 668265261u;
	s ^= s >> 15;
	return s;
}

__device__ unsigned xor_shift(unsigned s) {
	s ^= s << 13;
	s ^= s >> 17;
	s ^= s << 5;
	return s;
}

__device__ float rnd01(unsigned* s) {
	*s = xor_shift(*s);
	return (float)(*s) * (1.0f / 4294967295.0f);
}

__device__ void rnd_gauss(unsigned* s, float* g1, float* g2) {
	float r1 = rnd01(s);
	float r2 = rnd01(s);
	float r = sqrtf(-2.0f * logf(r1 + 1e-12f));
	*g1 = r * cosf(2.0f * 3.14159265358979f * r2);
	*g2 = r * sinf(2.0f * 3.14159265358979f * r2);
}

__device__ unsigned rnd_poisson(unsigned* s, float lam, float exp_lam) {
	float u = rnd01(s);
	float prod = exp_lam;
	float summ = exp_lam;
	unsigned x = 0u;
	unsigned lim = (unsigned)floorf(lam + 6.0f * sqrtf(lam)) + 1u;
	while (u > summ && x < lim) {
		x += 1u;
		prod *= lam / (float)x;
		summ += prod;
	}
	return x;
}

__device__ void hemi_cosine(unsigned* s, float* dx, float* dy, float* dz) {
	float r1 = rnd01(s);
	float r2 = rnd01(s);
	float z = sqrtf(r1);
	float r = sqrtf(1.0f - z * z);
	float phi = 2.0f * 3.14159265358979f * r2;
	*dx = r * cosf(phi);
	*dy = r * sinf(phi);
	*dz = z;
}

__global__ void render_kernel(const float* __restrict__ src, float* __restrict__ dst, RenderParams p) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y + (int)p.y_offset;
	if (px >= (int)p.width || py >= (int)p.height) {
		return;
	}

	unsigned fseed = wang(p.seed);
	unsigned st = wang((unsigned)py * 73856093u ^ (unsigned)px * 19349663u ^ fseed);
	float hit = 0.0f;
	int W = (int)p.width;
	int H = (int)p.height;

	for (unsigned s = 0u; s < p.n_samples; s++) {
		float xS = (float)px;
		float yS = (float)py;
		if (p.sigma_f > 1e-4f) {
			float g1x, g1y, g2x, g2y;
			rnd_gauss(&st, &g1x, &g1y);
			rnd_gauss(&st, &g2x, &g2y);
			xS += p.sigma_f * g1x;
			yS += p.sigma_f * g2x;
		}
		float xG = xS + p.frame_off_x;
		float yG = yS + p.frame_off_y;

		int ix = (int)floorf(xS);
		if (ix < 0) { ix = 0; }
		if (ix > W - 1) { ix = W - 1; }
		int iy = (int)floorf(yS);
		if (iy < 0) { iy = 0; }
		if (iy > H - 1) { iy = H - 1; }
		float u = __ldg(&src[iy * W + ix]);
		if (u < 0.0f) { u = 0.0f; }
		if (u > 1.0f - 1e-5f) { u = 1.0f - 1e-5f; }
		float lam = -p.lambda_fac * logf(1.0f - u);
		float decay = 1.0f - 0.7f * fminf(1.0f, fmaxf(0.0f, (u - 0.5f) / 0.45f));
		lam *= decay;
		if (lam < 1e-5f) {
			continue;
		}
		float r_scale = 1.0f / sqrtf(decay);
		float max_r_eff = p.max_r * r_scale;
		float exp_lam = expf(-lam);

		int min_x = (int)floorf((xG - max_r_eff) / p.ag);
		int max_x = (int)floorf((xG + max_r_eff) / p.ag);
		int min_y = (int)floorf((yG - max_r_eff) / p.ag);
		int max_y = (int)floorf((yG + max_r_eff) / p.ag);

		bool covered = false;
		for (int cx = min_x; cx <= max_x && !covered; cx++) {
			for (int cy = min_y; cy <= max_y && !covered; cy++) {
				unsigned cs = wang(((((unsigned)cy & 0xFFFFu) << 16) | ((unsigned)cx & 0xFFFFu)) + fseed);
				unsigned n_grains = rnd_poisson(&cs, lam, exp_lam);
				for (unsigned z = 0u; z < n_grains; z++) {
					float ru = rnd01(&cs);
					float rv = rnd01(&cs);
					float xc = p.ag * ((float)cx + ru);
					float yc = p.ag * ((float)cy + rv);
					float gr2 = p.r2 * r_scale * r_scale;
					if (p.sigma > 0.0f) {
						float rgx, rgy;
						rnd_gauss(&cs, &rgx, &rgy);
						float rad = expf(p.mu_ln + p.sigma_ln * rgx);
						if (rad > p.max_r) { rad = p.max_r; }
						rad *= r_scale;
						gr2 = rad * rad;
					}
					float dx = xc - xG;
					float dy = yc - yG;
					if (dx * dx + dy * dy < gr2) {
						covered = true;
						break;
					}
				}
			}
		}
		if (covered) {
			hit += 1.0f;
		}
	}

	dst[py * W + px] = 1.0f - hit / (float)p.n_samples;
}

__global__ void bounce_kernel(
	const float* __restrict__ front,
	float* __restrict__ bounced,
	const float* __restrict__ dens,
	const float* __restrict__ absorb,
	const float* __restrict__ depth,
	BounceParams p
) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y + (int)p.y_offset;
	if (px >= (int)p.width || py >= (int)p.height) {
		return;
	}

	unsigned fseed = wang(p.seed);
	unsigned st = wang((unsigned)py * 9781u ^ (unsigned)px * 6271u ^ fseed);
	float acc[3] = {0.0f, 0.0f, 0.0f};
	float x0 = (float)px;
	float y0 = (float)py;
	int W = (int)p.width;
	int H = (int)p.height;

	for (unsigned s = 0u; s < p.n_samples; s++) {
		float dx, dy, dz;
		hemi_cosine(&st, &dx, &dy, &dz);
		if (dz < 1e-4f) {
			continue;
		}
		float x1[3], y1[3];
		float col[3] = {0.0f, 0.0f, 0.0f};
		for (int c = 0; c < 3; c++) {
			float fp = p.film_px * CHAN_DEPTH[c];
			x1[c] = x0 + dx * fp / dz;
			y1[c] = y0 + dy * fp / dz;
			int ix = (int)floorf(x1[c]);
			int iy = (int)floorf(y1[c]);
			if (ix < 0 || ix >= W || iy < 0 || iy >= H) {
				continue;
			}
			col[c] = __ldg(&front[(iy * W + ix) * 3 + c]);
			for (unsigned l = 0u; l < p.n_layers; l++) {
				float z_l = depth[l] * CHAN_DEPTH[c];
				float x_l = x0 + dx * z_l / dz;
				float y_l = y0 + dy * z_l / dz;
				int ix_l = (int)floorf(x_l);
				int iy_l = (int)floorf(y_l);
				float dv = 1.0f;
				if (ix_l >= 0 && ix_l < W && iy_l >= 0 && iy_l < H) {
					dv = __ldg(&dens[(l * (unsigned)H + (unsigned)iy_l) * (unsigned)W + (unsigned)ix_l]);
				}
				int ab = (int)l * 3;
				col[c] *= 1.0f - absorb[ab + c] + absorb[ab + c] * dv;
			}
		}
		float dx2, dy2, dz2;
		hemi_cosine(&st, &dx2, &dy2, &dz2);
		if (dz2 >= 1e-4f) {
			for (int c = 0; c < 3; c++) {
				float fp = p.film_px * CHAN_DEPTH[c];
				float x2 = x1[c] + dx2 * fp / dz2;
				float y2 = y1[c] + dy2 * fp / dz2;
				int ix = (int)floorf(x2);
				int iy = (int)floorf(y2);
				if (ix < 0 || ix >= W || iy < 0 || iy >= H) {
					continue;
				}
				float col2 = __ldg(&front[(iy * W + ix) * 3 + c]);
				for (unsigned l = 0u; l < p.n_layers; l++) {
					float z_l = depth[l] * CHAN_DEPTH[c];
					float x_l = x1[c] + dx2 * z_l / dz2;
					float y_l = y1[c] + dy2 * z_l / dz2;
					int ix_l = (int)floorf(x_l);
					int iy_l = (int)floorf(y_l);
					float dv = 1.0f;
					if (ix_l >= 0 && ix_l < W && iy_l >= 0 && iy_l < H) {
						dv = __ldg(&dens[(l * (unsigned)H + (unsigned)iy_l) * (unsigned)W + (unsigned)ix_l]);
					}
					int ab = (int)l * 3;
					col2 *= 1.0f - absorb[ab + c] + absorb[ab + c] * dv;
				}
				col[c] += col2 * HALATION_R_FRONT;
			}
		}
		acc[0] += col[0];
		acc[1] += col[1];
		acc[2] += col[2];
	}

	float inv = 1.0f / (float)((int)p.n_samples > 1 ? (int)p.n_samples : 1);
	int obase = (py * W + px) * 3;
	bounced[obase] = acc[0] * inv;
	bounced[obase + 1] = acc[1] * inv;
	bounced[obase + 2] = acc[2] * inv;
}

__global__ void lap_kernel(const float* __restrict__ src, float* __restrict__ dst, unsigned w, unsigned h) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= w * h) {
		return;
	}
	unsigned x = i % w;
	unsigned y = i / w;
	if (x == 0u || y == 0u || x == w - 1u || y == h - 1u) {
		dst[i] = 0.0f;
		return;
	}
	dst[i] = fabsf(4.0f * src[i] - src[i - w] - src[i + w] - src[i - 1] - src[i + 1]);
}

__global__ void gauss_h_kernel(const float* __restrict__ src, float* __restrict__ dst, unsigned w, unsigned h, const float* __restrict__ kernel, int r) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= w * h) {
		return;
	}
	unsigned x = i % w;
	unsigned y = i / w;
	float acc = 0.0f;
	float ksum = 0.0f;
	for (int j = 0; j <= 2 * r; j++) {
		int sx = (int)x + j - r;
		if (sx >= 0 && sx < (int)w) {
			float k = kernel[j];
			acc += src[y * w + (unsigned)sx] * k;
			ksum += k;
		}
	}
	dst[i] = acc / ksum;
}

__global__ void gauss_v_kernel(const float* __restrict__ src, float* __restrict__ dst, unsigned w, unsigned h, const float* __restrict__ kernel, int r) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= w * h) {
		return;
	}
	unsigned x = i % w;
	unsigned y = i / w;
	float acc = 0.0f;
	float ksum = 0.0f;
	for (int j = 0; j <= 2 * r; j++) {
		int sy = (int)y + j - r;
		if (sy >= 0 && sy < (int)h) {
			float k = kernel[j];
			acc += src[(unsigned)sy * w + x] * k;
			ksum += k;
		}
	}
	dst[i] = acc / ksum;
}

__global__ void tmap_kernel(const float* __restrict__ energy, float* __restrict__ t, const float* __restrict__ sum, unsigned n) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) {
		return;
	}
	float mean = sum[0] / (float)n;
	float v = energy[i] / (2.0f * mean + 1e-6f);
	t[i] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

__global__ void sum_kernel(const float* __restrict__ src, float* __restrict__ out, unsigned n) {
	__shared__ float sdata[256];
	unsigned tid = threadIdx.x;
	unsigned stride = gridDim.x * blockDim.x;
	float acc = 0.0f;
	for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
		acc += src[i];
	}
	sdata[tid] = acc;
	__syncthreads();
	for (unsigned s = 128u; s > 0u; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0u) {
		out[blockIdx.x] = sdata[0];
	}
}

__global__ void sum_final_kernel(const float* __restrict__ partials, float* __restrict__ out, unsigned nblocks) {
	__shared__ float sdata[256];
	unsigned tid = threadIdx.x;
	float acc = 0.0f;
	for (unsigned i = tid; i < nblocks; i += 256u) {
		acc += partials[i];
	}
	sdata[tid] = acc;
	__syncthreads();
	for (unsigned s = 128u; s > 0u; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0u) {
		out[0] = sdata[0];
	}
}

__global__ void blend_kernel(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ t, float* __restrict__ dst, unsigned n) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) {
		return;
	}
	dst[i] = a[i] * (1.0f - t[i]) + b[i] * t[i];
}

struct Vec3 { float x, y, z; };

struct GradingParams {
	float gamma, exposure, contrast, film, print_toe, print_shoulder, sat_lo, sat_hi, cross;
	int negative;
};

__global__ void linearize_kernel(const unsigned char* __restrict__ in, float* __restrict__ out, float gamma, unsigned n) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) {
		return;
	}
	for (int c = 0; c < 3; c++) {
		float a = (float)in[i * 3 + c] / 255.0f;
		out[i * 3 + c] = a <= 0.04045f ? a / 12.92f : powf((a + 0.055f) / 1.055f, gamma);
	}
}

__global__ void src_kernel(const float* __restrict__ ray_front, float* __restrict__ dst, Vec3 expo_w, unsigned n) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) {
		return;
	}
	float lum = ray_front[i * 3 + 0] * expo_w.x + ray_front[i * 3 + 1] * expo_w.y + ray_front[i * 3 + 2] * expo_w.z;
	if (lum < 0.0f) { lum = 0.0f; }
	if (lum > 1.0f) { lum = 1.0f; }
	dst[i] = 1.0f - lum;
}

__global__ void recip_kernel(float* __restrict__ src, float p, unsigned n) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) {
		return;
	}
	float base = 1.0f - src[i];
	if (base > 0.001f && base < 0.999f) {
		float expo = powf(base, 1.0f - p);
		src[i] = 1.0f - powf(base, expo);
	}
}

__global__ void front_update_kernel(float* __restrict__ ray_front, const float* __restrict__ neg, Vec3 absorb, unsigned n) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) {
		return;
	}
	float dd = neg[i];
	ray_front[i * 3 + 0] *= 1.0f - absorb.x * (1.0f - dd);
	ray_front[i * 3 + 1] *= 1.0f - absorb.y * (1.0f - dd);
	ray_front[i * 3 + 2] *= 1.0f - absorb.z * (1.0f - dd);
}

__global__ void filter_kernel(float* __restrict__ ray_front, Vec3 col, unsigned n) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) {
		return;
	}
	ray_front[i * 3 + 0] *= col.x / 255.0f;
	ray_front[i * 3 + 1] *= col.y / 255.0f;
	ray_front[i * 3 + 2] *= col.z / 255.0f;
}

__global__ void assemble_kernel(float* __restrict__ front, const float* __restrict__ dens, Vec3 absorb, unsigned n) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) {
		return;
	}
	float dd = dens[i];
	front[i * 3 + 0] *= 1.0f - absorb.x + absorb.x * dd;
	front[i * 3 + 1] *= 1.0f - absorb.y + absorb.y * dd;
	front[i * 3 + 2] *= 1.0f - absorb.z + absorb.z * dd;
}

__global__ void min_kernel(float* __restrict__ front, const float* __restrict__ smooth, unsigned n) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) {
		return;
	}
	if (smooth[i] < front[i]) {
		front[i] = smooth[i];
	}
}

__global__ void hdr_kernel(const float* __restrict__ front, float* __restrict__ dst, unsigned n) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) {
		return;
	}
	float x = front[i];
	if (x < 0.0f) { x = 0.0f; }
	if (x > 1.0f - 1e-5f) { x = 1.0f - 1e-5f; }
	float v = x / (1.0f - x + 1e-5f);
	if (v < 0.0f) { v = 0.0f; }
	if (v > 3.0f) { v = 3.0f; }
	dst[i] = v;
}

__global__ void final_kernel(float* __restrict__ dst, const float* __restrict__ bounced, float back_refl, unsigned n) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) {
		return;
	}
	float v = dst[i] + bounced[i] * back_refl;
	if (v < 0.0f) { v = 0.0f; }
	if (v > 1.0f) { v = 1.0f; }
	dst[i] = v;
}

__device__ float linear_to_srgb_d(float x, float gamma) {
	if (x <= 0.0031308f) {
		return x * 12.92f;
	}
	return 1.055f * powf(x, 1.0f / gamma) - 0.055f;
}

__device__ float print_curve_d(float x, float toe, float shoulder) {
	if (x <= 0.0f) {
		return 0.0f;
	}
	if (x >= 1.0f) {
		return 1.0f;
	}
	float t = 0.25f * toe;
	float s = 0.25f * shoulder;
	if (x <= t) {
		return x * x / (2.0f * t);
	}
	if (x >= 1.0f - s) {
		float u = (1.0f - x) / s;
		return 1.0f - u * u / 2.0f;
	}
	float k = (1.0f - s / 2.0f - t / 2.0f) / (1.0f - s - t);
	return t / 2.0f + k * (x - t);
}

__device__ float filmic_curve_d(float x, float strength) {
	if (strength <= 0.0f) {
		return x;
	}
	float a = 2.51f * x;
	float c = 2.43f * x;
	float y = (x * (a + 0.03f)) / (x * (c + 0.59f) + 0.14f);
	return x * (1.0f - strength) + y * strength;
}

__global__ void out8_kernel(const float* __restrict__ final, unsigned char* __restrict__ out8, GradingParams p, unsigned n) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) {
		return;
	}
	float r = linear_to_srgb_d(final[i * 3 + 0], p.gamma);
	float g = linear_to_srgb_d(final[i * 3 + 1], p.gamma);
	float b = linear_to_srgb_d(final[i * 3 + 2], p.gamma);
	r = (r - 0.5f + p.exposure * 0.5f) * p.contrast + 0.5f;
	g = (g - 0.5f + p.exposure * 0.5f) * p.contrast + 0.5f;
	b = (b - 0.5f + p.exposure * 0.5f) * p.contrast + 0.5f;
	r = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
	g = g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g);
	b = b < 0.0f ? 0.0f : (b > 1.0f ? 1.0f : b);
	float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
	if (p.negative) {
		float toe = p.print_toe < 0.0f ? 0.3f : p.print_toe;
		float shoulder = p.print_shoulder < 0.0f ? 0.3f : p.print_shoulder;
		float y = print_curve_d(luma, toe, shoulder);
		r = y + (r - luma);
		g = y + (g - luma);
		b = y + (b - luma);
		float mask_w = (1.0f - y) * (1.0f - y) * 0.5f;
		r = r + mask_w * 0.12f;
		b = b - mask_w * 0.08f;
		float nr = r * 1.02f - g * 0.02f;
		float nb = b * 0.98f + r * 0.02f;
		r = nr;
		b = nb;
	}
	float sat_w = 1.0f - p.sat_lo * (1.0f - luma) * (1.0f - luma) - p.sat_hi * luma * luma;
	r = luma + (r - luma) * sat_w;
	g = luma + (g - luma) * sat_w;
	b = luma + (b - luma) * sat_w;
	if (p.film > 0.0f) {
		float luma2 = 0.2126f * r + 0.7152f * g + 0.0722f * b;
		float y = filmic_curve_d(luma2, p.film);
		float tint = p.film * 0.04f;
		float rr = y + (r - luma2) + (y - luma2) * tint;
		float gg = y + (g - luma2);
		float bb = y + (b - luma2) - (y - luma2) * tint * 0.8f;
		float cr = 1.0f - p.cross;
		r = rr * cr + (gg + bb) * p.cross * 0.5f;
		g = gg * cr + (rr + bb) * p.cross * 0.5f;
		b = bb * cr + (rr + gg) * p.cross * 0.5f;
	}
	float v0 = 255.0f * r;
	float v1 = 255.0f * g;
	float v2 = 255.0f * b;
	out8[i * 3 + 0] = (unsigned char)(v0 < 0.0f ? 0.0f : (v0 > 255.0f ? 255.0f : v0));
	out8[i * 3 + 1] = (unsigned char)(v1 < 0.0f ? 0.0f : (v1 > 255.0f ? 255.0f : v1));
	out8[i * 3 + 2] = (unsigned char)(v2 < 0.0f ? 0.0f : (v2 > 255.0f ? 255.0f : v2));
}

__global__ void box_down3_kernel(const float* __restrict__ src, float* __restrict__ dst, unsigned sw, unsigned sh, unsigned dw, unsigned dh) {
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= dw || y >= dh) {
		return;
	}
	unsigned sx0 = x * sw / dw;
	unsigned sx1 = (x + 1) * sw / dw;
	if (sx1 <= sx0) { sx1 = sx0 + 1; }
	unsigned sy0 = y * sh / dh;
	unsigned sy1 = (y + 1) * sh / dh;
	if (sy1 <= sy0) { sy1 = sy0 + 1; }
	float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
	for (unsigned sy = sy0; sy < sy1; sy++) {
		for (unsigned sx = sx0; sx < sx1; sx++) {
			unsigned si = (sy * sw + sx) * 3;
			acc0 += src[si + 0];
			acc1 += src[si + 1];
			acc2 += src[si + 2];
		}
	}
	float inv = 1.0f / (float)((sx1 - sx0) * (sy1 - sy0));
	unsigned di = (y * dw + x) * 3;
	dst[di + 0] = acc0 * inv;
	dst[di + 1] = acc1 * inv;
	dst[di + 2] = acc2 * inv;
}

__global__ void box_down_planes_kernel(const float* __restrict__ src, float* __restrict__ dst, unsigned sw, unsigned sh, unsigned dw, unsigned dh, unsigned planes) {
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= dw || y >= dh) {
		return;
	}
	unsigned sx0 = x * sw / dw;
	unsigned sx1 = (x + 1) * sw / dw;
	if (sx1 <= sx0) { sx1 = sx0 + 1; }
	unsigned sy0 = y * sh / dh;
	unsigned sy1 = (y + 1) * sh / dh;
	if (sy1 <= sy0) { sy1 = sy0 + 1; }
	float inv = 1.0f / (float)((sx1 - sx0) * (sy1 - sy0));
	for (unsigned l = 0u; l < planes; l++) {
		float acc = 0.0f;
		for (unsigned sy = sy0; sy < sy1; sy++) {
			for (unsigned sx = sx0; sx < sx1; sx++) {
				acc += src[(l * sh + sy) * sw + sx];
			}
		}
		dst[(l * dh + y) * dw + x] = acc * inv;
	}
}

__global__ void bilinear_up3_kernel(const float* __restrict__ src, float* __restrict__ dst, unsigned sw, unsigned sh, unsigned dw, unsigned dh) {
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= dw || y >= dh) {
		return;
	}
	float sx = (float)sw / (float)dw;
	float sy = (float)sh / (float)dh;
	float fy = ((float)y + 0.5f) * sy - 0.5f;
	int y0 = (int)fy;
	if (y0 < 0) { y0 = 0; }
	if (y0 > (int)sh - 1) { y0 = (int)sh - 1; }
	int y1 = y0 + 1;
	if (y1 > (int)sh - 1) { y1 = (int)sh - 1; }
	float ty = fy - (float)y0;
	if (ty < 0.0f) { ty = 0.0f; }
	if (ty > 1.0f) { ty = 1.0f; }
	float fx = ((float)x + 0.5f) * sx - 0.5f;
	int x0 = (int)fx;
	if (x0 < 0) { x0 = 0; }
	if (x0 > (int)sw - 1) { x0 = (int)sw - 1; }
	int x1 = x0 + 1;
	if (x1 > (int)sw - 1) { x1 = (int)sw - 1; }
	float tx = fx - (float)x0;
	if (tx < 0.0f) { tx = 0.0f; }
	if (tx > 1.0f) { tx = 1.0f; }
	for (int c = 0; c < 3; c++) {
		float v00 = src[(y0 * sw + x0) * 3 + c];
		float v10 = src[(y0 * sw + x1) * 3 + c];
		float v01 = src[(y1 * sw + x0) * 3 + c];
		float v11 = src[(y1 * sw + x1) * 3 + c];
		dst[(y * dw + x) * 3 + c] = (v00 * (1.0f - tx) + v10 * tx) * (1.0f - ty) + (v01 * (1.0f - tx) + v11 * tx) * ty;
	}
}

} // extern "C"
