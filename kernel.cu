extern "C" {

struct RenderParams {
	unsigned width, height, n_samples, seed, y_offset;
	float sigma_f, sigma, r2, sigma_ln, mu_ln, max_r, ag, lambda_fac;
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
		float g1x, g1y, g2x, g2y;
		rnd_gauss(&st, &g1x, &g1y);
		rnd_gauss(&st, &g2x, &g2y);
		float xG = (float)px + p.sigma_f * g1x;
		float yG = (float)py + p.sigma_f * g2x;

		int ix = (int)floorf(xG);
		if (ix < 0) { ix = 0; }
		if (ix > W - 1) { ix = W - 1; }
		int iy = (int)floorf(yG);
		if (iy < 0) { iy = 0; }
		if (iy > H - 1) { iy = H - 1; }
		float u = __ldg(&src[iy * W + ix]);
		if (u < 0.0f) { u = 0.0f; }
		if (u > 1.0f - 1e-5f) { u = 1.0f - 1e-5f; }
		float lam = -p.lambda_fac * logf(1.0f - u);
		float exp_lam = expf(-lam);

		int min_x = (int)floorf((xG - p.max_r) / p.ag);
		int max_x = (int)floorf((xG + p.max_r) / p.ag);
		int min_y = (int)floorf((yG - p.max_r) / p.ag);
		int max_y = (int)floorf((yG + p.max_r) / p.ag);

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
					float gr2 = p.r2;
					if (p.sigma > 0.0f) {
						float rgx, rgy;
						rnd_gauss(&cs, &rgx, &rgy);
						float rad = expf(p.mu_ln + p.sigma_ln * rgx);
						if (rad > p.max_r) { rad = p.max_r; }
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
		float x_exit = x0 + dx * p.film_px / dz;
		float y_exit = y0 + dy * p.film_px / dz;
		int ix = (int)floorf(x_exit);
		int iy = (int)floorf(y_exit);
		if (ix < 0 || ix >= W || iy < 0 || iy >= H) {
			continue;
		}
		int base = (iy * W + ix) * 3;
		float col[3] = {__ldg(&front[base]), __ldg(&front[base + 1]), __ldg(&front[base + 2])};
		for (unsigned l = 0u; l < p.n_layers; l++) {
			float z_l = depth[l];
			float x_l = x0 + dx * z_l / dz;
			float y_l = y0 + dy * z_l / dz;
			int ix_l = (int)floorf(x_l);
			int iy_l = (int)floorf(y_l);
			float dv = 1.0f;
			if (ix_l >= 0 && ix_l < W && iy_l >= 0 && iy_l < H) {
				dv = __ldg(&dens[(l * (unsigned)H + (unsigned)iy_l) * (unsigned)W + (unsigned)ix_l]);
			}
			int ab = (int)l * 3;
			col[0] *= 1.0f - absorb[ab] + absorb[ab] * dv;
			col[1] *= 1.0f - absorb[ab + 1] + absorb[ab + 1] * dv;
			col[2] *= 1.0f - absorb[ab + 2] + absorb[ab + 2] * dv;
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

} // extern "C"
