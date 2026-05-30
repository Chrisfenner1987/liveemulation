// FilmGrain OFX - physically-based stochastic film-grain model.
// Copyright (c) 2026 Chris Fenner. SPDX-License-Identifier: BSD-3-Clause
//
// This implements the local Monte-Carlo evaluation of the Boolean (Poisson)
// grain model described in:
//   A. Newson, B. Galerne, J. Morel, J. Delon,
//   "A Stochastic Film Grain Model for Resolution-Independent Rendering",
//   Computer Graphics Forum, 2017.
//
// Each output pixel value u in [0,1] is treated as the local coverage of a
// silver-halide Boolean model: grains (discs of mean radius r) are scattered as
// a Poisson process whose intensity lambda is chosen so the expected uncovered
// fraction equals (1-u). We render by drawing N sample points inside the pixel
// aperture and measuring the fraction covered by at least one grain. The
// spatial randomness of the grains produces signal-dependent grain exactly the
// way real film does (most visible in the mid-tones, suppressed in deep
// shadow / clipped highlight).
//
// IMPORTANT: The Metal kernel in FilmGrainMetal.mm mirrors this math 1:1.
// If you change the model here, change it there too (search "KEEP IN SYNC").

#ifndef FILMGRAIN_GRAINMODEL_H
#define FILMGRAIN_GRAINMODEL_H

#include <cmath>

namespace grain {

static const float kPI = 3.14159265358979323846f;

// --- Hash-based PRNG ---------------------------------------------------------
// Deterministic and stateless given a seed, so the grain field is identical on
// CPU and GPU and independent of tiling / render order / resolution.

// PCG-style output hash (Jarzynski & Olano, "Hash Functions for GPU Rendering").
inline unsigned int pcgHash(unsigned int v) {
    unsigned int state = v * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

inline unsigned int hash2(unsigned int a, unsigned int b) {
    return pcgHash(a ^ (b * 0x9e3779b9u));
}

inline unsigned int hash3(unsigned int a, unsigned int b, unsigned int c) {
    return pcgHash(hash2(a, b) ^ (c * 0x85ebca6bu));
}

// A tiny stateful stream seeded from a hash; returns uniforms in [0,1).
struct RNG {
    unsigned int state;
    explicit RNG(unsigned int s) : state(s ? s : 0x12345u) {}
    inline unsigned int nextUint() {
        state = state * 747796405u + 2891336453u;
        unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }
    inline float nextFloat() {
        // 24-bit mantissa worth of randomness in [0,1)
        return (nextUint() >> 8) * (1.0f / 16777216.0f);
    }
};

// Knuth's Poisson sampler. Mean stays small (~<=3) for sane grain sizes, so the
// expected loop count is tiny.
inline int samplePoisson(RNG& rng, float mean) {
    if (mean <= 0.0f) return 0;
    if (mean > 30.0f) mean = 30.0f;           // safety clamp; never hit in practice
    float L = std::exp(-mean);
    int k = 0;
    float p = 1.0f;
    do {
        ++k;
        p *= rng.nextFloat();
    } while (p > L);
    return k - 1;
}

// Variant taking a precomputed L = exp(-mean) (mean is constant per pixel).
inline int samplePoissonL(RNG& rng, float L) {
    int k = 0;
    float p = 1.0f;
    do { ++k; p *= rng.nextFloat(); } while (p > L);
    return k - 1;
}

// --- Parameters --------------------------------------------------------------
struct Params {
    float grainRadius;   // mean grain radius, in pixels at the processing scale
    float sigmaR;        // stddev of the (log-normal) radius; 0 => monodisperse
    int   nSamples;      // Monte-Carlo samples per pixel (estimator quality)
    unsigned int seed;   // global seed (combine frame number for animation)

    Params() : grainRadius(0.85f), sigmaR(0.0f), nSamples(64), seed(2026u) {}
};

// Log-normal parameters (muLN, sigmaLN) of the underlying normal, derived from a
// target arithmetic mean and stddev of the radius.
inline void lognormalParams(float mean, float stddev, float& muLN, float& sigmaLN) {
    if (stddev <= 1e-6f || mean <= 1e-6f) { muLN = std::log(mean > 1e-6f ? mean : 1e-6f); sigmaLN = 0.0f; return; }
    float v = stddev * stddev;
    float m2 = mean * mean;
    sigmaLN = std::sqrt(std::log(1.0f + v / m2));
    muLN = std::log(m2 / std::sqrt(m2 + v));
}

// Soft, compact grain kernel: a smooth quartic bump of support radius R.
//   K(d) = (1 - (d/R)^2)^2  for d < R, else 0.
// Analytic integrals over the plane (used for the shot-noise mean/variance):
//   INT K  dA = pi R^2 / 3 ,  INT K^2 dA = pi R^2 / 5 .
static const float kKernelSupport = 2.2f;  // R = kKernelSupport * grainRadius

// --- Core: render one channel of one pixel -----------------------------------
// value  : clean input value in [0,1]
// px,py  : absolute pixel coordinates in the full image (stable across tiles)
// chan   : 0/1/2 -> independent grain field per channel (decorrelated layers)
//
// Returns a ZERO-MEAN, ~unit-variance grain fluctuation (both signs). The caller
// scales it by the per-level target RMS (the calibrated curve) and Amount, then
// ADDS it to the image. Because it is bipolar and built from overlapping soft
// grains, it darkens and lightens like real emulsion instead of stamping bright
// dots on top. (KEEP IN SYNC with the Metal kernel in FilmGrainMetal.mm.)
inline float renderPixel(float value, int px, int py, int chan, const Params& p) {
    float u = value;
    if (u < 0.0f) u = 0.0f;
    if (u > 0.999f) u = 0.999f;
    if (u <= 0.0f) return 0.0f;

    const float r = p.grainRadius > 1e-4f ? p.grainRadius : 1e-4f;

    float muLN, sigmaLN;
    lognormalParams(r, p.sigmaR, muLN, sigmaLN);
    const float expR2 = std::exp(2.0f * muLN + 2.0f * sigmaLN * sigmaLN); // E[R^2]
    const float meanGrainArea = kPI * expR2;
    const float lambda = -std::log(1.0f - u) / meanGrainArea; // grains per unit area

    const float R = kKernelSupport * r;     // soft kernel support radius
    const float R2 = R * R;

    // Shot-noise statistics of F = sum_g K(p - c_g) (Campbell's theorem):
    //   E[F]   = lambda * INT K  = lambda * pi R^2 / 3
    //   Var[F] = lambda * INT K^2 = lambda * pi R^2 / 5
    const float Efield = lambda * (kPI * R2 / 3.0f);
    const float varF   = lambda * (kPI * R2 / 5.0f);
    if (varF <= 1e-12f) return 0.0f;
    const float invSigmaF = 1.0f / std::sqrt(varF);

    // Grain cells sized to the kernel support; a grain within R of the sample
    // point lives in the 3x3 neighbourhood.
    const float cell = R > 1e-4f ? R : 1e-4f;
    float meanPerCell = lambda * cell * cell;
    if (meanPerCell > 30.0f) meanPerCell = 30.0f;
    const float poissonL = std::exp(-meanPerCell);

    const unsigned int seedC = p.seed + (unsigned int)chan * 0x6d2b79f5u;
    unsigned int pbase = (unsigned int)(px * 1973 + 9277) ^ ((unsigned int)(py * 26699 + 7919) * 0x9e3779b9u) ^ seedC;

    // Optional sub-pixel supersampling (Quality) to anti-alias very fine grain;
    // default 1 (center sample) since the field is smooth at the pixel scale.
    int N = p.nSamples > 0 ? p.nSamples : 1;
    if (N > 16) N = 16;
    int g = (int)std::floor(std::sqrt((float)N) + 0.5f); if (g < 1) g = 1;
    float inv = 1.0f / (float)g;

    float accum = 0.0f;
    for (int sj = 0; sj < g; ++sj) {
        for (int si = 0; si < g; ++si) {
            unsigned int hs = pcgHash(pbase + (unsigned int)(sj * g + si) * 0x27d4eb2fu);
            float jx = (hs & 0xffffu) * (1.0f / 65536.0f);
            float jy = ((hs >> 16) & 0xffffu) * (1.0f / 65536.0f);
            float spx = (float)px + ((float)si + jx) * inv;
            float spy = (float)py + ((float)sj + jy) * inv;

            int cxMin = (int)std::floor((spx - R) / cell);
            int cxMax = (int)std::floor((spx + R) / cell);
            int cyMin = (int)std::floor((spy - R) / cell);
            int cyMax = (int)std::floor((spy + R) / cell);

            float F = 0.0f;
            for (int cy = cyMin; cy <= cyMax; ++cy) {
                for (int cx = cxMin; cx <= cxMax; ++cx) {
                    RNG crng(hash3((unsigned int)(cx * 0x1f1f1f1fu + 0x2c1b3c6du),
                                   (unsigned int)(cy * 0x2545f491u + 0x1b873593u),
                                   seedC));
                    int nGrains = samplePoissonL(crng, poissonL);
                    for (int gi = 0; gi < nGrains; ++gi) {
                        float gx = ((float)cx + crng.nextFloat()) * cell;
                        float gy = ((float)cy + crng.nextFloat()) * cell;
                        if (sigmaLN > 0.0f) { crng.nextFloat(); crng.nextFloat(); } // keep stream aligned
                        float dx = spx - gx, dy = spy - gy;
                        float d2 = dx * dx + dy * dy;
                        if (d2 < R2) { float t = 1.0f - d2 / R2; F += t * t; }
                    }
                }
            }
            accum += (F - Efield);
        }
    }
    accum *= inv * inv;             // average over the g*g sub-samples
    return accum * invSigmaF;       // zero-mean, ~unit variance
}

} // namespace grain

#endif // FILMGRAIN_GRAINMODEL_H
