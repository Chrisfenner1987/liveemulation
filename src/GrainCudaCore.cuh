// GrainCudaCore.cuh - device-side grain + halation math for the CUDA path.
// Mirrors src/GrainModel.h (CPU) and the Metal kernel 1:1. Compiles under nvcc
// (__device__) and also as plain host C++ (macros expand to nothing) so the math
// can be unit-tested against GrainModel.h. KEEP IN SYNC with those two.
#ifndef FILMGRAIN_GRAINCUDACORE_CUH
#define FILMGRAIN_GRAINCUDACORE_CUH

#include <cmath>

#ifdef __CUDACC__
  #define FG_DEV __device__
  #define FG_CONST __constant__
#else
  #define FG_DEV
  #define FG_CONST
  #include <math.h>   // host build: expf/logf/sqrtf/floorf in the global namespace
#endif

static const float kPI = 3.14159265358979323846f;
static const float kKernelSupport = 2.2f;   // R = kKernelSupport * grainRadius
static const float kCoarseScale   = 2.5f;   // coarse octave radius multiplier

FG_CONST float c_gainLUT[64];                // gain LUT (kGainLUTSize entries used)

FG_DEV inline float fg_clamp(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }
FG_DEV inline float fg_ss(float a, float b, float x) { float t = (x - a) / (b - a); t = fg_clamp(t, 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); }

FG_DEV inline unsigned int pcgHash(unsigned int v) {
    unsigned int s = v * 747796405u + 2891336453u;
    unsigned int w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
FG_DEV inline unsigned int hash2(unsigned int a, unsigned int b) { return pcgHash(a ^ (b * 0x9e3779b9u)); }
FG_DEV inline unsigned int hash3(unsigned int a, unsigned int b, unsigned int c) { return pcgHash(hash2(a, b) ^ (c * 0x85ebca6bu)); }
FG_DEV inline unsigned int rngNext(unsigned int& s) {
    s = s * 747796405u + 2891336453u;
    unsigned int w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
FG_DEV inline float rngFloat(unsigned int& s) { return (rngNext(s) >> 8) * (1.0f / 16777216.0f); }
FG_DEV inline int samplePoissonL(unsigned int& s, float L) {
    int k = 0; float p = 1.0f;
    do { ++k; p *= rngFloat(s); } while (p > L);
    return k - 1;
}
FG_DEV inline void lognormalParams(float mean, float stddev, float& muLN, float& sigmaLN) {
    if (stddev <= 1e-6f || mean <= 1e-6f) { muLN = logf(mean > 1e-6f ? mean : 1e-6f); sigmaLN = 0.0f; return; }
    float v = stddev * stddev, m2 = mean * mean;
    sigmaLN = sqrtf(logf(1.0f + v / m2));
    muLN = logf(m2 / sqrtf(m2 + v));
}

// Soft-kernel shot-noise fluctuation (zero-mean, ~unit variance). Identical to
// grain::renderPixel in GrainModel.h.
FG_DEV inline float renderPixelDev(float value, int px, int py, int chan,
                                   float grainRadius, float sigmaRr, int nSamples, unsigned int seed) {
    float u = value;
    if (u < 0.0f) u = 0.0f;
    if (u > 0.999f) u = 0.999f;
    if (u <= 0.0f) return 0.0f;

    float r = grainRadius > 1e-4f ? grainRadius : 1e-4f;
    float muLN, sigmaLN; lognormalParams(r, sigmaRr, muLN, sigmaLN);
    float expR2 = expf(2.0f * muLN + 2.0f * sigmaLN * sigmaLN);
    float meanGrainArea = kPI * expR2;
    float lambda = -logf(1.0f - u) / meanGrainArea;

    float R = kKernelSupport * r, R2 = R * R;
    float Efield = lambda * (kPI * R2 / 3.0f);
    float varF   = lambda * (kPI * R2 / 5.0f);
    if (varF <= 1e-12f) return 0.0f;
    float invSigmaF = 1.0f / sqrtf(varF);

    float cell = R > 1e-4f ? R : 1e-4f;
    float meanPerCell = lambda * cell * cell;
    if (meanPerCell > 30.0f) meanPerCell = 30.0f;
    float poissonL = expf(-meanPerCell);

    unsigned int seedC = seed + (unsigned int)chan * 0x6d2b79f5u;
    unsigned int pbase = (unsigned int)(px * 1973 + 9277) ^ ((unsigned int)(py * 26699 + 7919) * 0x9e3779b9u) ^ seedC;

    int N = nSamples > 0 ? nSamples : 1;
    if (N > 16) N = 16;
    int g = (int)floorf(sqrtf((float)N) + 0.5f); if (g < 1) g = 1;
    float inv = 1.0f / (float)g;

    float accum = 0.0f;
    for (int sj = 0; sj < g; ++sj) {
        for (int si = 0; si < g; ++si) {
            unsigned int hs = pcgHash(pbase + (unsigned int)(sj * g + si) * 0x27d4eb2fu);
            float jx = (hs & 0xffffu) * (1.0f / 65536.0f);
            float jy = ((hs >> 16) & 0xffffu) * (1.0f / 65536.0f);
            float spx = (float)px + ((float)si + jx) * inv;
            float spy = (float)py + ((float)sj + jy) * inv;

            int cxMin = (int)floorf((spx - R) / cell), cxMax = (int)floorf((spx + R) / cell);
            int cyMin = (int)floorf((spy - R) / cell), cyMax = (int)floorf((spy + R) / cell);
            float F = 0.0f;
            for (int cy = cyMin; cy <= cyMax; ++cy) {
                for (int cx = cxMin; cx <= cxMax; ++cx) {
                    unsigned int cs = hash3((unsigned int)(cx * 0x1f1f1f1f + 0x2c1b3c6d),
                                            (unsigned int)(cy * 0x2545f491 + 0x1b873593), seedC);
                    int nG = samplePoissonL(cs, poissonL);
                    for (int gi = 0; gi < nG; ++gi) {
                        float gx = ((float)cx + rngFloat(cs)) * cell;
                        float gy = ((float)cy + rngFloat(cs)) * cell;
                        if (sigmaLN > 0.0f) { rngFloat(cs); rngFloat(cs); }
                        float dx = spx - gx, dy = spy - gy, d2 = dx * dx + dy * dy;
                        if (d2 < R2) { float t = 1.0f - d2 / R2; F += t * t; }
                    }
                }
            }
            accum += (F - Efield);
        }
    }
    accum *= inv * inv;
    return accum * invSigmaF;
}

FG_DEV inline float grainFluctDev(float cl, int px, int py, int chan,
                                  float radius, float sigmaR, int nSamples, unsigned int seed, float structure) {
    float fine = renderPixelDev(cl, px, py, chan, radius, sigmaR, nSamples, seed);
    if (structure <= 0.0f) return fine;
    float coarse = renderPixelDev(cl, px, py, chan + 3, radius * kCoarseScale, sigmaR, nSamples, seed);
    float m = structure > 1.0f ? 1.0f : structure;
    return sqrtf(1.0f - m * m) * fine + m * coarse;
}

FG_DEV inline float sampleGainLUTDev(int n, float u) {
    if (u <= 0.0f) return c_gainLUT[0];
    if (u >= 1.0f) return c_gainLUT[n - 1];
    float f = u * (float)(n - 1);
    int i = (int)f; float t = f - (float)i;
    return c_gainLUT[i] * (1.0f - t) + c_gainLUT[i + 1] * t;
}
FG_DEV inline float trimAtDev(float u, float s, float m, float h) {
    if (u <= 0.5f) { float t = u / 0.5f; return s * (1.0f - t) + m * t; }
    float t = (u - 0.5f) / 0.5f; return m * (1.0f - t) + h * t;
}
FG_DEV inline float tonalTaperDev(float u) { return fg_ss(0.0f, 0.20f, u) * fg_ss(0.0f, 0.12f, 1.0f - u); }
FG_DEV inline float effectiveGainDev(int n, float u, int matchCurve, float flat, float s, float m, float h) {
    float g = matchCurve ? sampleGainLUTDev(n, u) : flat;
    return g * trimAtDev(u, s, m, h) * tonalTaperDev(u);
}

#endif // FILMGRAIN_GRAINCUDACORE_CUH
