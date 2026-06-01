// FilmGrain OFX - Apple Metal GPU render path.
// Copyright (c) 2026 Chris Fenner. SPDX-License-Identifier: BSD-3-Clause
//
// KEEP IN SYNC with src/GrainModel.h. The kernel below re-implements the same
// stochastic Boolean grain model so the GPU preview matches the CPU render.
// (Results are visually identical, not bit-exact, due to GPU fast-math.)

#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>
#include <cstdio>

static const char* kKernelSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant float kPI = 3.14159265358979323846f;

inline uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
inline uint hash2(uint a, uint b) { return pcgHash(a ^ (b * 0x9e3779b9u)); }
inline uint hash3(uint a, uint b, uint c) { return pcgHash(hash2(a, b) ^ (c * 0x85ebca6bu)); }

inline uint rngNext(thread uint& s) {
    s = s * 747796405u + 2891336453u;
    uint word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (word >> 22u) ^ word;
}
inline float rngFloat(thread uint& s) { return (rngNext(s) >> 8) * (1.0f / 16777216.0f); }

inline int samplePoisson(thread uint& s, float mean) {
    if (mean <= 0.0f) return 0;
    if (mean > 30.0f) mean = 30.0f;
    float L = exp(-mean);
    int k = 0;
    float p = 1.0f;
    do { ++k; p *= rngFloat(s); } while (p > L);
    return k - 1;
}
// Variant taking a precomputed L = exp(-mean) (mean is constant per pixel).
inline int samplePoissonL(thread uint& s, float L) {
    int k = 0;
    float p = 1.0f;
    do { ++k; p *= rngFloat(s); } while (p > L);
    return k - 1;
}

inline void lognormalParams(float mean, float stddev, thread float& muLN, thread float& sigmaLN) {
    if (stddev <= 1e-6f || mean <= 1e-6f) { muLN = log(mean > 1e-6f ? mean : 1e-6f); sigmaLN = 0.0f; return; }
    float v = stddev * stddev;
    float m2 = mean * mean;
    sigmaLN = sqrt(log(1.0f + v / m2));
    muLN = log(m2 / sqrt(m2 + v));
}

// Soft compact grain kernel K(d) = (1 - (d/R)^2)^2 (support R). Shot-noise field
// F = sum_g K; returns the ZERO-MEAN, ~unit-variance fluctuation F-E[F] over
// sigma_F. Bipolar + soft -> dimensional grain, not bright dots on black.
// (KEEP IN SYNC with src/GrainModel.h.)
constant float kKernelSupport = 2.2f;

inline float renderPixel(float value, int px, int py, int chan, float grainRadius, float sigmaRr, int nSamples, uint seed) {
    float u = value;
    if (u < 0.0f) u = 0.0f;
    if (u > 0.999f) u = 0.999f;
    if (u <= 0.0f) return 0.0f;

    float r = grainRadius > 1e-4f ? grainRadius : 1e-4f;

    float muLN, sigmaLN;
    lognormalParams(r, sigmaRr, muLN, sigmaLN);
    float expR2 = exp(2.0f * muLN + 2.0f * sigmaLN * sigmaLN);
    float meanGrainArea = kPI * expR2;
    float lambda = -log(1.0f - u) / meanGrainArea;

    float R = kKernelSupport * r;
    float R2 = R * R;
    float Efield = lambda * (kPI * R2 / 3.0f);
    float varF   = lambda * (kPI * R2 / 5.0f);
    if (varF <= 1e-12f) return 0.0f;
    float invSigmaF = 1.0f / sqrt(varF);

    float cell = R > 1e-4f ? R : 1e-4f;
    float meanPerCell = lambda * cell * cell;
    if (meanPerCell > 30.0f) meanPerCell = 30.0f;
    float poissonL = exp(-meanPerCell);

    uint seedC = seed + (uint)chan * 0x6d2b79f5u;
    uint pbase = (uint)(px * 1973 + 9277) ^ ((uint)(py * 26699 + 7919) * 0x9e3779b9u) ^ seedC;

    int N = nSamples > 0 ? nSamples : 1;
    if (N > 16) N = 16;
    int g = (int)floor(sqrt((float)N) + 0.5f); if (g < 1) g = 1;
    float inv = 1.0f / (float)g;

    float accum = 0.0f;
    for (int sj = 0; sj < g; ++sj) {
        for (int si = 0; si < g; ++si) {
            uint hs = pcgHash(pbase + (uint)(sj * g + si) * 0x27d4eb2fu);
            float jx = (hs & 0xffffu) * (1.0f / 65536.0f);
            float jy = ((hs >> 16) & 0xffffu) * (1.0f / 65536.0f);
            float spx = (float)px + ((float)si + jx) * inv;
            float spy = (float)py + ((float)sj + jy) * inv;

            int cxMin = (int)floor((spx - R) / cell);
            int cxMax = (int)floor((spx + R) / cell);
            int cyMin = (int)floor((spy - R) / cell);
            int cyMax = (int)floor((spy + R) / cell);

            float F = 0.0f;
            for (int cy = cyMin; cy <= cyMax; ++cy) {
                for (int cx = cxMin; cx <= cxMax; ++cx) {
                    uint cs = hash3((uint)(cx * 0x1f1f1f1f + 0x2c1b3c6d),
                                    (uint)(cy * 0x2545f491 + 0x1b873593),
                                    seedC);
                    int nGrains = samplePoissonL(cs, poissonL);
                    for (int gi = 0; gi < nGrains; ++gi) {
                        float gx = ((float)cx + rngFloat(cs)) * cell;
                        float gy = ((float)cy + rngFloat(cs)) * cell;
                        if (sigmaLN > 0.0f) { rngFloat(cs); rngFloat(cs); } // keep stream aligned
                        float dx = spx - gx, dy = spy - gy;
                        float d2 = dx * dx + dy * dy;
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

// Two-octave grain: fine + independent coarse layer, unit-variance mix.
constant float kCoarseScale = 2.5f;
inline float grainFluct(float cl, int px, int py, int chan, float radius, float sigmaR, int nSamples, uint seed, float structure) {
    float fine = renderPixel(cl, px, py, chan, radius, sigmaR, nSamples, seed);
    if (structure <= 0.0f) return fine;
    float coarse = renderPixel(cl, px, py, chan + 3, radius * kCoarseScale, sigmaR, nSamples, seed);
    float m = structure > 1.0f ? 1.0f : structure;
    return sqrt(1.0f - m * m) * fine + m * coarse;
}

inline float sampleGainLUT(constant float* lut, int n, float u) {
    if (u <= 0.0f) return lut[0];
    if (u >= 1.0f) return lut[n - 1];
    float f = u * (float)(n - 1);
    int i = (int)f;
    float t = f - (float)i;
    return lut[i] * (1.0f - t) + lut[i + 1] * t;
}
inline float trimAt(float u, float s, float m, float h) {
    if (u <= 0.5f) { float t = u / 0.5f; return s * (1.0f - t) + m * t; }
    float t = (u - 0.5f) / 0.5f; return m * (1.0f - t) + h * t;
}
// Fade grain toward black/white so one-sided clipping can't make flat dots.
inline float tonalTaper(float u) {
    return smoothstep(0.0f, 0.20f, u) * smoothstep(0.0f, 0.12f, 1.0f - u);
}
inline float effectiveGain(constant float* lut, int n, float u, int matchCurve, float flat, float s, float m, float h) {
    float g = matchCurve != 0 ? sampleGainLUT(lut, n, u) : flat;
    return g * trimAt(u, s, m, h) * tonalTaper(u);
}

// --- Halation passes -------------------------------------------------------
// Extract a highlight mask (luma above threshold) for the halation glow source.
kernel void HalationExtract(constant int& p_Width [[buffer(11)]],
                            constant int& p_Height [[buffer(12)]],
                            constant float& p_Threshold [[buffer(13)]],
                            constant float& p_HiGain [[buffer(14)]],
                            const device float* p_Input [[buffer(0)]],
                            device float* p_Mask [[buffer(1)]],
                            uint2 id [[thread_position_in_grid]]) {
    if (id.x >= (uint)p_Width || id.y >= (uint)p_Height) return;
    int idx = ((int)id.y * p_Width + (int)id.x) * 4;
    float luma = p_Input[idx] * 0.2126f + p_Input[idx + 1] * 0.7152f + p_Input[idx + 2] * 0.0722f;
    // Highlight Gain scales luma into the threshold range so halation finds the
    // highlights regardless of where the node sits (log/flat vs display-referred).
    p_Mask[(int)id.y * p_Width + (int)id.x] = smoothstep(p_Threshold, 1.0f, luma * p_HiGain);
}

// Separable Gaussian blur of the mask. p_Horiz != 0 -> horizontal pass.
kernel void HalationBlur(constant int& p_Width [[buffer(11)]],
                         constant int& p_Height [[buffer(12)]],
                         constant int& p_Radius [[buffer(13)]],
                         constant int& p_Horiz [[buffer(14)]],
                         constant float* p_Weights [[buffer(2)]],
                         const device float* p_In [[buffer(1)]],
                         device float* p_Out [[buffer(3)]],
                         uint2 id [[thread_position_in_grid]]) {
    if (id.x >= (uint)p_Width || id.y >= (uint)p_Height) return;
    int x = (int)id.x, y = (int)id.y;
    float acc = 0.0f;
    for (int i = -p_Radius; i <= p_Radius; ++i) {
        int sx = x, sy = y;
        if (p_Horiz != 0) { sx = x + i; if (sx < 0) sx = 0; else if (sx >= p_Width) sx = p_Width - 1; }
        else             { sy = y + i; if (sy < 0) sy = 0; else if (sy >= p_Height) sy = p_Height - 1; }
        acc += p_Weights[i + p_Radius] * p_In[sy * p_Width + sx];
    }
    p_Out[y * p_Width + x] = acc;
}

kernel void FilmGrainKernel(constant int& p_Width   [[buffer(11)]],
                            constant int& p_Height  [[buffer(12)]],
                            constant int& p_OffX    [[buffer(13)]],
                            constant int& p_OffY    [[buffer(14)]],
                            constant float* p_Radius [[buffer(15)]],
                            constant float* p_SigmaR [[buffer(16)]],
                            constant float* p_Amount [[buffer(17)]],
                            constant int& p_NSamples [[buffer(18)]],
                            constant uint& p_SeedStatic [[buffer(19)]],
                            constant int& p_Mono     [[buffer(20)]],
                            constant int& p_MatchCurve [[buffer(21)]],
                            constant float* p_GainLUT  [[buffer(22)]],
                            constant int& p_LUTSize    [[buffer(23)]],
                            constant float& p_FlatScale [[buffer(24)]],
                            constant float* p_Trim     [[buffer(25)]],
                            constant float& p_Structure [[buffer(26)]],
                            constant float& p_Coupling [[buffer(27)]],
                            const device float* p_Halo [[buffer(28)]],
                            constant float& p_HalIntensity [[buffer(29)]],
                            constant float* p_HalTint [[buffer(30)]],
                            constant uint& p_SeedMoving [[buffer(9)]],
                            constant float& p_Motion [[buffer(10)]],
                            const device float* p_Input [[buffer(0)]],
                            device float* p_Output      [[buffer(8)]],
                            uint2 id [[thread_position_in_grid]]) {
    if (id.x >= (uint)p_Width || id.y >= (uint)p_Height) return;
    int index = ((int)id.y * p_Width + (int)id.x) * 4;
    int px = p_OffX + (int)id.x;
    int py = p_OffY + (int)id.y;

    float r0 = p_Input[index + 0];
    float g0 = p_Input[index + 1];
    float b0 = p_Input[index + 2];
    float a0 = p_Input[index + 3];

    // Halation: add the tinted, blurred highlight glow (screen-like, HDR-safe)
    // BEFORE grain, since it is an exposure-stage effect.
    if (p_HalIntensity > 0.0f) {
        float h = p_Halo[(int)id.y * p_Width + (int)id.x] * p_HalIntensity;
        r0 += p_HalTint[0] * h * (1.0f - clamp(r0, 0.0f, 1.0f));
        g0 += p_HalTint[1] * h * (1.0f - clamp(g0, 0.0f, 1.0f));
        b0 += p_HalTint[2] * h * (1.0f - clamp(b0, 0.0f, 1.0f));
    }

    float tS = p_Trim[0], tM = p_Trim[1], tH = p_Trim[2];

    // Grain Motion: blend a static-seed pattern and the moving-seed pattern.
    uint seeds[2]; float wts[2]; int nSeeds;
    if (p_Motion <= 0.0f)      { nSeeds = 1; seeds[0] = p_SeedStatic; wts[0] = 1.0f; }
    else if (p_Motion >= 1.0f) { nSeeds = 1; seeds[0] = p_SeedMoving; wts[0] = 1.0f; }
    else { nSeeds = 2; seeds[0] = p_SeedStatic; wts[0] = sqrt(1.0f - p_Motion * p_Motion);
                       seeds[1] = p_SeedMoving; wts[1] = p_Motion; }

    if (p_Mono != 0) {
        float luma = r0 * 0.2126f + g0 * 0.7152f + b0 * 0.0722f;
        float cl = clamp(luma, 0.0f, 1.0f);
        float gg = 0.0f;
        for (int s = 0; s < nSeeds; ++s)
            gg += wts[s] * grainFluct(cl, px, py, 0, p_Radius[0], p_SigmaR[0], p_NSamples, seeds[s], p_Structure);
        float eg = effectiveGain(p_GainLUT, p_LUTSize, cl, p_MatchCurve, p_FlatScale, tS, tM, tH);
        float delta = gg * eg * p_Amount[0];
        p_Output[index + 0] = r0 + delta;
        p_Output[index + 1] = g0 + delta;
        p_Output[index + 2] = b0 + delta;
        p_Output[index + 3] = a0;
    } else {
        float in[3] = { r0, g0, b0 };
        float wc = sqrt(p_Coupling), wi = sqrt(1.0f - p_Coupling);
        float luma = r0 * 0.2126f + g0 * 0.7152f + b0 * 0.0722f;
        float clL = clamp(luma, 0.0f, 1.0f);
        float nf[3] = { 0.0f, 0.0f, 0.0f };
        for (int s = 0; s < nSeeds; ++s) {
            float common = (p_Coupling > 0.0f)
                ? grainFluct(clL, px, py, 6, p_Radius[1], p_SigmaR[1], p_NSamples, seeds[s], p_Structure) : 0.0f;
            for (int c = 0; c < 3; ++c) {
                float cl = clamp(in[c], 0.0f, 1.0f);
                float indep = (p_Coupling < 1.0f)
                    ? grainFluct(cl, px, py, c, p_Radius[c], p_SigmaR[c], p_NSamples, seeds[s], p_Structure) : 0.0f;
                nf[c] += wts[s] * (wc * common + wi * indep);
            }
        }
        for (int c = 0; c < 3; ++c) {
            float cl = clamp(in[c], 0.0f, 1.0f);
            float eg = effectiveGain(p_GainLUT, p_LUTSize, cl, p_MatchCurve, p_FlatScale, tS, tM, tH);
            p_Output[index + c] = in[c] + nf[c] * eg * p_Amount[c];
        }
        p_Output[index + 3] = a0;
    }
}
)METAL";

struct FGPipelines {
    id<MTLComputePipelineState> grain;
    id<MTLComputePipelineState> extract;
    id<MTLComputePipelineState> blur;
};
std::mutex s_PipelineQueueMutex;
typedef std::unordered_map<id<MTLCommandQueue>, FGPipelines> PipelineQueueMap;
PipelineQueueMap s_PipelineQueueMap;

static id<MTLComputePipelineState> buildPipe(id<MTLLibrary> lib, id<MTLDevice> dev, const char* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) { fprintf(stderr, "FilmGrain: kernel %s not found\n", name); return nil; }
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!ps) fprintf(stderr, "FilmGrain: pipeline %s error: %s\n", name, err.localizedDescription.UTF8String);
    [fn release];
    return ps;
}

void RunFilmGrainMetal(void* p_CmdQ, int p_Width, int p_Height, int p_OffX, int p_OffY,
                       const float* p_Radius, const float* p_SigmaR, const float* p_Amount,
                       int p_NSamples, unsigned int p_SeedStatic, unsigned int p_SeedMoving, float p_Motion, int p_Mono,
                       int p_MatchCurve, const float* p_GainLUT, int p_LUTSize, float p_FlatScale,
                       const float* p_Trim, float p_Structure, float p_Coupling,
                       float p_HalIntensity, const float* p_HalTint, float p_HalSigma, float p_HalThreshold,
                       float p_HalHiGain, const float* p_Input, float* p_Output) {
    id<MTLCommandQueue> queue = static_cast<id<MTLCommandQueue>>(p_CmdQ);
    id<MTLDevice> device = queue.device;
    FGPipelines pipes;

    {
        std::unique_lock<std::mutex> lock(s_PipelineQueueMutex);
        const auto it = s_PipelineQueueMap.find(queue);
        if (it == s_PipelineQueueMap.end()) {
            NSError* err = nil;
            MTLCompileOptions* options = [MTLCompileOptions new];
            id<MTLLibrary> lib = [device newLibraryWithSource:@(kKernelSource) options:options error:&err];
            [options release];
            if (!lib) { fprintf(stderr, "FilmGrain: Metal compile failed: %s\n", err.localizedDescription.UTF8String); return; }
            pipes.grain   = buildPipe(lib, device, "FilmGrainKernel");
            pipes.extract = buildPipe(lib, device, "HalationExtract");
            pipes.blur    = buildPipe(lib, device, "HalationBlur");
            [lib release];
            if (!pipes.grain || !pipes.extract || !pipes.blur) return;
            s_PipelineQueueMap[queue] = pipes;
        } else {
            pipes = it->second;
        }
    }

    id<MTLBuffer> srcBuf = reinterpret_cast<id<MTLBuffer>>(const_cast<float*>(p_Input));
    id<MTLBuffer> dstBuf = reinterpret_cast<id<MTLBuffer>>(p_Output);

    int exeWidth = [pipes.grain threadExecutionWidth];
    MTLSize tgc = MTLSizeMake(exeWidth, 1, 1);
    MTLSize tg = MTLSizeMake((p_Width + exeWidth - 1) / exeWidth, p_Height, 1);

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    commandBuffer.label = @"FilmGrain";

    // ---- Halation pre-passes (extract highlights, separable Gaussian blur) ----
    id<MTLBuffer> haloBuf = srcBuf;   // dummy binding when halation is off
    id<MTLBuffer> maskBuf = nil, tmpBuf = nil;
    bool doHal = (p_HalIntensity > 0.0f && p_HalSigma > 0.25f);
    if (doHal) {
        int R = (int)ceil(3.0f * p_HalSigma); if (R < 1) R = 1; if (R > 256) R = 256;
        int nW = 2 * R + 1;
        float wsum = 0.0f; float weights[513];
        float inv2s2 = 1.0f / (2.0f * p_HalSigma * p_HalSigma);
        for (int i = -R; i <= R; ++i) { float w = expf(-(float)(i * i) * inv2s2); weights[i + R] = w; wsum += w; }
        for (int i = 0; i < nW; ++i) weights[i] /= wsum;

        size_t bytes = (size_t)p_Width * p_Height * sizeof(float);
        maskBuf = [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        tmpBuf  = [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];

        // extract: src -> mask
        id<MTLComputeCommandEncoder> e1 = [commandBuffer computeCommandEncoder];
        [e1 setComputePipelineState:pipes.extract];
        [e1 setBuffer:srcBuf offset:0 atIndex:0];
        [e1 setBuffer:maskBuf offset:0 atIndex:1];
        [e1 setBytes:&p_Width length:sizeof(int) atIndex:11];
        [e1 setBytes:&p_Height length:sizeof(int) atIndex:12];
        [e1 setBytes:&p_HalThreshold length:sizeof(float) atIndex:13];
        [e1 setBytes:&p_HalHiGain length:sizeof(float) atIndex:14];
        [e1 dispatchThreadgroups:tg threadsPerThreadgroup:tgc];
        [e1 endEncoding];

        // blur H: mask -> tmp
        int horiz = 1, vert = 0;
        id<MTLComputeCommandEncoder> e2 = [commandBuffer computeCommandEncoder];
        [e2 setComputePipelineState:pipes.blur];
        [e2 setBuffer:maskBuf offset:0 atIndex:1];
        [e2 setBuffer:tmpBuf offset:0 atIndex:3];
        [e2 setBytes:weights length:sizeof(float) * nW atIndex:2];
        [e2 setBytes:&p_Width length:sizeof(int) atIndex:11];
        [e2 setBytes:&p_Height length:sizeof(int) atIndex:12];
        [e2 setBytes:&R length:sizeof(int) atIndex:13];
        [e2 setBytes:&horiz length:sizeof(int) atIndex:14];
        [e2 dispatchThreadgroups:tg threadsPerThreadgroup:tgc];
        [e2 endEncoding];

        // blur V: tmp -> mask (final halo)
        id<MTLComputeCommandEncoder> e3 = [commandBuffer computeCommandEncoder];
        [e3 setComputePipelineState:pipes.blur];
        [e3 setBuffer:tmpBuf offset:0 atIndex:1];
        [e3 setBuffer:maskBuf offset:0 atIndex:3];
        [e3 setBytes:weights length:sizeof(float) * nW atIndex:2];
        [e3 setBytes:&p_Width length:sizeof(int) atIndex:11];
        [e3 setBytes:&p_Height length:sizeof(int) atIndex:12];
        [e3 setBytes:&R length:sizeof(int) atIndex:13];
        [e3 setBytes:&vert length:sizeof(int) atIndex:14];
        [e3 dispatchThreadgroups:tg threadsPerThreadgroup:tgc];
        [e3 endEncoding];

        haloBuf = maskBuf;
    }

    // ---- Main grain kernel (composites halation, then grain) ----
    id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
    [enc setComputePipelineState:pipes.grain];
    [enc setBuffer:srcBuf offset:0 atIndex:0];
    [enc setBuffer:dstBuf offset:0 atIndex:8];
    [enc setBytes:&p_Width    length:sizeof(int) atIndex:11];
    [enc setBytes:&p_Height   length:sizeof(int) atIndex:12];
    [enc setBytes:&p_OffX     length:sizeof(int) atIndex:13];
    [enc setBytes:&p_OffY     length:sizeof(int) atIndex:14];
    [enc setBytes:p_Radius    length:sizeof(float) * 3 atIndex:15];
    [enc setBytes:p_SigmaR    length:sizeof(float) * 3 atIndex:16];
    [enc setBytes:p_Amount    length:sizeof(float) * 3 atIndex:17];
    [enc setBytes:&p_NSamples length:sizeof(int) atIndex:18];
    [enc setBytes:&p_SeedStatic length:sizeof(unsigned int) atIndex:19];
    [enc setBytes:&p_Mono     length:sizeof(int) atIndex:20];
    [enc setBytes:&p_MatchCurve length:sizeof(int) atIndex:21];
    [enc setBytes:p_GainLUT   length:sizeof(float) * p_LUTSize atIndex:22];
    [enc setBytes:&p_LUTSize  length:sizeof(int) atIndex:23];
    [enc setBytes:&p_FlatScale length:sizeof(float) atIndex:24];
    [enc setBytes:p_Trim      length:sizeof(float) * 3 atIndex:25];
    [enc setBytes:&p_Structure length:sizeof(float) atIndex:26];
    [enc setBytes:&p_Coupling length:sizeof(float) atIndex:27];
    [enc setBuffer:haloBuf offset:0 atIndex:28];
    float halI = doHal ? p_HalIntensity : 0.0f;
    [enc setBytes:&halI length:sizeof(float) atIndex:29];
    [enc setBytes:p_HalTint length:sizeof(float) * 3 atIndex:30];
    [enc setBytes:&p_SeedMoving length:sizeof(unsigned int) atIndex:9];
    [enc setBytes:&p_Motion length:sizeof(float) atIndex:10];
    [enc dispatchThreadgroups:tg threadsPerThreadgroup:tgc];
    [enc endEncoding];

    if (maskBuf || tmpBuf) {
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) { [maskBuf release]; [tmpBuf release]; }];
    }
    [commandBuffer commit];
}
