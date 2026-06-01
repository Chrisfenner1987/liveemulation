// FilmGrainCuda.cu - CUDA GPU render path (Windows / Linux NVIDIA).
// Mirrors the Metal kernel; shares the device math with the CPU model via
// GrainCudaCore.cuh (verified bit-identical to GrainModel.h on the host).
// Copyright (c) 2026 Chris Fenner. SPDX-License-Identifier: BSD-3-Clause

#include <cuda_runtime.h>
#include "GrainCudaCore.cuh"
#include "FilmGrainCuda.h"

// ---- Halation pre-passes ---------------------------------------------------
__global__ void HalationExtractCuda(int W, int H, float thr, float hiGain,
                                    const float* in, float* mask) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = (y * W + x) * 4;
    float luma = in[idx] * 0.2126f + in[idx + 1] * 0.7152f + in[idx + 2] * 0.0722f;
    mask[y * W + x] = fg_ss(thr, 1.0f, luma * hiGain);
}

// Separable Gaussian blur (weights computed inline + normalized).
__global__ void HalationBlurCuda(int W, int H, int R, float invTwoSigma2, int horiz,
                                 const float* in, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float acc = 0.0f, wsum = 0.0f;
    for (int i = -R; i <= R; ++i) {
        int sx = x, sy = y;
        if (horiz) { sx = x + i; sx = sx < 0 ? 0 : (sx >= W ? W - 1 : sx); }
        else       { sy = y + i; sy = sy < 0 ? 0 : (sy >= H ? H - 1 : sy); }
        float w = expf(-(float)(i * i) * invTwoSigma2);
        acc += w * in[sy * W + sx];
        wsum += w;
    }
    out[y * W + x] = acc / wsum;
}

// ---- Main grain kernel (composites halation, then grain) -------------------
__global__ void FilmGrainKernelCuda(FGCudaParams P, int lutSize, const float* halo,
                                    const float* input, float* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.width || y >= P.height) return;
    int index = (y * P.width + x) * 4;
    int px = P.offX + x, py = P.offY + y;

    float r0 = input[index], g0 = input[index + 1], b0 = input[index + 2], a0 = input[index + 3];

    if (P.halIntensity > 0.0f) {
        float h = halo[y * P.width + x] * P.halIntensity;
        r0 += P.tintR * h * (1.0f - fg_clamp(r0, 0.0f, 1.0f));
        g0 += P.tintG * h * (1.0f - fg_clamp(g0, 0.0f, 1.0f));
        b0 += P.tintB * h * (1.0f - fg_clamp(b0, 0.0f, 1.0f));
    }

    float radius[3] = { P.r0, P.r1, P.r2 };
    float sig[3]    = { P.s0, P.s1, P.s2 };
    float amt[3]    = { P.a0, P.a1, P.a2 };

    // Grain Motion: blend a static-seed pattern and the moving-seed pattern.
    unsigned int seeds[2]; float wts[2]; int nSeeds;
    if (P.motion <= 0.0f)      { nSeeds = 1; seeds[0] = P.seedStatic; wts[0] = 1.0f; }
    else if (P.motion >= 1.0f) { nSeeds = 1; seeds[0] = P.seedMoving; wts[0] = 1.0f; }
    else { nSeeds = 2; seeds[0] = P.seedStatic; wts[0] = sqrtf(1.0f - P.motion * P.motion);
                       seeds[1] = P.seedMoving; wts[1] = P.motion; }

    if (P.mono) {
        float luma = r0 * 0.2126f + g0 * 0.7152f + b0 * 0.0722f;
        float cl = fg_clamp(luma, 0.0f, 1.0f);
        float gg = 0.0f;
        for (int s = 0; s < nSeeds; ++s)
            gg += wts[s] * grainFluctDev(cl, px, py, 0, radius[0], sig[0], P.nSamples, seeds[s], P.structure);
        float eg = effectiveGainDev(lutSize, cl, P.matchCurve, P.flatScale, P.trimS, P.trimM, P.trimH);
        float delta = gg * eg * amt[0];
        output[index] = r0 + delta; output[index + 1] = g0 + delta; output[index + 2] = b0 + delta;
        output[index + 3] = a0;
    } else {
        float in3[3] = { r0, g0, b0 };
        float wc = sqrtf(P.coupling), wi = sqrtf(1.0f - P.coupling);
        float luma = r0 * 0.2126f + g0 * 0.7152f + b0 * 0.0722f;
        float clL = fg_clamp(luma, 0.0f, 1.0f);
        float nf[3] = { 0.0f, 0.0f, 0.0f };
        for (int s = 0; s < nSeeds; ++s) {
            float common = (P.coupling > 0.0f)
                ? grainFluctDev(clL, px, py, 6, radius[1], sig[1], P.nSamples, seeds[s], P.structure) : 0.0f;
            for (int c = 0; c < 3; ++c) {
                float cl = fg_clamp(in3[c], 0.0f, 1.0f);
                float indep = (P.coupling < 1.0f)
                    ? grainFluctDev(cl, px, py, c, radius[c], sig[c], P.nSamples, seeds[s], P.structure) : 0.0f;
                nf[c] += wts[s] * (wc * common + wi * indep);
            }
        }
        for (int c = 0; c < 3; ++c) {
            float cl = fg_clamp(in3[c], 0.0f, 1.0f);
            float eg = effectiveGainDev(lutSize, cl, P.matchCurve, P.flatScale, P.trimS, P.trimM, P.trimH);
            output[index + c] = in3[c] + nf[c] * eg * amt[c];
        }
        output[index + 3] = a0;
    }
}

// ---- Host launcher ---------------------------------------------------------
void RunFilmGrainCuda(void* p_Stream, FGCudaParams P, int p_LUTSize,
                      const float* p_LUT, const float* p_Input, float* p_Output) {
    cudaStream_t stream = static_cast<cudaStream_t>(p_Stream);

    // Upload the (small, constant) gain LUT to __constant__ memory.
    cudaMemcpyToSymbolAsync(c_gainLUT, p_LUT, p_LUTSize * sizeof(float), 0,
                            cudaMemcpyHostToDevice, stream);

    dim3 threads(16, 16, 1);
    dim3 blocks((P.width + threads.x - 1) / threads.x, (P.height + threads.y - 1) / threads.y, 1);

    const float* halo = p_Input;   // dummy pointer when halation is off
    float* mask = nullptr;
    float* tmp  = nullptr;
    bool doHal = (P.halIntensity > 0.0f && P.halSigma > 0.25f);
    if (doHal) {
        int R = (int)ceilf(3.0f * P.halSigma); if (R < 1) R = 1; if (R > 256) R = 256;
        float invTwoSigma2 = 1.0f / (2.0f * P.halSigma * P.halSigma);
        size_t bytes = (size_t)P.width * P.height * sizeof(float);
        cudaMallocAsync((void**)&mask, bytes, stream);
        cudaMallocAsync((void**)&tmp,  bytes, stream);

        HalationExtractCuda<<<blocks, threads, 0, stream>>>(P.width, P.height, P.halThreshold, P.halHiGain, p_Input, mask);
        HalationBlurCuda<<<blocks, threads, 0, stream>>>(P.width, P.height, R, invTwoSigma2, 1, mask, tmp);
        HalationBlurCuda<<<blocks, threads, 0, stream>>>(P.width, P.height, R, invTwoSigma2, 0, tmp, mask);
        halo = mask;
    } else {
        P.halIntensity = 0.0f;
    }

    FilmGrainKernelCuda<<<blocks, threads, 0, stream>>>(P, p_LUTSize, halo, p_Input, p_Output);

    if (mask) cudaFreeAsync(mask, stream);
    if (tmp)  cudaFreeAsync(tmp, stream);
}
