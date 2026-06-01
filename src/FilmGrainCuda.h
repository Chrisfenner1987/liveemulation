// FilmGrainCuda.h - shared CUDA launch interface (used by FilmGrain.cpp and
// FilmGrainCuda.cu). Plain C++ POD struct so both sides agree on layout.
#ifndef FILMGRAIN_CUDA_H
#define FILMGRAIN_CUDA_H

struct FGCudaParams {
    int width, height, offX, offY;
    float r0, r1, r2;          // per-channel grain radius
    float s0, s1, s2;          // per-channel sigmaR (irregularity)
    float a0, a1, a2;          // per-channel effective amount
    int nSamples;
    unsigned int seedStatic, seedMoving;   // frame-0 pattern, current-frame pattern
    float motion;                          // 0 = static, 1 = full movement
    int mono;
    int matchCurve;
    float flatScale;
    float trimS, trimM, trimH;
    float structure, coupling;
    float halIntensity, tintR, tintG, tintB, halThreshold, halHiGain, halSigma;
};

// Implemented in FilmGrainCuda.cu. p_Stream is the host-provided cudaStream_t;
// p_Input/p_Output are CUDA device pointers (RGBA float). lut is the host gain LUT.
extern void RunFilmGrainCuda(void* p_Stream, FGCudaParams p_Params, int p_LUTSize,
                             const float* p_LUT, const float* p_Input, float* p_Output);

#endif // FILMGRAIN_CUDA_H
