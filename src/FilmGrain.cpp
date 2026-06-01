// FilmGrain OFX - physically-based stochastic film-grain plugin.
// Copyright (c) 2026 Chris Fenner. SPDX-License-Identifier: BSD-3-Clause
//
// CPU render path + Apple Metal GPU path. The grain itself is the Boolean
// silver-grain model in GrainModel.h (CPU) / FilmGrainMetal.mm (GPU); both
// paths share identical math so previews and renders match.

#include "FilmGrain.h"
#include "GrainModel.h"
#include "GrainLUT.h"
#ifdef FILMGRAIN_CUDA
#include "FilmGrainCuda.h"
#endif

#include <cstdio>
#include <cmath>
#include <vector>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"

#define kPluginName        "Film Grain (Stochastic)"
#define kPluginGrouping    "Fenner"
#define kPluginDescription \
    "Physically-based film grain via a stochastic silver-halide (Boolean) grain model.\n" \
    "Grain is signal-dependent (strongest in the mid-tones) and resolution-independent.\n" \
    "Defaults are tuned toward Kodak Vision3 500T; calibrate further from a scan."
#define kPluginIdentifier  "com.chrisfenner.FilmGrainStochastic"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

#define kSupportsTiles              false
#define kSupportsMultiResolution    false
#define kSupportsMultipleClipPARs   false

////////////////////////////////////////////////////////////////////////////////
// Intensity response: per-level grain gain (KEEP IN SYNC with FilmGrainMetal.mm).

static inline float sampleGainLUT(float u) {
    if (u <= 0.0f) return kGainLUT[0];
    if (u >= 1.0f) return kGainLUT[kGainLUTSize - 1];
    float f = u * (kGainLUTSize - 1);
    int i = (int)f;
    float t = f - (float)i;
    return kGainLUT[i] * (1.0f - t) + kGainLUT[i + 1] * t;
}

// Smoothly blends shadow/mid/highlight trims across the tonal range.
static inline float trimAt(float u, float s, float m, float h) {
    if (u <= 0.5f) { float t = u / 0.5f; return s * (1.0f - t) + m * t; }
    float t = (u - 0.5f) / 0.5f; return m * (1.0f - t) + h * t;
}

static inline float fgSmoothstep(float a, float b, float x) {
    float t = (x - a) / (b - a);
    if (t < 0.0f) t = 0.0f; else if (t > 1.0f) t = 1.0f;
    return t * t * (3.0f - 2.0f * t);
}

// Tonal taper: fades grain toward pure black and pure white. Additive grain in
// those regions can only clip one-sided (bright specks on black / dark specks on
// white), which reads as flat dots; tapering keeps grain in the mid-tones where
// it looks dimensional. Mids (>~0.2) are unaffected, so scan-matching holds there.
static inline float tonalTaper(float u) {
    return fgSmoothstep(0.0f, 0.20f, u) * fgSmoothstep(0.0f, 0.12f, 1.0f - u);
}

static inline float effectiveGain(float u, bool matchCurve, float s, float m, float h) {
    float g = matchCurve ? sampleGainLUT(u) : kGainFlatScale;
    return g * trimAt(u, s, m, h) * tonalTaper(u);
}

static const float kCoarseScale = 2.5f;   // coarse octave radius multiplier

// Two-octave grain for dimensional clumping: a fine layer plus an independent
// coarser layer (decorrelated via channel offset). Mixed as
//   sqrt(1-m^2)*fine + m*coarse
// which keeps unit variance, so overall RMS / calibration is unchanged; only the
// spatial structure gains large-scale clumping. m=0 -> single fine layer.
static inline float grainFluct(float cl, int x, int y, int chan,
                               const grain::Params& base, float structure) {
    float fine = grain::renderPixel(cl, x, y, chan, base);
    if (structure <= 0.0f) return fine;
    grain::Params cp = base;
    cp.grainRadius = base.grainRadius * kCoarseScale;
    float coarse = grain::renderPixel(cl, x, y, chan + 3, cp);
    float m = structure > 1.0f ? 1.0f : structure;
    return std::sqrt(1.0f - m * m) * fine + m * coarse;
}

////////////////////////////////////////////////////////////////////////////////
// Processor

class FilmGrainProcessor : public OFX::ImageProcessor {
public:
    explicit FilmGrainProcessor(OFX::ImageEffect& p_Instance);

    virtual void processImagesMetal();
    virtual void processImagesCuda();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

    void setSrcImg(OFX::Image* p_SrcImg) { _srcImg = p_SrcImg; }
    void setParams(const float radius[3], const float sigmaR[3], const float amount[3],
                   int nSamples, unsigned int seedStatic, unsigned int seedMoving, float motion, bool monochrome,
                   bool matchCurve, float trimS, float trimM, float trimH, float structure, float coupling);
    void setHalation(float intensity, const float tint[3], float sigma, float threshold, float hiGain) {
        _halIntensity = intensity; _halSigma = sigma; _halThresh = threshold; _halGain = hiGain;
        _halTint[0] = tint[0]; _halTint[1] = tint[1]; _halTint[2] = tint[2];
    }
    // CPU-only: precompute the blurred highlight halo from the source image.
    void computeHaloCPU();
    // Compute the CPU halo only when not using a GPU path (GPU does it on-device).
    void prepareHalationCPU() {
        if (!_isEnabledMetalRender && !_isEnabledCudaRender && !_isEnabledOpenCLRender) computeHaloCPU();
        else _halo.clear();
    }

private:
    OFX::Image* _srcImg = nullptr;
    float _radius[3] = {0.85f, 0.85f, 0.85f};
    float _sigmaR[3] = {0, 0, 0};
    float _amount[3] = {1, 1, 1};
    int   _nSamples = 64;
    unsigned int _seedStatic = 2026u;   // pattern when not moving (frame 0)
    unsigned int _seedMoving = 2026u;    // pattern for the current (held) frame
    float _motion = 1.0f;                // 0 = static plate, 1 = full per-frame movement
    bool  _monochrome = false;
    bool  _matchCurve = true;
    float _trimS = 1.0f, _trimM = 1.0f, _trimH = 1.0f;
    float _structure = 0.0f;
    float _coupling = 0.75f;
    float _halIntensity = 0.0f;
    float _halTint[3] = {1.0f, 0.30f, 0.15f};
    float _halSigma = 0.0f;
    float _halThresh = 0.65f;
    float _halGain = 1.0f;
    std::vector<float> _halo;          // CPU halo (per pixel), over src bounds
    int _halX0 = 0, _halY0 = 0, _halW = 0, _halH = 0;
};

void FilmGrainProcessor::computeHaloCPU() {
    if (_halIntensity <= 0.0f || _halSigma <= 0.25f || !_srcImg) { _halo.clear(); return; }
    const OfxRectI b = _srcImg->getBounds();
    const int W = b.x2 - b.x1, H = b.y2 - b.y1;
    if (W <= 0 || H <= 0) { _halo.clear(); return; }
    _halX0 = b.x1; _halY0 = b.y1; _halW = W; _halH = H;

    static const float kLR = 0.2126f, kLG = 0.7152f, kLB = 0.0722f;
    std::vector<float> mask((size_t)W * H, 0.0f), tmp((size_t)W * H, 0.0f);
    _halo.assign((size_t)W * H, 0.0f);

    // highlight extraction
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float* p = static_cast<float*>(_srcImg->getPixelAddress(b.x1 + x, b.y1 + y));
            float luma = p ? (p[0] * kLR + p[1] * kLG + p[2] * kLB) : 0.0f;
            mask[(size_t)y * W + x] = fgSmoothstep(_halThresh, 1.0f, luma * _halGain);
        }

    // separable Gaussian weights
    int R = (int)std::ceil(3.0f * _halSigma); if (R < 1) R = 1; if (R > 256) R = 256;
    std::vector<float> wt(2 * R + 1); float wsum = 0.0f;
    float inv2s2 = 1.0f / (2.0f * _halSigma * _halSigma);
    for (int i = -R; i <= R; ++i) { float w = std::exp(-(float)(i * i) * inv2s2); wt[i + R] = w; wsum += w; }
    for (auto& w : wt) w /= wsum;

    // horizontal then vertical
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int i = -R; i <= R; ++i) { int sx = x + i; if (sx < 0) sx = 0; else if (sx >= W) sx = W - 1; acc += wt[i + R] * mask[(size_t)y * W + sx]; }
            tmp[(size_t)y * W + x] = acc;
        }
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int i = -R; i <= R; ++i) { int sy = y + i; if (sy < 0) sy = 0; else if (sy >= H) sy = H - 1; acc += wt[i + R] * tmp[(size_t)sy * W + x]; }
            _halo[(size_t)y * W + x] = acc;
        }
}

FilmGrainProcessor::FilmGrainProcessor(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance) {}

void FilmGrainProcessor::setParams(const float radius[3], const float sigmaR[3], const float amount[3],
                                   int nSamples, unsigned int seedStatic, unsigned int seedMoving, float motion, bool monochrome,
                                   bool matchCurve, float trimS, float trimM, float trimH, float structure, float coupling) {
    for (int c = 0; c < 3; ++c) { _radius[c] = radius[c]; _sigmaR[c] = sigmaR[c]; _amount[c] = amount[c]; }
    _nSamples = nSamples;
    _seedStatic = seedStatic; _seedMoving = seedMoving; _motion = motion;
    _monochrome = monochrome;
    _matchCurve = matchCurve;
    _trimS = trimS; _trimM = trimM; _trimH = trimH;
    _structure = structure;
    _coupling = coupling;
}

#ifdef __APPLE__
extern void RunFilmGrainMetal(void* p_CmdQ, int p_Width, int p_Height, int p_OffX, int p_OffY,
                              const float* p_Radius, const float* p_SigmaR, const float* p_Amount,
                              int p_NSamples, unsigned int p_SeedStatic, unsigned int p_SeedMoving, float p_Motion, int p_Mono,
                              int p_MatchCurve, const float* p_GainLUT, int p_LUTSize, float p_FlatScale,
                              const float* p_Trim, float p_Structure, float p_Coupling,
                              float p_HalIntensity, const float* p_HalTint, float p_HalSigma, float p_HalThreshold,
                              float p_HalHiGain, const float* p_Input, float* p_Output);
#endif

void FilmGrainProcessor::processImagesMetal() {
#ifdef __APPLE__
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    float trim[3] = { _trimS, _trimM, _trimH };
    RunFilmGrainMetal(_pMetalCmdQ, width, height, bounds.x1, bounds.y1,
                      _radius, _sigmaR, _amount, _nSamples, _seedStatic, _seedMoving, _motion, _monochrome ? 1 : 0,
                      _matchCurve ? 1 : 0, kGainLUT, kGainLUTSize, kGainFlatScale, trim,
                      _structure, _coupling,
                      _halIntensity, _halTint, _halSigma, _halThresh,
                      _halGain, input, output);
#endif
}

void FilmGrainProcessor::processImagesCuda() {
#ifdef FILMGRAIN_CUDA
    const OfxRectI& bounds = _srcImg->getBounds();
    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    FGCudaParams P;
    P.width = bounds.x2 - bounds.x1; P.height = bounds.y2 - bounds.y1;
    P.offX = bounds.x1; P.offY = bounds.y1;
    P.r0 = _radius[0]; P.r1 = _radius[1]; P.r2 = _radius[2];
    P.s0 = _sigmaR[0]; P.s1 = _sigmaR[1]; P.s2 = _sigmaR[2];
    P.a0 = _amount[0]; P.a1 = _amount[1]; P.a2 = _amount[2];
    P.nSamples = _nSamples; P.seedStatic = _seedStatic; P.seedMoving = _seedMoving; P.motion = _motion; P.mono = _monochrome ? 1 : 0;
    P.matchCurve = _matchCurve ? 1 : 0; P.flatScale = kGainFlatScale;
    P.trimS = _trimS; P.trimM = _trimM; P.trimH = _trimH;
    P.structure = _structure; P.coupling = _coupling;
    P.halIntensity = _halIntensity;
    P.tintR = _halTint[0]; P.tintG = _halTint[1]; P.tintB = _halTint[2];
    P.halThreshold = _halThresh; P.halHiGain = _halGain; P.halSigma = _halSigma;

    RunFilmGrainCuda(_pCudaStream, P, kGainLUTSize, kGainLUT, input, output);
#endif
}

void FilmGrainProcessor::multiThreadProcessImages(OfxRectI p_ProcWindow) {
    static const float kLumaR = 0.2126f, kLumaG = 0.7152f, kLumaB = 0.0722f;

    for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y) {
        if (_effect.abort()) break;

        float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

        for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x) {
            float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : nullptr);

            if (srcPix) {
                // Composite halation (tinted, blurred highlight glow) BEFORE grain.
                float base[4] = { srcPix[0], srcPix[1], srcPix[2], srcPix[3] };
                if (_halIntensity > 0.0f && !_halo.empty()) {
                    int hx = x - _halX0, hy = y - _halY0;
                    if (hx >= 0 && hy >= 0 && hx < _halW && hy < _halH) {
                        float h = _halo[(size_t)hy * _halW + hx] * _halIntensity;
                        for (int c = 0; c < 3; ++c) {
                            float bc = base[c] < 0 ? 0 : (base[c] > 1 ? 1 : base[c]);
                            base[c] += _halTint[c] * h * (1.0f - bc);
                        }
                    }
                }

                // Grain Motion: blend a static-seed pattern (frame 0) and the
                // moving-seed pattern (current held frame). sqrt-weighting keeps
                // unit variance. nSeeds==1 at motion 0 or 1 -> no extra cost.
                unsigned int seeds[2]; float wts[2]; int nSeeds;
                if (_motion <= 0.0f)      { nSeeds = 1; seeds[0] = _seedStatic; wts[0] = 1.0f; }
                else if (_motion >= 1.0f) { nSeeds = 1; seeds[0] = _seedMoving; wts[0] = 1.0f; }
                else { nSeeds = 2; seeds[0] = _seedStatic; wts[0] = std::sqrt(1.0f - _motion * _motion);
                                   seeds[1] = _seedMoving; wts[1] = _motion; }

                if (_monochrome) {
                    float luma = base[0] * kLumaR + base[1] * kLumaG + base[2] * kLumaB;
                    float cl = luma < 0 ? 0 : (luma > 1 ? 1 : luma);
                    float g = 0.0f;
                    for (int s = 0; s < nSeeds; ++s) {
                        grain::Params p; p.grainRadius = _radius[0]; p.sigmaR = _sigmaR[0];
                        p.nSamples = _nSamples; p.seed = seeds[s];
                        g += wts[s] * grainFluct(cl, x, y, 0, p, _structure);
                    }
                    float eg = effectiveGain(cl, _matchCurve, _trimS, _trimM, _trimH);
                    float delta = g * eg * _amount[0];
                    for (int c = 0; c < 3; ++c) dstPix[c] = base[c] + delta;
                    dstPix[3] = base[3];
                } else {
                    // Shared luminance grain (correlated across channels) + independent
                    // per-channel grain, mixed sqrt(coupling)*common + sqrt(1-coupling)*indep.
                    float wc = std::sqrt(_coupling), wi = std::sqrt(1.0f - _coupling);
                    float luma = base[0] * kLumaR + base[1] * kLumaG + base[2] * kLumaB;
                    float clL = luma < 0 ? 0 : (luma > 1 ? 1 : luma);
                    float nf[3] = {0, 0, 0};
                    for (int s = 0; s < nSeeds; ++s) {
                        float common = 0.0f;
                        if (_coupling > 0.0f) {
                            grain::Params pc; pc.grainRadius = _radius[1]; pc.sigmaR = _sigmaR[1];
                            pc.nSamples = _nSamples; pc.seed = seeds[s];
                            common = grainFluct(clL, x, y, 6, pc, _structure);
                        }
                        for (int c = 0; c < 3; ++c) {
                            float cl = base[c] < 0 ? 0 : (base[c] > 1 ? 1 : base[c]);
                            float indep = 0.0f;
                            if (_coupling < 1.0f) {
                                grain::Params p; p.grainRadius = _radius[c]; p.sigmaR = _sigmaR[c];
                                p.nSamples = _nSamples; p.seed = seeds[s];
                                indep = grainFluct(cl, x, y, c, p, _structure);
                            }
                            nf[c] += wts[s] * (wc * common + wi * indep);
                        }
                    }
                    for (int c = 0; c < 3; ++c) {
                        float cl = base[c] < 0 ? 0 : (base[c] > 1 ? 1 : base[c]);
                        float eg = effectiveGain(cl, _matchCurve, _trimS, _trimM, _trimH);
                        dstPix[c] = base[c] + nf[c] * eg * _amount[c];
                    }
                    dstPix[3] = base[3];
                }
            } else {
                for (int c = 0; c < 4; ++c) dstPix[c] = 0;
            }
            dstPix += 4;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Plugin

class FilmGrainPlugin : public OFX::ImageEffect {
public:
    explicit FilmGrainPlugin(OfxImageEffectHandle p_Handle);

    virtual void render(const OFX::RenderArguments& p_Args);
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

    void setupAndProcess(FilmGrainProcessor& p_Proc, const OFX::RenderArguments& p_Args);
    void setEnabledness();

private:
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;

    OFX::ChoiceParam*  m_Format;
    OFX::DoubleParam*  m_Amount;
    OFX::DoubleParam*  m_GrainSize;
    OFX::DoubleParam*  m_Irregularity;
    OFX::IntParam*     m_Quality;
    OFX::BooleanParam* m_Monochrome;
    OFX::BooleanParam* m_Animate;
    OFX::DoubleParam*  m_Motion;
    OFX::IntParam*     m_FrameHold;
    OFX::IntParam*     m_Seed;
    OFX::BooleanParam* m_MatchCurve;
    OFX::DoubleParam*  m_TrimS; OFX::DoubleParam* m_TrimM; OFX::DoubleParam* m_TrimH;
    OFX::DoubleParam*  m_Structure;
    OFX::DoubleParam*  m_Coupling;
    OFX::DoubleParam*  m_HalIntensity;
    OFX::DoubleParam*  m_HalSize;
    OFX::DoubleParam*  m_HalThreshold;
    OFX::DoubleParam*  m_HalGain;
    OFX::RGBParam*     m_HalTint;
    OFX::DoubleParam*  m_AmountR; OFX::DoubleParam* m_AmountG; OFX::DoubleParam* m_AmountB;
    OFX::DoubleParam*  m_SizeR;   OFX::DoubleParam* m_SizeG;   OFX::DoubleParam* m_SizeB;
};

FilmGrainPlugin::FilmGrainPlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle) {
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

    m_Format       = fetchChoiceParam("format");
    m_Amount       = fetchDoubleParam("amount");
    m_GrainSize    = fetchDoubleParam("grainSize");
    m_Irregularity = fetchDoubleParam("irregularity");
    m_Quality      = fetchIntParam("quality");
    m_Monochrome   = fetchBooleanParam("monochrome");
    m_Animate      = fetchBooleanParam("animate");
    m_Motion       = fetchDoubleParam("motionAmount");
    m_FrameHold    = fetchIntParam("frameHold");
    m_Seed         = fetchIntParam("seed");
    m_MatchCurve   = fetchBooleanParam("matchCurve");
    m_TrimS = fetchDoubleParam("trimShadow"); m_TrimM = fetchDoubleParam("trimMid"); m_TrimH = fetchDoubleParam("trimHigh");
    m_Structure = fetchDoubleParam("structure");
    m_Coupling = fetchDoubleParam("colorCoupling");
    m_HalIntensity = fetchDoubleParam("halation");
    m_HalSize = fetchDoubleParam("halationSize");
    m_HalThreshold = fetchDoubleParam("halationThreshold");
    m_HalGain = fetchDoubleParam("halationHighlightGain");
    m_HalTint = fetchRGBParam("halationTint");
    m_AmountR = fetchDoubleParam("amountR"); m_AmountG = fetchDoubleParam("amountG"); m_AmountB = fetchDoubleParam("amountB");
    m_SizeR   = fetchDoubleParam("sizeR");   m_SizeG   = fetchDoubleParam("sizeG");   m_SizeB   = fetchDoubleParam("sizeB");

    setEnabledness();
}

void FilmGrainPlugin::render(const OFX::RenderArguments& p_Args) {
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) &&
        (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA)) {
        FilmGrainProcessor proc(*this);
        setupAndProcess(proc, p_Args);
    } else {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool FilmGrainPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime) {
    double amount = m_Amount->getValueAtTime(p_Args.time);
    double halation = m_HalIntensity->getValueAtTime(p_Args.time);
    if (amount <= 0.0 && halation <= 0.0) {   // nothing to do only if both are off
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }
    return false;
}

void FilmGrainPlugin::changedParam(const OFX::InstanceChangedArgs& /*p_Args*/, const std::string& p_ParamName) {
    if (p_ParamName == "monochrome") setEnabledness();
}

void FilmGrainPlugin::setEnabledness() {
    bool perChannel = !m_Monochrome->getValue();
    m_AmountR->setEnabled(perChannel); m_AmountG->setEnabled(perChannel); m_AmountB->setEnabled(perChannel);
    m_SizeR->setEnabled(perChannel);   m_SizeG->setEnabled(perChannel);   m_SizeB->setEnabled(perChannel);
}

void FilmGrainPlugin::setupAndProcess(FilmGrainProcessor& p_Proc, const OFX::RenderArguments& p_Args) {
    std::unique_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));
    std::unique_ptr<OFX::Image> src(m_SrcClip->fetchImage(p_Args.time));

    if ((src->getPixelDepth() != dst->getPixelDepth()) ||
        (src->getPixelComponents() != dst->getPixelComponents())) {
        OFX::throwSuiteStatusException(kOfxStatErrValue);
    }

    const double t = p_Args.time;

    // 7219 (Super 16) and 5219 (Super 35) are the SAME Vision3 500T emulsion; the
    // larger S35 negative is enlarged less to a given output, so both apparent
    // grain size and RMS scale by the frame-width ratio k = 11.78/24.89 = 0.4733
    // (Selwyn granularity: sigma ~ 1/frameWidth). S16 is the measured reference.
    // The soft-kernel model normalizes amplitude independently of grain size, so
    // BOTH size and amount scale by exactly k (no compensation needed).
    int format = 0;
    m_Format->getValueAtTime(t, format);
    const double formatSizeScale   = (format == 1) ? (11.78 / 24.89) : 1.0;
    const double formatAmountScale = formatSizeScale;

    const double amount  = m_Amount->getValueAtTime(t) * formatAmountScale;
    const double size    = m_GrainSize->getValueAtTime(t) * formatSizeScale;
    const double irreg   = m_Irregularity->getValueAtTime(t);
    const int    quality = m_Quality->getValueAtTime(t);
    const bool   mono    = m_Monochrome->getValueAtTime(t);
    const bool   animate = m_Animate->getValueAtTime(t);
    const int    seedBase= m_Seed->getValueAtTime(t);
    const int    frameHold = m_FrameHold->getValueAtTime(t);
    const float  motionAmt = (float)m_Motion->getValueAtTime(t);
    const bool   matchCurve = m_MatchCurve->getValueAtTime(t);
    const float  trimS = (float)m_TrimS->getValueAtTime(t);
    const float  trimM = (float)m_TrimM->getValueAtTime(t);
    const float  trimH = (float)m_TrimH->getValueAtTime(t);
    const float  structure = (float)m_Structure->getValueAtTime(t);
    const float  coupling = (float)m_Coupling->getValueAtTime(t);
    const float  halIntensity = (float)m_HalIntensity->getValueAtTime(t);
    const float  halThreshold = (float)m_HalThreshold->getValueAtTime(t);
    const float  halGain = (float)m_HalGain->getValueAtTime(t);
    double htR, htG, htB; m_HalTint->getValueAtTime(t, htR, htG, htB);
    const float  halTint[3] = { (float)htR, (float)htG, (float)htB };

    const double aR = m_AmountR->getValueAtTime(t), aG = m_AmountG->getValueAtTime(t), aB = m_AmountB->getValueAtTime(t);
    const double sR = m_SizeR->getValueAtTime(t),   sG = m_SizeG->getValueAtTime(t),   sB = m_SizeB->getValueAtTime(t);

    // Keep grain size constant in screen pixels when rendering at proxy scale.
    const double rs = p_Args.renderScale.x > 0 ? p_Args.renderScale.x : 1.0;

    float radius[3], sigmaR[3], amt[3];
    if (mono) {
        radius[0] = radius[1] = radius[2] = (float)(size * rs);
        sigmaR[0] = sigmaR[1] = sigmaR[2] = (float)(size * rs * irreg);
        amt[0] = amt[1] = amt[2] = (float)amount;
    } else {
        radius[0] = (float)(size * sR * rs); radius[1] = (float)(size * sG * rs); radius[2] = (float)(size * sB * rs);
        sigmaR[0] = radius[0] * (float)irreg; sigmaR[1] = radius[1] * (float)irreg; sigmaR[2] = radius[2] * (float)irreg;
        amt[0] = (float)(amount * aR); amt[1] = (float)(amount * aG); amt[2] = (float)(amount * aB);
    }

    // Frame Hold (#2): grain pattern updates every N frames (1 = on ones, 2 = on twos...).
    int hold = frameHold < 1 ? 1 : frameHold;
    int frameIdx = (int)std::floor(t / (double)hold);
    // Distinct constant so the static pattern never coincides with a moving frame
    // (else the static+moving blend would double-count and over-strengthen grain).
    unsigned int seedStatic = (unsigned int)seedBase * 2654435761u + 0x9e3779b9u;
    unsigned int seedMoving = (unsigned int)seedBase * 2654435761u + (unsigned int)frameIdx * 40503u + 1u;
    // Motion Amount (#3): 0 = static plate, 1 = full per-frame movement. Animate off forces static.
    float motion = animate ? (motionAmt < 0 ? 0.0f : (motionAmt > 1 ? 1.0f : motionAmt)) : 0.0f;

    // Halation blur radius (sigma) in screen pixels, scaled for proxy.
    const float halSigma = (float)(m_HalSize->getValueAtTime(t) * rs);

    p_Proc.setDstImg(dst.get());
    p_Proc.setSrcImg(src.get());
    p_Proc.setGPURenderArgs(p_Args);
    p_Proc.setRenderWindow(p_Args.renderWindow);
    p_Proc.setParams(radius, sigmaR, amt, quality < 1 ? 1 : quality, seedStatic, seedMoving, motion, mono,
                     matchCurve, trimS, trimM, trimH, structure, coupling);
    p_Proc.setHalation(halIntensity, halTint, halSigma, halThreshold, halGain);
    p_Proc.prepareHalationCPU();   // builds the CPU halo only on the CPU path

    p_Proc.process();
}

////////////////////////////////////////////////////////////////////////////////
// Factory

using namespace OFX;

FilmGrainFactory::FilmGrainFactory()
    : OFX::PluginFactoryHelper<FilmGrainFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor) {}

void FilmGrainFactory::describe(OFX::ImageEffectDescriptor& p_Desc) {
    p_Desc.setLabels(kPluginName, kPluginName, kPluginName);
    p_Desc.setPluginGrouping(kPluginGrouping);
    p_Desc.setPluginDescription(kPluginDescription);

    p_Desc.addSupportedContext(eContextFilter);
    p_Desc.addSupportedContext(eContextGeneral);
    p_Desc.addSupportedBitDepth(eBitDepthFloat);

    p_Desc.setSingleInstance(false);
    p_Desc.setHostFrameThreading(false);
    p_Desc.setSupportsMultiResolution(kSupportsMultiResolution);
    p_Desc.setSupportsTiles(kSupportsTiles);
    p_Desc.setTemporalClipAccess(false);
    p_Desc.setRenderTwiceAlways(false);
    p_Desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);

#ifdef __APPLE__
    p_Desc.setSupportsMetalRender(true);
#endif
#ifdef FILMGRAIN_CUDA
    p_Desc.setSupportsCudaRender(true);
    p_Desc.setSupportsCudaStream(true);
#endif
}

static DoubleParamDescriptor* defineDouble(OFX::ImageEffectDescriptor& d, const std::string& name, const std::string& label,
                                           const std::string& hint, double def, double lo, double hi,
                                           double dlo, double dhi, GroupParamDescriptor* parent) {
    DoubleParamDescriptor* p = d.defineDoubleParam(name);
    p->setLabels(label, label, label);
    p->setScriptName(name);
    p->setHint(hint);
    p->setDefault(def);
    p->setRange(lo, hi);
    p->setDisplayRange(dlo, dhi);
    if (parent) p->setParent(*parent);
    return p;
}

void FilmGrainFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/) {
    ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);

    ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->setSupportsTiles(kSupportsTiles);

    PageParamDescriptor* page = p_Desc.definePageParam("Controls");

    // Film format: selects the enlargement that scales grain size & strength.
    // Same emulsion (Vision3 500T); Super 16 (7219) is the measured reference.
    ChoiceParamDescriptor* format = p_Desc.defineChoiceParam("format");
    format->setLabels("Film Format", "Film Format", "Film Format");
    format->setHint("Vision3 500T on either gauge. Same emulsion; the format sets the enlargement. "
                    "Super 16 (7219) is the scanned reference. Super 35 (5219) is enlarged ~2.1x less, "
                    "so its grain is ~0.47x the size and strength (finer, cleaner). Sliders below are relative to this.");
    format->appendOption("Super 16mm \xE2\x80\x94 Vision3 500T 7219");
    format->appendOption("Super 35mm \xE2\x80\x94 Vision3 500T 5219");
    format->setDefault(0);
    page->addChild(*format);

    // Defaults below are calibrated to a Kodak Vision3 500T grey-card scan
    // (see analysis/plugin_defaults.json): green grain radius ~1.0px, blue
    // ~18% coarser & ~20% grainier, red ~5% finer/cleaner. Base amount ~0.04
    // reproduces the scan's mid-tone grain RMS 1:1 (1080p, at display levels).
    DoubleParamDescriptor* amount = defineDouble(p_Desc, "amount", "Amount",
        "Overall grain strength. With Match 500T Curve on, 1.0 reproduces the scan's grain at every exposure; raise for a heavier look.",
        1.0, 0.0, 8.0, 0.0, 3.0, nullptr);
    page->addChild(*amount);

    BooleanParamDescriptor* matchCurve = p_Desc.defineBooleanParam("matchCurve");
    matchCurve->setLabels("Match 500T Curve", "Match 500T Curve", "Match 500T Curve");
    matchCurve->setHint("Drive grain strength by the measured 500T grain-vs-exposure curve (U-shaped: stronger in shadow/highlight). Off = uniform grain across all levels.");
    matchCurve->setDefault(true);
    page->addChild(*matchCurve);

    DoubleParamDescriptor* size = defineDouble(p_Desc, "grainSize", "Grain Size",
        "Grain radius in pixels (at full res). 0.85 ~ 500T at 1080p. Smaller = finer/tighter, larger = softer/clumpier.",
        0.85, 0.10, 6.0, 0.2, 3.0, nullptr);
    page->addChild(*size);

    DoubleParamDescriptor* structure = defineDouble(p_Desc, "structure", "Grain Structure",
        "Blends in a coarser second grain layer for dimensional clumping/depth (0 = single fine layer). "
        "Overall strength is preserved; only the texture changes.",
        0.35, 0.0, 1.0, 0.0, 1.0, nullptr);
    page->addChild(*structure);

    DoubleParamDescriptor* irreg = defineDouble(p_Desc, "irregularity", "Irregularity",
        "Spread of the grain-size distribution (0 = uniform discs, higher = log-normal mix). Note: values > 0 are noticeably slower (per-grain size sampling).",
        0.0, 0.0, 1.0, 0.0, 1.0, nullptr);
    page->addChild(*irreg);

    IntParamDescriptor* quality = p_Desc.defineIntParam("quality");
    quality->setLabels("Quality", "Quality", "Quality");
    quality->setHint("Anti-alias supersampling of the grain field. 1 is ideal for normal grain; raise to 4-9 only if very fine grain (small size) looks aliased. Higher = slower.");
    quality->setDefault(1);
    quality->setRange(1, 16);
    quality->setDisplayRange(1, 9);
    page->addChild(*quality);

    BooleanParamDescriptor* mono = p_Desc.defineBooleanParam("monochrome");
    mono->setLabels("Monochrome Grain", "Monochrome Grain", "Monochrome Grain");
    mono->setHint("Drive one luminance grain layer applied to all channels (less chroma noise) instead of independent RGB grain.");
    mono->setDefault(false);
    page->addChild(*mono);

    DoubleParamDescriptor* coupling = defineDouble(p_Desc, "colorCoupling", "Color Coupling",
        "How correlated the grain is across R/G/B. Real film grain is highly cross-color correlated "
        "(mostly a shared luminance grain). 1 = monochrome-like (least chroma noise), 0 = fully independent "
        "RGB grain (most chroma noise). ~0.75 is film-like.",
        0.75, 0.0, 1.0, 0.0, 1.0, nullptr);
    page->addChild(*coupling);

    // --- Halation: warm highlight glow from light reflecting off the film base
    // (the effect anti-halation layers suppress; pronounced in rem-jet-removed
    // 500T, i.e. CineStill 800T). Extract highlights -> blur -> tint -> screen. ---
    GroupParamDescriptor* hal = p_Desc.defineGroupParam("halationGroup");
    hal->setLabels("Halation", "Halation", "Halation");
    hal->setHint("Warm highlight glow from light reflecting off the film base (CineStill-style, rem-jet-removed 500T).");
    hal->setOpen(false);

    DoubleParamDescriptor* halI = defineDouble(p_Desc, "halation", "Halation Intensity",
        "Strength of the highlight glow. 0 = off; 0.2-0.5 reads as classic film halation.",
        0.0, 0.0, 2.0, 0.0, 1.0, hal);
    page->addChild(*halI);
    DoubleParamDescriptor* halSz = defineDouble(p_Desc, "halationSize", "Halation Size",
        "Glow spread radius in pixels (at full resolution). Larger = wider, softer halo.",
        12.0, 1.0, 100.0, 2.0, 50.0, hal);
    page->addChild(*halSz);
    DoubleParamDescriptor* halTh = defineDouble(p_Desc, "halationThreshold", "Halation Threshold",
        "Luminance above which highlights begin to bloom. Lower = more of the image glows.",
        0.65, 0.0, 1.0, 0.3, 1.0, hal);
    page->addChild(*halTh);
    DoubleParamDescriptor* halHi = defineDouble(p_Desc, "halationHighlightGain", "Highlight Gain",
        "Scales luma into the highlight range so halation works wherever this node sits. "
        "Leave at 1 for display-referred footage; raise (2-4) if applied early on log/flat footage where highlights sit low.",
        1.0, 0.1, 16.0, 0.5, 4.0, hal);
    page->addChild(*halHi);
    RGBParamDescriptor* halTint = p_Desc.defineRGBParam("halationTint");
    halTint->setLabels("Halation Color", "Halation Color", "Halation Color");
    halTint->setHint("Tint of the glow. Default warm red-orange — real halation is red-dominant (long wavelengths penetrate and reflect most).");
    halTint->setDefault(1.0, 0.30, 0.15);
    halTint->setParent(*hal);
    page->addChild(*halTint);

    BooleanParamDescriptor* animate = p_Desc.defineBooleanParam("animate");
    animate->setLabels("Animate", "Animate", "Animate");
    animate->setHint("Master switch for grain movement. Off = static grain plate (Motion Amount / Frame Hold ignored).");
    animate->setDefault(true);
    page->addChild(*animate);

    DoubleParamDescriptor* motion = defineDouble(p_Desc, "motionAmount", "Motion Amount",
        "How much the grain moves frame to frame: 0 = static plate, 1 = fully re-randomized each (held) frame. "
        "Values between blend a static base with moving grain for a calmer 'boil'. (Animate must be on.)",
        1.0, 0.0, 1.0, 0.0, 1.0, nullptr);
    page->addChild(*motion);

    IntParamDescriptor* frameHold = p_Desc.defineIntParam("frameHold");
    frameHold->setLabels("Frame Hold", "Frame Hold", "Frame Hold");
    frameHold->setHint("Hold each grain pattern for N frames before it changes. 1 = new grain every frame (on ones); 2 = on twos; 3 = on threes (classic film cadence).");
    frameHold->setDefault(1);
    frameHold->setRange(1, 24);
    frameHold->setDisplayRange(1, 6);
    page->addChild(*frameHold);

    IntParamDescriptor* seed = p_Desc.defineIntParam("seed");
    seed->setLabels("Seed", "Seed", "Seed");
    seed->setHint("Base random seed for the grain pattern.");
    seed->setDefault(2026);
    seed->setRange(0, 1000000);
    seed->setDisplayRange(0, 9999);
    page->addChild(*seed);

    GroupParamDescriptor* tonal = p_Desc.defineGroupParam("tonalGrain");
    tonal->setLabels("Tonal Grain Trim", "Tonal Grain Trim", "Tonal Grain Trim");
    tonal->setHint("Multiply grain strength in the shadows / mid-tones / highlights. Blended smoothly across the tonal range.");
    tonal->setOpen(false);
    page->addChild(*defineDouble(p_Desc, "trimShadow", "Shadow Grain", "Grain multiplier in the shadows.",   1.0, 0.0, 4.0, 0.0, 2.0, tonal));
    page->addChild(*defineDouble(p_Desc, "trimMid",    "Mid Grain",    "Grain multiplier in the mid-tones.", 1.0, 0.0, 4.0, 0.0, 2.0, tonal));
    page->addChild(*defineDouble(p_Desc, "trimHigh",   "Highlight Grain", "Grain multiplier in the highlights.", 1.0, 0.0, 4.0, 0.0, 2.0, tonal));

    GroupParamDescriptor* perCh = p_Desc.defineGroupParam("perChannel");
    perCh->setLabels("Per-Channel (RGB)", "Per-Channel (RGB)", "Per-Channel (RGB)");
    perCh->setHint("Independent grain strength and size per channel. Real film grain differs between dye layers.");
    perCh->setOpen(false);

    // 500T-calibrated per-channel balance (G normalized to 1).
    page->addChild(*defineDouble(p_Desc, "amountR", "Red Amount",   "Red-channel grain strength multiplier.",   0.947, 0.0, 4.0, 0.0, 2.0, perCh));
    page->addChild(*defineDouble(p_Desc, "amountG", "Green Amount", "Green-channel grain strength multiplier.", 1.000, 0.0, 4.0, 0.0, 2.0, perCh));
    page->addChild(*defineDouble(p_Desc, "amountB", "Blue Amount",  "Blue-channel grain strength multiplier.",  1.202, 0.0, 4.0, 0.0, 2.0, perCh));
    page->addChild(*defineDouble(p_Desc, "sizeR",   "Red Size",     "Red-channel grain size multiplier.",       0.957, 0.25, 4.0, 0.5, 2.0, perCh));
    page->addChild(*defineDouble(p_Desc, "sizeG",   "Green Size",   "Green-channel grain size multiplier.",     1.000, 0.25, 4.0, 0.5, 2.0, perCh));
    page->addChild(*defineDouble(p_Desc, "sizeB",   "Blue Size",    "Blue-channel grain size multiplier.",      1.178, 0.25, 4.0, 0.5, 2.0, perCh));
}

ImageEffect* FilmGrainFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/) {
    return new FilmGrainPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray) {
    static FilmGrainFactory factory;
    p_FactoryArray.push_back(&factory);
}
