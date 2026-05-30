# Film Grain (Stochastic) — OFX plugin

A physically-based film-grain plugin for OpenFX hosts (DaVinci Resolve, Fusion,
Nuke, Natron, …). Grain is generated with a **soft-kernel stochastic silver-grain
model** (a continuous, bipolar density field) rather than a static noise texture
or hard dots, so it is:

- **Dimensional, not dotty** — overlapping soft grains that darken *and* lighten,
  strongest in the mid-tones and tapered toward black/white like real emulsion.
- **Resolution-independent** — grain size is defined in pixels and behaves
  consistently across resolutions and proxy scales.
- **Per-channel** — independent grain in R/G/B (real dye layers differ), or a
  single monochrome luminance layer.
- **GPU-accelerated** — Apple **Metal** (macOS) and **NVIDIA CUDA** (Windows/Linux)
  render paths, plus a multi-threaded CPU fallback. All paths share identical math
  (verified bit-identical via `src/GrainCudaCore.cuh`).

Defaults lean toward Kodak Vision3 **500T (5219)**; the model is meant to be
calibrated against the grey-card scan (see *Calibration* below).

## Model

Grain is a **soft-kernel shot-noise field**. Silver grains are scattered as a
Poisson process whose intensity `λ(u) = -ln(1-u)/(π·E[R²])` is set by the local
exposure `u` (so grain tracks density like real emulsion). Each grain contributes
a smooth compact bump `K(d) = (1-(d/R)²)²`; their sum `F(p) = Σ K(p-cᵢ)` is a
continuous density field. We output the **zero-mean fluctuation** `F - E[F]`,
normalised to unit variance by Campbell's theorem (`E[F]=λ·πR²/3`,
`Var[F]=λ·πR²/5`), then scaled by the calibrated per-level RMS and *Amount*.

Because the field is built from overlapping *soft* grains and is **bipolar**
(darkens *and* lightens), it reads as dimensional film grain rather than bright
dots stamped on top. A **tonal taper** fades grain toward pure black/white, where
additive noise could only clip one-sided into flat specks. Grain *amplitude* is
normalised independently of grain *size*, so changing size never changes how
strong the grain is. Evaluating a smooth field needs no Monte-Carlo sample loop,
which also makes it fast (real-time at 4K).

(The earlier releases used the Boolean disc-coverage model of Newson, Galerne,
Morel & Delon, CGF 2017; the soft-kernel formulation here is derived from the
same Poisson-grain physics but is smoother, bipolar, and much faster.)

**Cross-color correlation.** Real film grain is strongly correlated across R/G/B
(the silver/dye is physically shared), which keeps grain reading as luminance
texture rather than colored speckle — Oh, Kuo, Sun & Lei (SPIE 2008) rank this
the single most important property for naturalness. We model it with a shared
luminance grain field blended with independent per-channel grain
(`√c·common + √(1-c)·indep`, *Color Coupling* = c), preserving unit variance.
Their other findings are already reflected here: signal-dependent power peaking in
the mid-tones (our tonal taper), a pinkish/low-frequency-rich PSD (our *Grain
Structure* octave), and a near-Gaussian PDF (the soft shot-noise sum).

## Parameters

| Parameter | Meaning |
|---|---|
| **Film Format** | Super 16 (7219) or Super 35 (5219) — same emulsion, scaled by enlargement (see *Film format* below). |
| **Amount** | Overall grain strength. With *Match 500T Curve* on, **1.0 reproduces the scan** at every exposure; raise for heavier grain. |
| **Match 500T Curve** | Drive grain strength by the measured 500T grain-vs-exposure curve (U-shaped: stronger in shadow & highlight). Off = uniform grain at all levels. |
| **Grain Size** | Grain radius in pixels (~0.85 = 500T at 1080p). Smaller = finer/tighter, larger = softer/clumpier. |
| **Grain Structure** | Blends in a coarser second grain layer for dimensional clumping/depth (0 = single fine layer, ~0.35 default). Overall strength unchanged. |
| **Irregularity** | Spread of grain-size distribution (0 = uniform discs, higher = log-normal mix). |
| **Quality** | Anti-alias supersampling (default 1). Raise to 4–9 only if very fine grain looks aliased. |
| **Monochrome Grain** | One luminance grain layer for all channels (less chroma noise). |
| **Color Coupling** | How correlated grain is across R/G/B (0 = independent RGB = most chroma noise, 1 = monochrome-like). ~0.75 is film-like; real grain is highly cross-color correlated. |
| **Animate** | Reseed grain each frame (running-film shimmer) vs. static plate. |
| **Seed** | Base random seed. |
| **Tonal Grain Trim** | Shadow / Mid / Highlight grain multipliers, blended smoothly across the tonal range. |
| **Per-Channel (RGB)** | Independent strength & size multipliers per channel. |
| **Halation Intensity** | Warm highlight glow (light reflecting off the film base). 0 = off; 0.2–0.5 ≈ classic film halation. |
| **Halation Size** | Glow spread radius (px). Larger = wider, softer halo. |
| **Halation Threshold** | Luminance above which highlights bloom. Lower = more of the image glows. |
| **Highlight Gain** | Scales luma into the highlight range so halation works wherever the node sits. 1 for display-referred; raise (2-4) on log/flat footage where highlights sit low. |
| **Halation Color** | Glow tint. Default warm red-orange (halation is red-dominant). |

### Intensity response (Match 500T Curve)

The pure Boolean grain model peaks in the mid-tones, but the 500T scan's grain is
**U-shaped** vs exposure (strong in shadow/highlight, minimum near the lower mids).
A baked gain LUT — `gain(u) = scanRMS(u) / modelCoverageRMS(u)`, generated by
`tools/make_lut.py` into `src/GrainLUT.h` — reconciles them so output grain RMS
tracks the scan at every level (within ~1–10%). Turn it off for uniform grain
across the tonal range.

## Film format (Super 16 vs Super 35)

Kodak **7219** (Super 16) and **5219** (Super 35) are the *same* Vision3 500T
emulsion — identical crystals and areal granularity. 16mm only looks grainier
because of **enlargement**: the smaller negative is blown up more to a given output.
The calibration scan is the **Super 16 (7219)** reference; Super 35 is derived.

Magnification to fill `N` output pixels is `M ∝ N / frameWidth`, so by **Selwyn's
granularity law** (`σ = G/√(2A)`, aperture `A = (frameWidth/N)²`, same `G`) both
apparent grain **size** and **RMS** scale as `1/frameWidth`. The frame-width ratio
`k = 11.78 / 24.89 = 0.473` sets Super 35 = Super 16 with size ×k and amount ×k.
Verified: S35 grain ≈ 0.47× S16 across the tonal range — finer and cleaner.

## Halation

Halation is the warm glow around bright highlights caused by light passing through
the emulsion, **reflecting off the film base back into the sensitive layer**, and
re-exposing a spreading region around the highlight. Early anti-halation patents
describe exactly this mechanism and how to suppress it (Nadeau & Starck, US2131747,
1938: light "reflected onto the sensitive material from the support… produce[s] a
spreading" halo; Glickman, US2606833, 1952: anti-halation dyes need "broad
absorption in the **red** region"). It is **red-dominant** because long wavelengths
penetrate deepest and reflect most. Stocks with the anti-halation (rem-jet) backing
removed — e.g. **CineStill 800T, which is Vision3 500T minus the rem-jet** — show
pronounced red halation around lights.

We emulate it the way colorists do: extract highlights above a threshold, blur them
(separable Gaussian, GPU), tint warm red-orange, and screen-composite the glow back
**before** grain. Controls: *Halation Intensity / Size / Threshold / Color*. Keep it
subtle — the look is in the restraint. Cheap on GPU (~5 fps off the 1080p rate).

## Performance

The soft-kernel field needs no Monte-Carlo sample loop, so it is fast. Rough
figures on an Apple M1 Max, RGBA float, default Quality 1:

| Config | 1080p | 4K UHD |
|---|---|---|
| Default (Structure 0.35 + Coupling 0.75, per-channel) | ~72 fps | ~18 fps |
| Single layer (Structure 0, Coupling 0) | ~208 fps | ~56 fps |
| Monochrome Grain | fastest (one layer) | |

Notes:
- **Quality** = anti-alias supersampling. 1 is right for normal grain; only raise
  (4–9) if very fine grain (small *Grain Size*) looks aliased.
- **Monochrome Grain** does one grain layer instead of three (faster), with less
  chroma noise.
- **Irregularity > 0** adds per-grain size variation (slightly slower); 0 is fine
  for most looks.
- Real-time even at 4K, so no render cache is needed for playback.

## Build — Windows / Linux (NVIDIA CUDA)

Use the cross-platform CMake build — see **WINDOWS_BUILD.md**. In short:
`cmake -B build -G "Visual Studio 17 2022" -A x64 && cmake --build build --config Release`
produces `build/FilmGrain.ofx.bundle/Contents/Win64/FilmGrain.ofx`. Requires the
CUDA Toolkit (11.2+). NVIDIA GPUs use the CUDA path; other GPUs fall back to CPU.
(CMake also builds macOS and Linux from the same file.)

## Build (macOS, universal arm64 + x86_64)

Requires the Xcode command-line tools (clang). The OpenFX SDK is vendored in
`openfx/`.

```sh
make            # builds FilmGrain.ofx.bundle
make grain_test # builds the headless grain validation harness
make clean
```

## Install

OFX plugins live in `/Library/OFX/Plugins` (root-owned), so installing needs sudo:

```sh
make install
# or manually:
sudo cp -R FilmGrain.ofx.bundle /Library/OFX/Plugins/
```

Then restart DaVinci Resolve. The effect appears in the OpenFX library under the
**"Fenner"** group as **"Film Grain (Stochastic)"**.

## Validation

`make grain_test && ./grain_test [grainRadius] [sigmaR] [nSamples] [size]`
prints the measured grain mean/stddev across input levels (stddev should peak in
the mid-tones) and writes `grain_ramp.pgm`, a grey ramp with grain for visual
inspection.

## Calibration to a film scan

The defaults are reasonable but the intent is to calibrate to a real **500T**
grey-card scan by measuring, per channel:

1. **Grain RMS vs. exposure** — the noise stddev at each grey level → sets
   *Amount* and the per-channel amounts.
2. **Grain size** — from the spatial autocorrelation / power spectrum of a flat
   patch → sets *Grain Size* and per-channel size.

Those measurements feed back into the plugin defaults. See `tools/` and the
analysis notes for the calibration workflow.
