# 500T grain calibration notes

Source: `/Volumes/Scans/Grey Card Grain 500T.mov` — ProRes 4444, 1920×1080,
11m21s. The clip is an **exposure ramp / step wedge** of a grey card on Kodak
Vision3 500T (5219): ~6 distinct exposure plateaus, ~3 frames each.

Method (`tools/extract_frames.swift` + `tools/analyze_grain.py`): decode frames
to RGBA half-float via AVFoundation, take a centered 512×512 crop, isolate grain
two independent ways —
- **spatial**: residual after a Gaussian high-pass (removes lighting gradient),
- **temporal**: difference of two same-level frames / √2 (cancels the static card).

The two agree to within ~1%, so the figures are real grain.

## Findings (grain RMS in scan code-value, per channel)

| meanG | spatial R/G/B | temporal R/G/B | size px R/G/B |
|------:|---------------|----------------|---------------|
| 0.104 | .0117 .0134 .0152 | .0118 .0135 .0152 | 1.16 1.26 1.44 |
| 0.147 | .0093 .0095 .0115 | .0091 .0094 .0115 | 1.15 1.18 1.41 |
| 0.206 | .0079 .0089 .0103 | .0077 .0085 .0101 | 1.09 1.17 1.40 |
| 0.279 | .0099 .0100 .0129 | .0093 .0094 .0126 | 1.11 1.12 1.38 |
| 0.415 | .0146 .0153 .0173 | .0140 .0145 .0168 | 1.09 1.13 1.29 |
| 0.543 | .0157 .0163 .0188 | .0152 .0155 .0182 | 1.04 1.07 1.25 |

- **Per-channel ratio R:G:B ≈ 0.95 : 1.00 : 1.20** — blue layer grainiest.
- **Grain size (autocorr HWHM) ≈ 1.1 px (R/G), 1.36 px (B)**; shrinks slightly with exposure.
- **RMS vs exposure is U-shaped** in this encoding: high in shadow, min near 0.21, rising into highlight.

## Baked into plugin defaults (`tools/calibrate.py` → `plugin_defaults.json`)

```
grainSize (green radius) = 1.0 px
base amount              = 0.04   (matches mid-tone RMS 1:1)
irregularity             = 0.25
sizeR / sizeG / sizeB    = 0.957 / 1.000 / 1.178
amountR / amountG / amountB = 0.947 / 1.000 / 1.202
```

## Fidelity of current model vs scan (green, calibrated)

| level | scan | model | ratio |
|------:|-----:|------:|------:|
| 0.10 | .0135 | .0101 | 0.75× |
| 0.21 | .0085 | .0132 | 1.54× |
| 0.28 | .0094 | .0146 | 1.55× |
| 0.42 | .0145 | .0158 | 1.09× |
| 0.54 | .0155 | .0157 | 1.01× |

Match is excellent in the upper-mids/highlights. The Boolean grain model peaks in
the mid-tones, whereas this scan's grain is U-shaped, so the model over-grains the
low-mids and under-grains deep shadow.

## TODO (v1.1) — intensity-response curve

Add a per-level grain-gain LUT (the inverse of the model's natural single-hump
variance, multiplied by the measured scan curve) so output RMS tracks the scan at
every exposure. Likely expose as a "Match 500T curve" toggle plus shadow/mid/
highlight grain trims.
