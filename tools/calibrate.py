#!/usr/bin/env python3
"""Solve for the model grain radius and amount that match the measured scan.

Strategy:
  - For a grid of grain radii, render a flat mid-level field with the C++ model,
    measure its output coverage RMS and autocorrelation HWHM (grain size).
  - Pick the radius whose HWHM matches the scan's green-channel grain size.
  - The plugin output delta = amount * (coverage - level), so output RMS =
    amount * coverageRMS. Solve amount = scanRMS / coverageRMS at a reference
    level for each channel.
"""
import json, os, subprocess, sys
import numpy as np

DIR = "analysis"
calib = json.load(open(os.path.join(DIR, "grain_calibration.json")))
scan_size = calib["size"]      # [R,G,B] HWHM px
ratio = calib["ratio"]         # R:G:B grain ratio, G=1
rows = calib["rows"]

# Reference green RMS: use a mid-high level (where the model also has grain).
ref = min(rows, key=lambda r: abs(r["level"] - 0.45))
ref_rms_g = (ref["temp"][1] if not np.isnan(ref["temp"][1]) else ref["spat"][1])
print(f"reference level={ref['level']:.3f}  scan green RMS={ref_rms_g:.5f}")
print(f"scan grain size HWHM px R/G/B = {scan_size[0]:.2f}/{scan_size[1]:.2f}/{scan_size[2]:.2f}")

SZ = 256
NS = 256        # high samples to suppress estimator noise during calibration
LEVEL = 0.45

def autocorr_hwhm(res):
    res = res - res.mean()
    F = np.fft.fft2(res)
    ac = np.real(np.fft.ifft2(F * np.conj(F)))
    ac = np.fft.fftshift(ac); ac /= ac.max()
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    prof = (ac[cy, cx:cx+8] + ac[cy:cy+8, cx]) / 2.0
    for k in range(1, len(prof)):
        if prof[k] < 0.5:
            return (k - 1) + (prof[k-1] - 0.5) / (prof[k-1] - prof[k] + 1e-9)
    return float(len(prof))

def render(radius):
    subprocess.run(["./calib_field", str(radius), "0.25", str(NS), str(LEVEL), str(SZ),
                    "/tmp/field.bin"], check=True, capture_output=True)
    img = np.fromfile("/tmp/field.bin", dtype=np.float32).reshape(SZ, SZ)
    return img

print("\n radius |  cov RMS  |  HWHM px")
print(" -------+-----------+---------")
best = None
for radius in [0.5, 0.7, 0.85, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
    img = render(radius)
    rms = img.std()
    hwhm = autocorr_hwhm(img)
    print(f"  {radius:.2f}  |  {rms:.4f}  |  {hwhm:.2f}")
    d = abs(hwhm - scan_size[1])  # match green size
    if best is None or d < best[0]:
        best = (d, radius, rms, hwhm)

_, br, brms, bhwhm = best
print(f"\nbest green radius = {br:.2f}px (model HWHM {bhwhm:.2f} vs scan {scan_size[1]:.2f})")

# amount so model output RMS matches scan green RMS at the reference level
amount = ref_rms_g / brms
print(f"coverage RMS at r={br:.2f}, level={LEVEL}: {brms:.4f}")
print(f"=> base Amount = {amount:.4f}  (output RMS = Amount * coverageRMS)")

# per-channel: size multiplier from scan size ratio, amount multiplier from grain ratio
sizeR = scan_size[0] / scan_size[1]
sizeB = scan_size[2] / scan_size[1]
print(f"\n--- calibrated defaults ---")
print(f"grainSize (green radius) = {br:.2f}")
print(f"sizeR={sizeR:.3f}  sizeG=1.000  sizeB={sizeB:.3f}")
print(f"amountR={ratio[0]:.3f}  amountG=1.000  amountB={ratio[2]:.3f}")
print(f"base amount = {amount:.3f}")
print(f"irregularity = 0.25 (matches scan's slight size spread)")

json.dump(dict(grainSize=float(br), amount=float(amount), sizeR=float(sizeR), sizeB=float(sizeB),
               amountR=float(ratio[0]), amountB=float(ratio[2]), irregularity=0.25),
          open(os.path.join(DIR, "plugin_defaults.json"), "w"), indent=2)
print(f"\nwrote {DIR}/plugin_defaults.json")
