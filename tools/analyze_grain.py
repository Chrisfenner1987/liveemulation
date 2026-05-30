#!/usr/bin/env python3
"""Analyze film-grain statistics from extracted grey-card crops.

Reads analysis/crop_rgba_f32.bin (RGBA float32, nframes x H x W x 4) and:
  1. Groups frames by exposure level (clusters mean luma).
  2. Isolates grain two ways:
       - spatial detrend (residual after Gaussian high-pass), and
       - temporal frame-difference within a level (cancels the static card).
  3. Measures per-channel grain RMS vs exposure.
  4. Measures grain size via the spatial autocorrelation of the residual.

Outputs a calibration table and a JSON summary.
"""
import json, sys, os
import numpy as np

DIR = sys.argv[1] if len(sys.argv) > 1 else "analysis"
meta = open(os.path.join(DIR, "crop_meta.txt")).read().split()
W, H, N = int(meta[0]), int(meta[1]), int(meta[2])
data = np.fromfile(os.path.join(DIR, "crop_rgba_f32.bin"), dtype=np.float32)
data = data.reshape(N, H, W, 4)[..., :3]  # drop alpha
print(f"loaded {N} frames, {W}x{H}, channels RGB")

CH = ["R", "G", "B"]

def gaussian_blur(img, sigma):
    """Separable Gaussian blur via FFT (handles float, no scipy dependency)."""
    n = img.shape[0]
    ax = np.fft.fftfreq(img.shape[0])
    ay = np.fft.fftfreq(img.shape[1])
    fx = np.exp(-2 * (np.pi * sigma * ax) ** 2)
    fy = np.exp(-2 * (np.pi * sigma * ay) ** 2)
    F = np.fft.fft2(img)
    F *= np.outer(fx, fy)
    return np.real(np.fft.ifft2(F))

def detrend(img, sigma=16.0):
    """High-pass: subtract low-frequency lighting/card variation."""
    return img - gaussian_blur(img, sigma)

# Per-frame mean luma to cluster exposure levels.
luma = data[..., 1].reshape(N, -1).mean(axis=1)  # green as proxy
order = np.argsort(luma)
# cluster: greedy grouping where consecutive sorted lumas within 1% are same level
levels = []
cur = [order[0]]
for i in order[1:]:
    if abs(luma[i] - luma[cur[-1]]) < 0.012:
        cur.append(i)
    else:
        levels.append(cur); cur = [i]
levels.append(cur)
print(f"detected {len(levels)} exposure levels")

rows = []
for grp in levels:
    grp = list(grp)
    meanRGB = data[grp].reshape(len(grp), -1, 3).mean(axis=(0, 1))

    # --- spatial grain RMS (detrended residual), averaged over frames in group
    spat = np.zeros(3)
    autosize = np.zeros(3)
    for c in range(3):
        res_all = []
        for f in grp:
            res = detrend(data[f, :, :, c])
            res_all.append(res)
            spat[c] += res.std()
        spat[c] /= len(grp)
        # grain size from autocorrelation of the residual (use first frame)
        res = res_all[0]
        res = res - res.mean()
        F = np.fft.fft2(res)
        ac = np.real(np.fft.ifft2(F * np.conj(F)))
        ac = np.fft.fftshift(ac)
        ac /= ac.max()
        cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
        # radial: find radius where autocorr falls to 0.5 along x and y near center
        prof = (ac[cy, cx:cx+8] + ac[cy:cy+8, cx]) / 2.0
        # half-width at half-max -> approximate grain radius (px)
        hwhm = 0.5
        r = 0.5
        for k in range(1, len(prof)):
            if prof[k] < hwhm:
                # linear interp
                r = (k - 1) + (prof[k-1] - hwhm) / (prof[k-1] - prof[k] + 1e-9)
                break
        else:
            r = len(prof)
        autosize[c] = r

    # --- temporal grain RMS: difference frames within the level (cancels card)
    temp = np.full(3, np.nan)
    if len(grp) >= 2:
        t = np.zeros(3)
        npair = 0
        for a in range(len(grp)):
            for b in range(a + 1, len(grp)):
                d = data[grp[a]].astype(np.float64) - data[grp[b]].astype(np.float64)
                t += d.reshape(-1, 3).std(axis=0) / np.sqrt(2.0)
                npair += 1
        temp = t / npair

    rows.append(dict(
        level=float(meanRGB[1]),
        meanR=float(meanRGB[0]), meanG=float(meanRGB[1]), meanB=float(meanRGB[2]),
        nframes=len(grp),
        spat=[float(x) for x in spat],
        temp=[float(x) for x in temp],
        size=[float(x) for x in autosize],
    ))

rows.sort(key=lambda r: r["level"])

print("\n=== Grain RMS vs exposure (code value) ===")
print("  meanG  | nf | spatial R/G/B (detrended)    | temporal R/G/B (frame-diff)  | size px R/G/B")
print("  -------+----+------------------------------+------------------------------+----------------")
for r in rows:
    s = r["spat"]; t = r["temp"]; z = r["size"]
    tstr = (f"{t[0]:.5f} {t[1]:.5f} {t[2]:.5f}" if not np.isnan(t[0]) else "      (single frame)        ")
    print(f"  {r['level']:.3f}  | {r['nframes']:2d} | {s[0]:.5f} {s[1]:.5f} {s[2]:.5f}      | {tstr} | {z[0]:.2f} {z[1]:.2f} {z[2]:.2f}")

# Per-channel grain ratios (relative to green), averaged over levels using temporal where available
use = [r["temp"] if not np.isnan(r["temp"][0]) else r["spat"] for r in rows]
use = np.array(use)
ratios = use / use[:, 1:2]
mean_ratio = ratios.mean(axis=0)
print("\nPer-channel grain ratio (R:G:B, G=1):  "
      f"{mean_ratio[0]:.3f} : 1.000 : {mean_ratio[2]:.3f}")
sizes = np.array([r["size"] for r in rows]).mean(axis=0)
print(f"Mean grain size (autocorr HWHM, px):   R {sizes[0]:.2f}  G {sizes[1]:.2f}  B {sizes[2]:.2f}")

json.dump(dict(rows=rows, ratio=mean_ratio.tolist(), size=sizes.tolist()),
          open(os.path.join(DIR, "grain_calibration.json"), "w"), indent=2)
print(f"\nwrote {DIR}/grain_calibration.json")
