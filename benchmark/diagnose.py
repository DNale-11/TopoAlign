"""Diagnose root cause - write results to file."""
import os, sys
_torch_lib = os.path.join(sys.prefix, "lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_lib):
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_torch_lib)
import torch  # noqa

from pathlib import Path
import numpy as np
from tifffile import imread
from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim,
)
from skimage.exposure import match_histograms
from scipy.signal import fftconvolve

BENCHMARK_DIR = Path(__file__).resolve().parent
UNREG_DIR = BENCHMARK_DIR / "unregistration"
REG_DIR = BENCHMARK_DIR / "registration"
OUTPUT_DIR = BENCHMARK_DIR / "results"

def to_2d_gray(img):
    if img.ndim == 2: return img
    if img.ndim == 3:
        if img.shape[0] <= 10 and img.shape[1] > 10 and img.shape[2] > 10: return img[-1]
        if img.shape[2] <= 4: return img[..., -1]
        return np.max(img, axis=0)
    return img

def ncc(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    m1, m2, s1, s2 = a.mean(), b.mean(), a.std(), b.std()
    if s1 < 1e-10 or s2 < 1e-10: return 0.0
    return float(np.sum((a - m1) * (b - m2)) / (a.size * s1 * s2))

lines = []
lines.append("=" * 80)
lines.append("  DIAGNOSTIC: Why are metrics poor?")
lines.append("=" * 80)

for sg in [1,2,3]:
    for rnd in ["A","C","D","E"]:
        lines.append(f"\n--- G{sg}-{rnd}->B ---")
        reg_path = OUTPUT_DIR / f"registered_G{sg}_{rnd}_to_B.tif"
        reg = to_2d_gray(imread(str(reg_path))).astype(np.float64)
        gt = None
        for f in (REG_DIR / f"C1-{sg}").iterdir():
            if f"C3-{rnd}-63-{sg}" in f.name and "C3-B" not in f.name:
                gt = to_2d_gray(imread(str(f))).astype(np.float64); break
        h, w = min(reg.shape[0], gt.shape[0]), min(reg.shape[1], gt.shape[1])
        reg, gt = reg[:h, :w], gt[:h, :w]

        lines.append(f"  Ours: min={reg.min():.0f} max={reg.max():.0f} mean={reg.mean():.1f} std={reg.std():.1f}")
        lines.append(f"  GT:   min={gt.min():.0f} max={gt.max():.0f} mean={gt.mean():.1f} std={gt.std():.1f}")

        dr = max(reg.max()-reg.min(), gt.max()-gt.min(), 1.0)
        lines.append(f"  Raw:          PSNR={psnr(gt,reg,data_range=dr):.2f}  SSIM={ssim(gt,reg,data_range=dr,win_size=7):.4f}  NCC={ncc(reg,gt):.4f}")

        rn = (reg-reg.min())/max(reg.max()-reg.min(),1e-10)
        gn = (gt-gt.min())/max(gt.max()-gt.min(),1e-10)
        lines.append(f"  Normalized:   PSNR={psnr(gn,rn,data_range=1.0):.2f}  SSIM={ssim(gn,rn,data_range=1.0,win_size=7):.4f}  NCC={ncc(rn,gn):.4f}")

        rm = match_histograms(reg, gt).astype(np.float64)
        dr2 = max(rm.max()-rm.min(), gt.max()-gt.min(), 1.0)
        lines.append(f"  Hist-matched: PSNR={psnr(gt,rm,data_range=dr2):.2f}  SSIM={ssim(gt,rm,data_range=dr2,win_size=7):.4f}  NCC={ncc(rm,gt):.4f}")

        rp = (reg > 0).sum()/reg.size*100
        gp = (gt > 0).sum()/gt.size*100
        lines.append(f"  Non-zero:     Ours={rp:.1f}%  GT={gp:.1f}%")

        rc, gc = reg-reg.mean(), gt-gt.mean()
        corr = fftconvolve(gc, rc[::-1,::-1], mode='full')
        pk = np.unravel_index(np.argmax(corr), corr.shape)
        dy, dx = pk[0]-(reg.shape[0]-1), pk[1]-(reg.shape[1]-1)
        lines.append(f"  Residual shift: dy={dy} dx={dx}")

report = "\n".join(lines)
with open(OUTPUT_DIR / "diagnostic_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print(report)
print("\nSaved to results/diagnostic_report.txt")
