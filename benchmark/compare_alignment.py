"""Compare two image alignment strategies: Crop vs Resize.
Uses existing registered images, no re-registration needed.
"""
import os, sys
_torch_lib = os.path.join(sys.prefix, "lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_lib):
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_torch_lib)
import torch  # noqa

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tifffile import imread
from skimage.transform import resize
from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim,
    mean_squared_error as mse,
    normalized_root_mse as nrmse,
)

BENCHMARK_DIR = Path(__file__).resolve().parent
UNREG_DIR = BENCHMARK_DIR / "unregistration"
REG_DIR = BENCHMARK_DIR / "registration"
OUTPUT_DIR = BENCHMARK_DIR / "results"
ROUNDS = ["A", "C", "D", "E"]
SAMPLE_GROUPS = [1, 2, 3]


def to_2d_gray(img):
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[0] <= 10 and img.shape[1] > 10 and img.shape[2] > 10:
            return img[-1]
        if img.shape[2] <= 4:
            return img[..., -1]
        return np.max(img, axis=0)
    return img


def ncc(img1, img2):
    a, b = img1.astype(np.float64), img2.astype(np.float64)
    m1, m2, s1, s2 = a.mean(), b.mean(), a.std(), b.std()
    if s1 < 1e-10 or s2 < 1e-10:
        return 0.0
    return float(np.sum((a - m1) * (b - m2)) / (a.size * s1 * s2))


def mi(img1, img2, bins=256):
    a = np.clip(img1, 0, None)
    b = np.clip(img2, 0, None)
    a = (a / max(a.max(), 1) * (bins - 1)).astype(np.int32)
    b = (b / max(b.max(), 1) * (bins - 1)).astype(np.int32)
    jh = np.zeros((bins, bins), dtype=np.float64)
    np.add.at(jh, (a.ravel(), b.ravel()), 1)
    jh /= jh.sum()
    p1, p2, nz = jh.sum(1), jh.sum(0), jh > 0
    return float(np.sum(jh[nz] * np.log2(jh[nz] / (p1[:, None] * p2[None, :])[nz])))


def compute_metrics(reg, gt):
    r, g = reg.astype(np.float64), gt.astype(np.float64)
    dr = max(g.max() - g.min(), r.max() - r.min(), 1e-10)
    md = min(r.shape[0], r.shape[1])
    ws = min(7, md if md % 2 == 1 else md - 1)
    if ws < 3:
        ws = 3
    return {
        "PSNR": float(psnr(g, r, data_range=dr)),
        "SSIM": float(ssim(g, r, data_range=dr, win_size=ws)),
        "MSE": float(mse(g, r)),
        "NRMSE": float(nrmse(g, r)),
        "NCC": ncc(r, g),
        "MI": mi(r, g),
    }


# ── Alignment Method A: CROP to common region ───────────────────────────────
def align_crop(reg_2d, gt_2d):
    h = min(reg_2d.shape[0], gt_2d.shape[0])
    w = min(reg_2d.shape[1], gt_2d.shape[1])
    return reg_2d[:h, :w].astype(np.float64), gt_2d[:h, :w].astype(np.float64)


# ── Alignment Method B: RESIZE registered to GT dimensions ──────────────────
def align_resize(reg_2d, gt_2d):
    if reg_2d.shape != gt_2d.shape:
        reg_2d = resize(reg_2d.astype(np.float64), gt_2d.shape,
                        preserve_range=True, anti_aliasing=True)
    return reg_2d.astype(np.float64), gt_2d.astype(np.float64)


# ── Discover pairs ──────────────────────────────────────────────────────────
def discover_pairs():
    pairs = []
    for sg in SAMPLE_GROUPS:
        ref = None
        for f in (UNREG_DIR / f"C-{sg}").iterdir():
            if "C3-B" in f.name:
                ref = f
                break
        for rnd in ROUNDS:
            src = gt = None
            for f in (UNREG_DIR / f"C-{sg}").iterdir():
                if f"C3-{rnd}-63-{sg}" in f.name:
                    src = f
                    break
            for f in (REG_DIR / f"C1-{sg}").iterdir():
                if f"C3-{rnd}-63-{sg}" in f.name and f"C3-B-63-{sg}" not in f.name:
                    gt = f
                    break
            if src and gt and ref:
                pairs.append((sg, rnd, src, ref, gt))
    return pairs


def main():
    print("=" * 80)
    print("  COMPARISON: Crop vs Resize alignment")
    print("=" * 80)

    pairs = discover_pairs()
    print(f"\nFound {len(pairs)} pairs\n")

    rows_crop = []
    rows_resize = []

    for i, (sg, rnd, src_path, ref_path, gt_path) in enumerate(pairs):
        label = f"G{sg}-{rnd}"
        reg_path = OUTPUT_DIR / f"registered_G{sg}_{rnd}_to_B.tif"
        if not reg_path.exists():
            print(f"  [{i+1}] {label}: SKIP")
            continue

        reg_img = to_2d_gray(imread(str(reg_path)))
        gt_img = to_2d_gray(imread(str(gt_path)))

        print(f"  [{i+1}/{len(pairs)}] {label}: ours={reg_img.shape} gt={gt_img.shape} diff=({reg_img.shape[0]-gt_img.shape[0]:+d}, {reg_img.shape[1]-gt_img.shape[1]:+d})")

        # Method A: Crop
        r_crop, g_crop = align_crop(reg_img, gt_img)
        m_crop = compute_metrics(r_crop, g_crop)
        m_crop["pair"] = label
        rows_crop.append(m_crop)

        # Method B: Resize
        r_rsz, g_rsz = align_resize(reg_img, gt_img)
        m_rsz = compute_metrics(r_rsz, g_rsz)
        m_rsz["pair"] = label
        rows_resize.append(m_rsz)

        print(f"    CROP:   PSNR={m_crop['PSNR']:.2f}  SSIM={m_crop['SSIM']:.4f}  NCC={m_crop['NCC']:.4f}  MI={m_crop['MI']:.4f}")
        print(f"    RESIZE: PSNR={m_rsz['PSNR']:.2f}  SSIM={m_rsz['SSIM']:.4f}  NCC={m_rsz['NCC']:.4f}  MI={m_rsz['MI']:.4f}")

    df_crop = pd.DataFrame(rows_crop)
    df_resize = pd.DataFrame(rows_resize)

    # Summary comparison
    metrics = ["PSNR", "SSIM", "NCC", "MI"]
    print("\n" + "=" * 80)
    print(f"{'Metric':<10} {'Crop (avg)':<15} {'Resize (avg)':<15} {'Winner':<10} {'Improvement':<15}")
    print("-" * 65)
    for m in metrics:
        avg_c = df_crop[m].mean()
        avg_r = df_resize[m].mean()
        winner = "CROP" if avg_c >= avg_r else "RESIZE"
        diff = avg_c - avg_r
        pct = diff / abs(avg_r) * 100 if abs(avg_r) > 1e-10 else 0
        print(f"{m:<10} {avg_c:<15.4f} {avg_r:<15.4f} {winner:<10} {pct:+.2f}%")

    # For error metrics (lower is better)
    for m in ["MSE", "NRMSE"]:
        avg_c = df_crop[m].mean()
        avg_r = df_resize[m].mean()
        winner = "CROP" if avg_c <= avg_r else "RESIZE"
        diff = avg_r - avg_c
        pct = diff / abs(avg_r) * 100 if abs(avg_r) > 1e-10 else 0
        print(f"{m:<10} {avg_c:<15.4f} {avg_r:<15.4f} {winner:<10} {pct:+.2f}% (lower=better)")
    print("=" * 80)

    # Save comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Crop vs Resize Alignment Comparison", fontsize=16, fontweight="bold")
    labels = df_crop["pair"].tolist()
    x = np.arange(len(labels))
    w = 0.35

    for idx, m in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        bars1 = ax.bar(x - w/2, df_crop[m].values, w, label="Crop", color="#4CAF50", alpha=0.85)
        bars2 = ax.bar(x + w/2, df_resize[m].values, w, label="Resize", color="#2196F3", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(m)
        ax.set_title(m, fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "crop_vs_resize_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Chart saved: {out_path}")

    # Save CSVs
    df_crop.to_csv(OUTPUT_DIR / "metrics_crop.csv", index=False)
    df_resize.to_csv(OUTPUT_DIR / "metrics_resize.csv", index=False)
    print("✓ CSVs saved: metrics_crop.csv, metrics_resize.csv")
    print("\nDone!")


if __name__ == "__main__":
    main()
