"""Re-evaluate benchmark: recompute metrics & visualizations from saved registered images.
Self-contained (no import from run_benchmark to avoid torch DLL chain).
"""
import os, sys

# Fix torch DLL before any imports
_torch_lib = os.path.join(sys.prefix, "lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_lib):
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_torch_lib)
import torch  # noqa

from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
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

# ── Constants ────────────────────────────────────────────────────────────────
BENCHMARK_DIR = Path(__file__).resolve().parent
UNREG_DIR = BENCHMARK_DIR / "unregistration"
REG_DIR = BENCHMARK_DIR / "registration"
OUTPUT_DIR = BENCHMARK_DIR / "results"
ROUNDS = ["A", "C", "D", "E"]
SAMPLE_GROUPS = [1, 2, 3]


@dataclass
class ImagePair:
    sample_group: int
    round_name: str
    source_path: Path
    reference_path: Path
    ground_truth_path: Path


# ── Functions (self-contained copies with fix) ───────────────────────────────
def find_unreg_image(sg, rnd):
    folder = UNREG_DIR / f"C-{sg}"
    if not folder.exists(): return None
    for f in folder.iterdir():
        if f.suffix.lower() in (".tif", ".tiff") and f"C3-{rnd}-63-{sg}" in f.name:
            return f
    return None

def find_reg_ground_truth(sg, rnd):
    folder = REG_DIR / f"C1-{sg}"
    if not folder.exists(): return None
    for f in folder.iterdir():
        if f.suffix.lower() in (".tif", ".tiff") and f"C3-{rnd}-63-{sg}" in f.name:
            if f"C3-B-63-{sg}" in f.name: continue
            return f
    return None

def discover_image_pairs():
    pairs = []
    for sg in SAMPLE_GROUPS:
        ref = find_unreg_image(sg, "B")
        if ref is None: continue
        for rnd in ROUNDS:
            src = find_unreg_image(sg, rnd)
            gt = find_reg_ground_truth(sg, rnd)
            if src and gt:
                pairs.append(ImagePair(sg, rnd, src, ref, gt))
    return pairs

def to_2d_gray(img):
    """Extract 2D grayscale. For channels-first (C,H,W), use LAST channel (DAPI = ch5)."""
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[0] <= 10 and img.shape[1] > 10 and img.shape[2] > 10:
            return img[-1]  # last channel = DAPI (channel 5)
        if img.shape[2] <= 4:
            return img[..., -1]  # last channel
        return np.max(img, axis=0)
    return img

def align_images_for_comparison(registered, ground_truth):
    """Crop both images to their common overlapping region (no interpolation)."""
    reg_2d = to_2d_gray(registered)
    gt_2d = to_2d_gray(ground_truth)
    h = min(reg_2d.shape[0], gt_2d.shape[0])
    w = min(reg_2d.shape[1], gt_2d.shape[1])
    return reg_2d[:h, :w].astype(np.float64), gt_2d[:h, :w].astype(np.float64)

def normalized_cross_correlation(img1, img2):
    img1_f, img2_f = img1.astype(np.float64), img2.astype(np.float64)
    m1, m2 = img1_f.mean(), img2_f.mean()
    s1, s2 = img1_f.std(), img2_f.std()
    if s1 < 1e-10 or s2 < 1e-10: return 0.0
    return float(np.sum((img1_f - m1) * (img2_f - m2)) / (img1_f.size * s1 * s2))

def mutual_information(img1, img2, bins=256):
    img1_q = np.clip(img1, 0, None)
    img2_q = np.clip(img2, 0, None)
    max1 = max(img1_q.max(), 1)
    max2 = max(img2_q.max(), 1)
    img1_q = (img1_q / max1 * (bins - 1)).astype(np.int32)
    img2_q = (img2_q / max2 * (bins - 1)).astype(np.int32)
    jh = np.zeros((bins, bins), dtype=np.float64)
    np.add.at(jh, (img1_q.ravel(), img2_q.ravel()), 1)
    jh /= jh.sum()
    p1, p2 = jh.sum(1), jh.sum(0)
    nz = jh > 0
    return float(np.sum(jh[nz] * np.log2(jh[nz] / (p1[:, None] * p2[None, :])[nz])))

def compute_all_metrics(reg, gt):
    r, g = reg.astype(np.float64), gt.astype(np.float64)
    dr = max(g.max() - g.min(), r.max() - r.min(), 1e-10)
    md = min(r.shape[0], r.shape[1])
    ws = min(7, md if md % 2 == 1 else md - 1)
    if ws < 3: ws = 3
    return {
        "PSNR": float(psnr(g, r, data_range=dr)),
        "SSIM": float(ssim(g, r, data_range=dr, win_size=ws)),
        "MSE": float(mse(g, r)),
        "NRMSE": float(nrmse(g, r)),
        "NCC": normalized_cross_correlation(r, g),
        "MI": mutual_information(r, g),
    }


# ── Visualization ────────────────────────────────────────────────────────────
def plot_metrics_bar_chart(df, path):
    metrics = ["PSNR", "SSIM", "MSE", "NRMSE", "NCC", "MI"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Registration Quality Metrics", fontsize=16, fontweight="bold")
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    for idx, m in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        labels = [f"G{r['sample_group']:.0f}-{r['round_name']}→B" for _, r in df.iterrows()]
        vals = df[m].values
        bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="gray", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(m, fontsize=13, fontweight="bold"); ax.set_ylabel(m); ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{v:.3f}" if abs(v) < 1000 else f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {path.name}")

def plot_heatmap(df, path):
    metrics = ["PSNR", "SSIM", "MSE", "NRMSE", "NCC", "MI"]
    labels = [f"G{r['sample_group']:.0f}-{r['round_name']}→B" for _, r in df.iterrows()]
    data = df[metrics].values.astype(float)
    dn = np.zeros_like(data)
    for j in range(data.shape[1]):
        c = data[:, j]; rng = c.max() - c.min()
        dn[:, j] = (c - c.min()) / rng if rng > 1e-10 else 0.5
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels)*0.5+2)))
    im = ax.imshow(dn, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(metrics, fontsize=11)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(metrics)):
            v = data[i,j]; t = f"{v:.3f}" if abs(v)<1000 else f"{v:.1f}"
            ax.text(j,i,t,ha="center",va="center",fontsize=8,
                    color="white" if dn[i,j]>0.6 else "black")
    ax.set_title("Registration Quality Heatmap (normalized)", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8); plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {path.name}")

def plot_timing_chart(df, path):
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [f"G{r['sample_group']:.0f}-{r['round_name']}→B" for _, r in df.iterrows()]
    times = df["registration_time_sec"].values
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(times)))
    bars = ax.bar(range(len(times)), times, color=colors, edgecolor="gray", linewidth=0.5)
    for b, t in zip(bars, times):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{t:.1f}s", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Registration Time per Image Pair", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    avg = times.mean(); ax.axhline(avg, color="red", linestyle="--", alpha=0.7, label=f"Average: {avg:.1f}s")
    ax.legend(); plt.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {path.name}")

def plot_group_comparison(df, path):
    metrics = ["PSNR", "SSIM", "NCC"]
    groups = sorted(df["sample_group"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
    fig.suptitle("Metrics by Sample Group", fontsize=15, fontweight="bold")
    gc = plt.cm.tab10(np.linspace(0, 0.3, len(groups)))
    for mi, m in enumerate(metrics):
        ax = axes[mi]; x = np.arange(len(ROUNDS)); w = 0.8 / len(groups)
        for gi, g in enumerate(groups):
            gd = df[df["sample_group"] == g]
            vals = [gd[gd["round_name"]==r][m].values[0] if len(gd[gd["round_name"]==r])>0 else 0 for r in ROUNDS]
            ax.bar(x + gi*w, vals, w, label=f"Group {g:.0f}", color=gc[gi], edgecolor="gray", linewidth=0.5)
        ax.set_xticks(x + w*(len(groups)-1)/2); ax.set_xticklabels([f"{r}→B" for r in ROUNDS])
        ax.set_title(m, fontsize=13, fontweight="bold"); ax.set_ylabel(m); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  ✓ {path.name}")

def plot_difference_maps(pair, reg, gt, path):
    reg_2d, gt_2d = align_images_for_comparison(reg, gt)
    diff = np.abs(reg_2d - gt_2d)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f"Group {pair.sample_group} - Round {pair.round_name}→B", fontsize=14, fontweight="bold")
    src_2d = to_2d_gray(imread(str(pair.source_path))).astype(np.float64)
    axes[0].imshow(src_2d, cmap="gray"); axes[0].set_title("Source (Unregistered)"); axes[0].axis("off")
    axes[1].imshow(reg_2d, cmap="gray"); axes[1].set_title("Our Registration"); axes[1].axis("off")
    axes[2].imshow(gt_2d, cmap="gray"); axes[2].set_title("Ground Truth"); axes[2].axis("off")
    im = axes[3].imshow(diff, cmap="hot"); axes[3].set_title("Difference (|Ours - GT|)"); axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], shrink=0.8)
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=== Re-evaluating benchmark with fixed image handling ===\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify fix
    gt_test = imread(str(REG_DIR / "C1-1" /
        "MAX_C-1.lif - C3-A-63-1 Merged channel 1_MAX_C-1.lif - C3-A-63-1 Merged channel 1_xfm_0.tif"))
    gt_2d = to_2d_gray(gt_test)
    print(f"GT raw: {gt_test.shape} -> 2D: {gt_2d.shape}")
    assert gt_2d.ndim == 2 and gt_2d.shape[0] > 100, f"Fix broken! Got {gt_2d.shape}"
    print("✓ to_2d_gray verified\n")

    pairs = discover_image_pairs()
    print(f"Found {len(pairs)} pairs\n")

    csv_path = OUTPUT_DIR / "benchmark_results.csv"
    old_df = pd.read_csv(csv_path) if csv_path.exists() else None

    rows = []
    for i, pair in enumerate(pairs):
        label = f"G{pair.sample_group}-{pair.round_name}→B"
        reg_path = OUTPUT_DIR / f"registered_G{pair.sample_group}_{pair.round_name}_to_B.tif"
        if not reg_path.exists():
            print(f"  [{i+1}/{len(pairs)}] {label}: SKIP (no registered image)")
            continue
        reg_img = imread(str(reg_path))
        gt_img = imread(str(pair.ground_truth_path))
        reg_aligned, gt_aligned = align_images_for_comparison(reg_img, gt_img)
        metrics = compute_all_metrics(reg_aligned, gt_aligned)
        time_sec = 0.0
        if old_df is not None:
            m = old_df[(old_df["sample_group"]==pair.sample_group)&(old_df["round_name"]==pair.round_name)]
            if len(m) > 0: time_sec = m["registration_time_sec"].values[0]
        print(f"  [{i+1}/{len(pairs)}] {label}: PSNR={metrics['PSNR']:.2f} SSIM={metrics['SSIM']:.4f} NCC={metrics['NCC']:.4f}")
        row = {"sample_group": pair.sample_group, "round_name": pair.round_name,
               "registration_time_sec": time_sec}
        row.update(metrics)
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV saved: {csv_path}")

    print("\nGenerating plots...")
    plot_metrics_bar_chart(results_df, OUTPUT_DIR / "metrics_bar_chart.png")
    plot_heatmap(results_df, OUTPUT_DIR / "metrics_heatmap.png")
    plot_timing_chart(results_df, OUTPUT_DIR / "timing_chart.png")
    plot_group_comparison(results_df, OUTPUT_DIR / "group_comparison.png")

    print("\nGenerating diff maps...")
    for pair in pairs:
        reg_path = OUTPUT_DIR / f"registered_G{pair.sample_group}_{pair.round_name}_to_B.tif"
        if not reg_path.exists(): continue
        reg_img = imread(str(reg_path))
        gt_img = imread(str(pair.ground_truth_path))
        plot_difference_maps(pair, reg_img, gt_img,
                             OUTPUT_DIR / f"diff_G{pair.sample_group}_{pair.round_name}_to_B.png")

    print("\n" + "=" * 70)
    print(f"{'Pair':<15} {'PSNR':<10} {'SSIM':<10} {'NCC':<10} {'MI':<10} {'Time(s)':<10}")
    print("-" * 70)
    for _, r in results_df.iterrows():
        label = f"G{int(r['sample_group'])}-{r['round_name']}→B"
        print(f"{label:<15} {r['PSNR']:<10.2f} {r['SSIM']:<10.4f} {r['NCC']:<10.4f} {r['MI']:<10.4f} {r['registration_time_sec']:<10.2f}")
    print("-" * 70)
    print(f"{'Average':<15} {results_df['PSNR'].mean():<10.2f} {results_df['SSIM'].mean():<10.4f} "
          f"{results_df['NCC'].mean():<10.4f} {results_df['MI'].mean():<10.4f} "
          f"{results_df['registration_time_sec'].mean():<10.2f}")
    print("=" * 70)
    print("\nDone!")


if __name__ == "__main__":
    main()
