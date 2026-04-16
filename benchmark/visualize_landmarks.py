"""
Visualize MNN landmarks on the C2-1/C2-A-1 images.
Export overlay image showing all 871 matched points.
"""
from __future__ import annotations
import sys, os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_tl = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_tl):
    os.environ["PATH"] = _tl + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_tl)
try: import torch
except: pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "napari-cell-registration" / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import functools
print = functools.partial(print, flush=True)

from registration_runtime import (
    DEFAULT_REG_PARAMS, build_segmenter, ensure_registration_dependencies,
    load_segmentation_artifacts, to_2d_gray, run_registration, warp_image_with_transform,
)
from napari_cell_registration.core.workflow import apply_rigid_to_points


def main():
    ensure_registration_dependencies()
    reg_params = dict(DEFAULT_REG_PARAMS)
    segmenter = build_segmenter(reg_params, gpu=True)
    seg_cache = {}

    UNREG_ROOT = Path(r"F:\programme\nxy\Cell registration\benchmark\unregistration")
    fixed_path = UNREG_ROOT / "C2-1" / "fixed" / "C2-C-1.tif"
    moving_path = UNREG_ROOT / "C2-1" / "moving" / "C2-A-1.tif"

    ref_art, _ = load_segmentation_artifacts(fixed_path, segmenter, seg_cache)
    src_art, _ = load_segmentation_artifacts(moving_path, segmenter, seg_cache)
    ref_shape = tuple(int(v) for v in to_2d_gray(ref_art.image).shape[:2])

    print("Running registration...")
    registration = run_registration(src_art, ref_art, reg_params)
    rot = registration.transform.params[:2, :2]
    trans = registration.transform.params[:2, 2]

    # Get images
    fixed_img = to_2d_gray(ref_art.image).astype(float)
    moving_img = to_2d_gray(src_art.image).astype(float)
    warped_img = warp_image_with_transform(src_art.image, registration.transform, ref_shape, order=1)
    warped_img = to_2d_gray(np.asarray(warped_img)).astype(float)

    # Normalize for display
    def norm(img):
        p1, p99 = np.percentile(img, [1, 99])
        return np.clip((img - p1) / max(p99 - p1, 1), 0, 1)

    fixed_norm = norm(fixed_img)
    warped_norm = norm(warped_img)

    # MNN landmarks
    pts_fixed = ref_art.features[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    pts_moving = src_art.features[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    pts_moving_warped = apply_rigid_to_points(pts_moving, rot, trans)

    tree_f = cKDTree(pts_fixed)
    tree_m = cKDTree(pts_moving_warped)
    d_m2f, idx_m2f = tree_f.query(pts_moving_warped)
    d_f2m, idx_f2m = tree_m.query(pts_fixed)

    mnn_pairs = []
    for j in range(len(pts_moving_warped)):
        i = idx_m2f[j]
        if idx_f2m[i] == j and d_m2f[j] <= 8.0:
            mnn_pairs.append((i, j, d_m2f[j]))

    print(f"MNN landmarks: {len(mnn_pairs)}")

    OUT_DIR = Path(r"F:\programme\nxy\Cell registration\benchmark\results")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: Full overview with all MNN points ----
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Panel 1: Fixed image with fixed landmark points
    axes[0].imshow(fixed_norm, cmap="gray")
    fx = [pts_fixed[i, 0] for i, j, d in mnn_pairs]
    fy = [pts_fixed[i, 1] for i, j, d in mnn_pairs]
    axes[0].scatter(fx, fy, c="lime", s=3, alpha=0.6, linewidths=0)
    axes[0].set_title(f"Fixed (C2-C-1) + {len(mnn_pairs)} MNN landmarks", fontsize=14)
    axes[0].axis("off")

    # Panel 2: Warped moving with moving landmark points
    axes[1].imshow(warped_norm, cmap="gray")
    mx = [pts_moving_warped[j, 0] for i, j, d in mnn_pairs]
    my = [pts_moving_warped[j, 1] for i, j, d in mnn_pairs]
    axes[1].scatter(mx, my, c="red", s=3, alpha=0.6, linewidths=0)
    axes[1].set_title(f"Warped Moving (C2-A-1) + landmarks", fontsize=14)
    axes[1].axis("off")

    # Panel 3: Overlay with connection lines
    composite = np.zeros((*fixed_norm.shape, 3))
    composite[..., 1] = fixed_norm   # green = fixed
    composite[..., 0] = warped_norm  # red = moving
    axes[2].imshow(composite)
    # Color lines by distance
    dists = [d for i, j, d in mnn_pairs]
    for i_f, j_m, d in mnn_pairs:
        color = "lime" if d < 3.0 else ("yellow" if d < 5.0 else "red")
        axes[2].plot(
            [pts_fixed[i_f, 0], pts_moving_warped[j_m, 0]],
            [pts_fixed[i_f, 1], pts_moving_warped[j_m, 1]],
            color=color, linewidth=0.3, alpha=0.5,
        )
    axes[2].set_title(f"Overlay + match lines (green<3px, yellow<5px, red>5px)", fontsize=14)
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "C2-1_MNN_landmarks_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: C2-1_MNN_landmarks_overview.png")

    # ---- Figure 2: 4 zoomed regions ----
    h, w = fixed_norm.shape
    regions = [
        ("Top-Left",    0, 0, h//2, w//2),
        ("Top-Right",   0, w//2, h//2, w),
        ("Bottom-Left", h//2, 0, h, w//2),
        ("Bottom-Right", h//2, w//2, h, w),
    ]

    fig2, axes2 = plt.subplots(2, 2, figsize=(24, 24))
    for ax, (name, y0, x0, y1, x1) in zip(axes2.flat, regions):
        comp_crop = composite[y0:y1, x0:x1]
        ax.imshow(comp_crop)

        count = 0
        for i_f, j_m, d in mnn_pairs:
            fx, fy = pts_fixed[i_f]
            mx, my = pts_moving_warped[j_m]
            if x0 <= fx < x1 and y0 <= fy < y1:
                color = "lime" if d < 3.0 else ("yellow" if d < 5.0 else "red")
                ax.plot([fx - x0, mx - x0], [fy - y0, my - y0],
                        color=color, linewidth=0.5, alpha=0.7)
                ax.plot(fx - x0, fy - y0, "o", color="lime", markersize=2)
                ax.plot(mx - x0, my - y0, "o", color="red", markersize=2)
                count += 1

        ax.set_title(f"{name}: {count} landmarks", fontsize=14)
        ax.axis("off")

    plt.tight_layout()
    fig2.savefig(OUT_DIR / "C2-1_MNN_landmarks_zoomed.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: C2-1_MNN_landmarks_zoomed.png")

    # Stats by region
    print(f"\nLandmarks per quadrant:")
    for name, y0, x0, y1, x1 in regions:
        cnt = sum(1 for i, j, d in mnn_pairs
                  if x0 <= pts_fixed[i, 0] < x1 and y0 <= pts_fixed[i, 1] < y1)
        print(f"  {name}: {cnt}")

    # Distance distribution
    print(f"\nDistance distribution:")
    d_arr = np.array(dists)
    print(f"  < 1px: {(d_arr < 1).sum()}")
    print(f"  1-3px: {((d_arr >= 1) & (d_arr < 3)).sum()}")
    print(f"  3-5px: {((d_arr >= 3) & (d_arr < 5)).sum()}")
    print(f"  5-8px: {((d_arr >= 5) & (d_arr <= 8)).sum()}")


if __name__ == "__main__":
    main()
