"""
Benchmark script for cell registration pipeline.

Runs the existing registration pipeline (Cellpose segmentation → feature extraction →
two-stage matching → RANSAC + KNN refinement → image warping) on unregistered images,
compares results against ground truth, and produces quality metrics + visualizations.

Usage:
    python run_benchmark.py
"""

# ── Fix Windows DLL loading for torch ────────────────────────────────────────
import argparse
import json
import os
import sys

# Prepend torch DLL directory to PATH so Windows can find c10.dll
_torch_lib = os.path.join(sys.prefix, "lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_lib):
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_torch_lib)

# Pre-import torch before any other torch-dependent package
import torch  # noqa: F401, E402

import time
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from tifffile import imread, imwrite
from skimage.measure import ransac
from skimage.transform import SimilarityTransform, resize
from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim,
    mean_squared_error as mse,
    normalized_root_mse as nrmse,
)

# ── Add project to path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "napari-cell-registration" / "src"))

from napari_cell_registration.core import (
    CellposeConfig,
    CellposeSegmenter,
    compute_cell_features,
    CellFeaturesConfig,
    greedy_match_cells,
    MatchingConfig,
)
from napari_cell_registration.core.matching import apply_transform_to_features, two_stage_match_cells
from napari_cell_registration.core.point_registration import (
    estimate_robust_transform,
    refine_transform_with_neighbors,
    refine_transform_with_phase_correlation,
    warp_image_with_transform,
)
from napari_cell_registration.core.validation import validate_matches

# ── Constants ─────────────────────────────────────────────────────────────────
BENCHMARK_DIR = Path(__file__).resolve().parent
UNREG_DIR = BENCHMARK_DIR / "unregistration"
REG_DIR = BENCHMARK_DIR / "registration"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

ROUNDS = ["A", "C", "D", "E"]  # rounds to register (B is reference)
SAMPLE_GROUPS = [1, 2, 3]

# Registration parameters aligned with the widget's robust workflow defaults.
DEFAULT_REG_PARAMS = {
    "cellpose_diameter": 10.0,
    "cellpose_flow_threshold": 0.0,
    "cellpose_cellprob_threshold": 0.0,
    "cellpose_min_size": 15,
    "position_weight": 3.0,
    "distance_threshold": 2.5,
    "spatial_window_size": 100.0,
    "top_k": 100,
    "min_cells_for_two_stage": 20,
    "coarse_top_k": 50,
    "coarse_distance_threshold": 2.0,
    "neighbor_k": 5,
    "neighbor_weight": 1.0,
    "landmark_weight": 10.0,
    "max_theta_deg": 10.0,
    "max_translation": 30.0,
    "max_scale_change": 0.05,
    "guided_min_position_weight": 4.0,
    "guided_spatial_window_cap": 60.0,
    "validation_min_confidence": 0.2,
    "validation_max_feature_diff": 0.7,
    "prefer_affine": True,
    "ransac_max_trials": 500,
    "ransac_residual_threshold": 3.0,
    "similarity_residual_threshold": 5.0,
    "fft_upsample_factor": 20,
    "fft_max_shift": 20.0,
    "fft_crop_ratio": 0.8,
    "fft_max_iterations": 2,
}
REG_PARAMS = DEFAULT_REG_PARAMS.copy()

# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class ImagePair:
    """One registration pair: round X → round B."""
    sample_group: int
    round_name: str  # A, C, D, E
    source_path: Path  # unregistered X image
    reference_path: Path  # unregistered B image (reference)
    ground_truth_path: Path  # registered ground truth (X→B)


@dataclass
class BenchmarkResult:
    """Result of one registration + evaluation."""
    sample_group: int
    round_name: str
    registration_time_sec: float
    metrics: Dict[str, float]
    registered_image: Optional[np.ndarray] = field(default=None, repr=False)


# ── File discovery ────────────────────────────────────────────────────────────
def find_unreg_image(sample_group: int, round_name: str) -> Optional[Path]:
    """Find unregistered image for a given sample group and round."""
    folder = UNREG_DIR / f"C-{sample_group}"
    if not folder.exists():
        return None
    pattern = f"C3-{round_name}-63-{sample_group}"
    for f in folder.iterdir():
        if f.suffix.lower() in (".tif", ".tiff") and pattern in f.name:
            return f
    return None


def find_reg_ground_truth(sample_group: int, round_name: str) -> Optional[Path]:
    """Find ground truth registered image for a given sample group and round."""
    folder = REG_DIR / f"C1-{sample_group}"
    if not folder.exists():
        return None
    pattern = f"C3-{round_name}-63-{sample_group}"
    for f in folder.iterdir():
        if f.suffix.lower() in (".tif", ".tiff") and pattern in f.name:
            # Skip the B reference image itself
            if f"C3-B-63-{sample_group}" in f.name:
                continue
            return f
    return None


def discover_image_pairs() -> List[ImagePair]:
    """Discover all image pairs for benchmarking."""
    pairs = []
    for sg in SAMPLE_GROUPS:
        ref_path = find_unreg_image(sg, "B")
        if ref_path is None:
            print(f"  ⚠ Reference image B not found for sample group {sg}, skipping")
            continue
        for rnd in ROUNDS:
            src = find_unreg_image(sg, rnd)
            gt = find_reg_ground_truth(sg, rnd)
            if src is None:
                print(f"  ⚠ Source image {rnd} not found for sample group {sg}, skipping")
                continue
            if gt is None:
                print(f"  ⚠ Ground truth {rnd}→B not found for sample group {sg}, skipping")
                continue
            pairs.append(ImagePair(
                sample_group=sg,
                round_name=rnd,
                source_path=src,
                reference_path=ref_path,
                ground_truth_path=gt,
            ))
    return pairs


# ── Image quality metrics ─────────────────────────────────────────────────────
def normalized_cross_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute NCC (normalized cross-correlation) between two images."""
    img1_f = img1.astype(np.float64)
    img2_f = img2.astype(np.float64)
    mean1, mean2 = img1_f.mean(), img2_f.mean()
    std1, std2 = img1_f.std(), img2_f.std()
    if std1 < 1e-10 or std2 < 1e-10:
        return 0.0
    n = img1_f.size
    ncc = np.sum((img1_f - mean1) * (img2_f - mean2)) / (n * std1 * std2)
    return float(ncc)


def mutual_information(img1: np.ndarray, img2: np.ndarray, bins: int = 256) -> float:
    """Compute mutual information between two images."""
    # Quantize to integer bins
    img1_q = np.clip(img1, 0, None)
    img2_q = np.clip(img2, 0, None)
    # Normalize to [0, bins-1]
    max1 = img1_q.max() if img1_q.max() > 0 else 1
    max2 = img2_q.max() if img2_q.max() > 0 else 1
    img1_q = (img1_q / max1 * (bins - 1)).astype(np.int32)
    img2_q = (img2_q / max2 * (bins - 1)).astype(np.int32)

    # Joint histogram
    joint_hist = np.zeros((bins, bins), dtype=np.float64)
    np.add.at(joint_hist, (img1_q.ravel(), img2_q.ravel()), 1)
    joint_hist /= joint_hist.sum()

    # Marginals
    p1 = joint_hist.sum(axis=1)
    p2 = joint_hist.sum(axis=0)

    # Mutual information
    nonzero = joint_hist > 0
    mi = np.sum(
        joint_hist[nonzero] * np.log2(joint_hist[nonzero] / (p1[:, None] * p2[None, :])[nonzero])
    )
    return float(mi)


def compute_all_metrics(registered: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """Compute all image quality metrics between registered image and ground truth."""
    # Ensure same dtype for comparison
    reg = registered.astype(np.float64)
    gt = ground_truth.astype(np.float64)

    # Data range for PSNR/SSIM
    data_range = max(gt.max() - gt.min(), reg.max() - reg.min())
    if data_range < 1e-10:
        data_range = 1.0

    # Determine appropriate win_size for SSIM
    min_dim = min(reg.shape[0], reg.shape[1])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if win_size < 3:
        win_size = 3

    metrics = {}
    metrics["PSNR"] = float(psnr(gt, reg, data_range=data_range))
    metrics["SSIM"] = float(ssim(gt, reg, data_range=data_range, win_size=win_size))
    metrics["MSE"] = float(mse(gt, reg))
    metrics["NRMSE"] = float(nrmse(gt, reg))
    metrics["NCC"] = normalized_cross_correlation(reg, gt)
    metrics["MI"] = mutual_information(reg, gt)

    return metrics


# ── Registration pipeline ────────────────────────────────────────────────────
def run_registration(
    source_img: np.ndarray,
    reference_img: np.ndarray,
    segmenter: CellposeSegmenter,
) -> Tuple[np.ndarray, float]:
    """
    Run the complete registration pipeline on a pair of images.

    Returns (registered_image, elapsed_seconds).
    """
    t_start = time.perf_counter()

    # ── 1. Segmentation ──────────────────────────────────────────────────
    mask_src, _, _ = segmenter.segment_array(source_img)
    mask_ref, _, _ = segmenter.segment_array(reference_img)

    # ── 2. Feature extraction ────────────────────────────────────────────
    feat_config = CellFeaturesConfig()
    feats_src = compute_cell_features(mask_src, feat_config)
    feats_ref = compute_cell_features(mask_ref, feat_config)

    n_src = len(feats_src)
    n_ref = len(feats_ref)
    print(f"    Cells found: source={n_src}, reference={n_ref}")

    if n_src < 3 or n_ref < 3:
        print("    ✗ Too few cells for registration, returning source image as-is")
        elapsed = time.perf_counter() - t_start
        return source_img.copy(), elapsed

    # ── 3. Two-stage matching ────────────────────────────────────────────
    pw = REG_PARAMS["position_weight"]
    dt = REG_PARAMS["distance_threshold"]
    sw = REG_PARAMS["spatial_window_size"]
    tk = REG_PARAMS["top_k"]
    match_result = two_stage_match_cells(
        feats_ref,
        feats_src,
        mask_ref.shape,
        position_weight=pw,
        top_k=tk,
        distance_threshold=dt,
        spatial_window_size=sw,
        min_cells_for_two_stage=REG_PARAMS["min_cells_for_two_stage"],
        coarse_top_k=REG_PARAMS["coarse_top_k"],
        coarse_distance_threshold=REG_PARAMS["coarse_distance_threshold"],
    )
    if not match_result.coarse_matches.empty:
        print(f"    Coarse matches: {len(match_result.coarse_matches)}")
        print(
            "    Coarse offset: "
            f"dx={match_result.coarse_offset_xy[0]:.2f}, dy={match_result.coarse_offset_xy[1]:.2f}"
        )
    matches = match_result.matches
    print(f"    Fine matches: {len(matches)}")

    # Validate matches
    config_val = MatchingConfig(
        distance_threshold=dt,
        min_confidence=REG_PARAMS["validation_min_confidence"],
        max_feature_diff=REG_PARAMS["validation_max_feature_diff"],
    )
    matches = validate_matches(matches, feats_ref, match_result.aligned_df2, config_val)
    print(f"    Validated matches: {len(matches)}")

    if len(matches) < 3:
        print("    ✗ Too few matches for transform estimation, returning source as-is")
        elapsed = time.perf_counter() - t_start
        return source_img.copy(), elapsed

    # ── 4. Transform estimation (RANSAC + AffineTransform) ────────────────
    pts_ref_yx = feats_ref.loc[matches["idx1"], ["centroid_y", "centroid_x"]].to_numpy()
    pts_src_yx = feats_src.loc[matches["idx2"], ["centroid_y", "centroid_x"]].to_numpy()
    pts_ref_xy = pts_ref_yx[:, ::-1]
    pts_src_xy = pts_src_yx[:, ::-1]
    robust = estimate_robust_transform(
        pts_ref_xy,
        pts_src_xy,
        prefer_affine=bool(REG_PARAMS["prefer_affine"]),
        residual_threshold=REG_PARAMS["ransac_residual_threshold"],
        similarity_residual_threshold=REG_PARAMS["similarity_residual_threshold"],
        max_trials=REG_PARAMS["ransac_max_trials"],
    )
    affine = robust.transform
    print(
        f"    RANSAC model: {robust.method} | "
        f"inliers: {robust.inlier_count}/{len(matches)} | "
        f"median residual: {robust.median_inlier_residual:.2f}px"
    )
    inliers = np.asarray(robust.inliers, dtype=bool)
    model_robust = affine

    if inliers is None or inliers.sum() < 4:
        # Fall back to SimilarityTransform (fewer parameters)
        print("    ⚠ Affine RANSAC failed, trying SimilarityTransform...")
        model_robust, inliers = ransac(
            (pts_src_xy, pts_ref_xy),
            SimilarityTransform,
            min_samples=3,
            residual_threshold=REG_PARAMS["similarity_residual_threshold"],
            max_trials=REG_PARAMS["ransac_max_trials"],
        )

    if inliers is None or inliers.sum() < 3:
        print("    ✗ RANSAC failed, using simple translation")
        tx = np.median(pts_ref_xy[:, 0] - pts_src_xy[:, 0])
        ty = np.median(pts_ref_xy[:, 1] - pts_src_xy[:, 1])
        affine = SimilarityTransform(translation=(tx, ty))
    else:
        print(f"    RANSAC inliers: {inliers.sum()}/{len(inliers)}")
        affine = model_robust

    guided_feats_src = apply_transform_to_features(feats_src, affine, mask_ref.shape)
    guided_spatial_cap = REG_PARAMS["guided_spatial_window_cap"]
    if guided_spatial_cap is None or guided_spatial_cap <= 0:
        guided_window = sw
    else:
        guided_window = sw if sw is None else min(sw, guided_spatial_cap)
    guided_config = MatchingConfig(
        position_weight=max(pw, REG_PARAMS["guided_min_position_weight"]),
        top_k=tk,
        distance_threshold=None if dt is None else dt + 0.5,
        spatial_window_size=guided_window,
    )
    guided_matches = greedy_match_cells(feats_ref, guided_feats_src, guided_config)
    guided_matches = validate_matches(guided_matches, feats_ref, guided_feats_src, config_val)
    if len(guided_matches) >= 3:
        guided_ref_yx = feats_ref.loc[guided_matches["idx1"], ["centroid_y", "centroid_x"]].to_numpy()
        guided_src_yx = feats_src.loc[guided_matches["idx2"], ["centroid_y", "centroid_x"]].to_numpy()
        guided = estimate_robust_transform(
            guided_ref_yx[:, ::-1],
            guided_src_yx[:, ::-1],
            prefer_affine=bool(REG_PARAMS["prefer_affine"]),
            residual_threshold=REG_PARAMS["ransac_residual_threshold"],
            similarity_residual_threshold=REG_PARAMS["similarity_residual_threshold"],
            max_trials=REG_PARAMS["ransac_max_trials"],
        )
        if guided.score() > robust.score():
            matches = guided_matches
            pts_ref_yx = guided_ref_yx
            pts_src_yx = guided_src_yx
            pts_ref_xy = guided_ref_yx[:, ::-1]
            pts_src_xy = guided_src_yx[:, ::-1]
            robust = guided
            affine = robust.transform
            print(
                f"    Guided rematch improved support: {robust.method}, "
                f"{robust.inlier_count}/{len(matches)} inliers"
            )
    if robust.inlier_count >= 3:
        pts_ref_xy = pts_ref_xy[robust.inliers]
        pts_src_xy = pts_src_xy[robust.inliers]
    else:
        pts_ref_xy = np.empty((0, 2), dtype=float)
        pts_src_xy = np.empty((0, 2), dtype=float)

    # ── 5. KNN neighbor refinement ───────────────────────────────────────
    print(f"    Refining with k={REG_PARAMS['neighbor_k']} neighbors (KNN)...")
    affine = refine_transform_with_neighbors(
        pts_ref_xy,
        pts_src_xy,
        initial=affine,
        k=REG_PARAMS["neighbor_k"],
        neighbor_weight=REG_PARAMS["neighbor_weight"],
        landmark_weight=REG_PARAMS["landmark_weight"],
        max_theta_deg=REG_PARAMS["max_theta_deg"],
        max_translation=REG_PARAMS["max_translation"],
        max_scale_change=REG_PARAMS["max_scale_change"],
    )
    if robust.method != "similarity" and len(pts_ref_xy) >= 3:
        affine = refine_transform_with_neighbors(
            pts_ref_xy,
            pts_src_xy,
            initial=affine,
            k=REG_PARAMS["neighbor_k"],
            neighbor_weight=REG_PARAMS["neighbor_weight"],
            landmark_weight=REG_PARAMS["landmark_weight"],
            max_theta_deg=REG_PARAMS["max_theta_deg"],
            max_translation=REG_PARAMS["max_translation"],
            max_scale_change=REG_PARAMS["max_scale_change"],
            optimize_translation_only=True,
        )
    print("    KNN refinement complete")

    try:
        affine_fft, shift_yx = refine_transform_with_phase_correlation(
            source_img,
            reference_img,
            affine,
            output_shape=reference_img.shape[:2],
            upsample_factor=REG_PARAMS["fft_upsample_factor"],
            max_shift=REG_PARAMS["fft_max_shift"],
            crop_ratio=REG_PARAMS["fft_crop_ratio"],
            max_iterations=REG_PARAMS["fft_max_iterations"],
        )
        shift_mag = float(np.linalg.norm(shift_yx))
        if shift_mag > 0:
            print(f"    FFT correction: dy={shift_yx[0]:.2f} dx={shift_yx[1]:.2f} (mag={shift_mag:.2f}px)")
            affine = affine_fft
            print("    FFT refinement applied")
    except Exception as e:
        print(f"    FFT refinement failed: {e}")

    registered_f = warp_image_with_transform(
        source_img,
        affine,
        reference_img.shape[:2],
        order=1,
    )
    if np.issubdtype(source_img.dtype, np.integer):
        dtype_info = np.iinfo(source_img.dtype)
        registered = np.clip(np.rint(registered_f), dtype_info.min, dtype_info.max).astype(source_img.dtype)
    else:
        registered = registered_f.astype(source_img.dtype, copy=False)

    elapsed = time.perf_counter() - t_start
    return registered, elapsed
    print("    ✓ KNN refinement complete")

    # ── 6. Apply warp ────────────────────────────────────────────────────
    if source_img.ndim == 2:
        registered = warp(
            source_img.astype(float),
            inverse_map=affine.inverse,
            output_shape=reference_img.shape[:2],
            preserve_range=True,
        ).astype(source_img.dtype)
    elif source_img.ndim == 3:
        registered = np.zeros(
            (*reference_img.shape[:2], source_img.shape[2]),
            dtype=source_img.dtype,
        )
        for c in range(source_img.shape[2]):
            registered[..., c] = warp(
                source_img[..., c].astype(float),
                inverse_map=affine.inverse,
                output_shape=reference_img.shape[:2],
                preserve_range=True,
            ).astype(source_img.dtype)
    else:
        registered = source_img.copy()

    # ── 7. FFT phase-correlation refinement ──────────────────────────────
    #    Correct any residual sub-pixel or small-pixel shift after cell-based
    #    registration by using intensity-based cross-correlation.
    ref_f = reference_img.astype(np.float64)
    reg_f = registered.astype(np.float64)
    if ref_f.ndim == 3:
        ref_f = ref_f[..., 0] if ref_f.shape[2] <= 4 else ref_f.mean(axis=-1)
    if reg_f.ndim == 3:
        reg_f = reg_f[..., 0] if reg_f.shape[2] <= 4 else reg_f.mean(axis=-1)
    # Crop to same size for phase correlation
    h_pc = min(ref_f.shape[0], reg_f.shape[0])
    w_pc = min(ref_f.shape[1], reg_f.shape[1])
    try:
        shift_yx, _error, _phasediff = phase_cross_correlation(
            ref_f[:h_pc, :w_pc], reg_f[:h_pc, :w_pc],
            upsample_factor=10,  # sub-pixel precision
        )
        shift_mag = np.sqrt(shift_yx[0]**2 + shift_yx[1]**2)
        if shift_mag < 30:  # only apply if reasonable
            print(f"    FFT correction: dy={shift_yx[0]:.2f} dx={shift_yx[1]:.2f} (mag={shift_mag:.2f}px)")
            registered = ndi_shift(
                registered.astype(np.float64), shift_yx, order=3, mode='constant', cval=0
            ).astype(source_img.dtype)
            print("    ✓ FFT refinement applied")
        else:
            print(f"    FFT shift too large ({shift_mag:.1f}px), skipping")
    except Exception as e:
        print(f"    ⚠ FFT refinement failed: {e}")

    elapsed = time.perf_counter() - t_start
    return registered, elapsed


# ── Visualization ─────────────────────────────────────────────────────────────
def to_2d_gray(img: np.ndarray) -> np.ndarray:
    """Convert image to 2D grayscale for metric computation.
    
    For channels-first (C, H, W): use LAST channel (= DAPI / ch5).
    For channels-last (H, W, C): use LAST channel.
    """
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        # Detect channels-first (C, H, W) vs channels-last (H, W, C)
        if img.shape[0] <= 10 and img.shape[1] > 10 and img.shape[2] > 10:
            return img[-1]  # last channel = DAPI (channel 5)
        if img.shape[2] <= 4:
            return img[..., -1]  # last channel
        # Otherwise treat as z-stack, max project
        return np.max(img, axis=0)
    return img


def align_images_for_comparison(
    registered: np.ndarray, ground_truth: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop both images to their common overlapping region (no interpolation)."""
    reg_2d = to_2d_gray(registered)
    gt_2d = to_2d_gray(ground_truth)
    h = min(reg_2d.shape[0], gt_2d.shape[0])
    w = min(reg_2d.shape[1], gt_2d.shape[1])
    return reg_2d[:h, :w].astype(np.float64), gt_2d[:h, :w].astype(np.float64)


def plot_metrics_bar_chart(results_df: pd.DataFrame, output_path: Path) -> None:
    """Create bar charts for each metric across all pairs."""
    metrics_cols = ["PSNR", "SSIM", "MSE", "NRMSE", "NCC", "MI"]
    n_metrics = len(metrics_cols)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Registration Quality Metrics", fontsize=16, fontweight="bold")

    colors = plt.cm.Set2(np.linspace(0, 1, len(results_df)))

    for idx, metric in enumerate(metrics_cols):
        ax = axes[idx // 3, idx % 3]
        labels = [f"G{r['sample_group']}-{r['round_name']}→B" for _, r in results_df.iterrows()]
        values = results_df[metric].values

        bars = ax.bar(range(len(values)), values, color=colors, edgecolor="gray", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}" if abs(val) < 1000 else f"{val:.1f}",
                ha="center", va="bottom", fontsize=7,
            )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved metrics bar chart: {output_path.name}")


def plot_heatmap(results_df: pd.DataFrame, output_path: Path) -> None:
    """Create a heatmap of all metrics for each pair."""
    metrics_cols = ["PSNR", "SSIM", "MSE", "NRMSE", "NCC", "MI"]
    labels = [f"G{r['sample_group']}-{r['round_name']}→B" for _, r in results_df.iterrows()]

    # Normalize each metric to [0,1] for heatmap
    data = results_df[metrics_cols].values.astype(float)
    data_normalized = np.zeros_like(data)
    for j in range(data.shape[1]):
        col = data[:, j]
        rng = col.max() - col.min()
        if rng > 1e-10:
            data_normalized[:, j] = (col - col.min()) / rng
        else:
            data_normalized[:, j] = 0.5

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5 + 2)))
    im = ax.imshow(data_normalized, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(range(len(metrics_cols)))
    ax.set_xticklabels(metrics_cols, fontsize=11)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    # Annotate with actual values
    for i in range(len(labels)):
        for j in range(len(metrics_cols)):
            val = data[i, j]
            text = f"{val:.3f}" if abs(val) < 1000 else f"{val:.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if data_normalized[i, j] > 0.6 else "black")

    ax.set_title("Registration Quality Heatmap (normalized)", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Normalized value")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved heatmap: {output_path.name}")


def plot_difference_maps(
    pair: ImagePair,
    registered: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Path,
) -> None:
    """Create side-by-side difference visualization."""
    reg_2d, gt_2d = align_images_for_comparison(registered, ground_truth)
    diff = np.abs(reg_2d - gt_2d)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    title = f"Group {pair.sample_group} - Round {pair.round_name}→B"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Source (unregistered)
    src_img = imread(str(pair.source_path))
    src_2d = to_2d_gray(src_img).astype(np.float64)
    axes[0].imshow(src_2d, cmap="gray")
    axes[0].set_title("Source (Unregistered)", fontsize=11)
    axes[0].axis("off")

    # Registered by our pipeline
    axes[1].imshow(reg_2d, cmap="gray")
    axes[1].set_title("Our Registration", fontsize=11)
    axes[1].axis("off")

    # Ground truth
    axes[2].imshow(gt_2d, cmap="gray")
    axes[2].set_title("Ground Truth", fontsize=11)
    axes[2].axis("off")

    # Difference map
    im = axes[3].imshow(diff, cmap="hot")
    axes[3].set_title("Difference (|Ours - GT|)", fontsize=11)
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], shrink=0.8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_timing_chart(results_df: pd.DataFrame, output_path: Path) -> None:
    """Create a bar chart showing registration time per pair."""
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [f"G{r['sample_group']}-{r['round_name']}→B" for _, r in results_df.iterrows()]
    times = results_df["registration_time_sec"].values

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(times)))
    bars = ax.bar(range(len(times)), times, color=colors, edgecolor="gray", linewidth=0.5)

    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{t:.1f}s",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Registration Time per Image Pair", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add average line
    avg_time = times.mean()
    ax.axhline(avg_time, color="red", linestyle="--", alpha=0.7, label=f"Average: {avg_time:.1f}s")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved timing chart: {output_path.name}")


def plot_group_comparison(results_df: pd.DataFrame, output_path: Path) -> None:
    """Create grouped bar chart comparing metrics across sample groups."""
    metrics = ["PSNR", "SSIM", "NCC"]
    groups = sorted(results_df["sample_group"].unique())
    n_groups = len(groups)

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    fig.suptitle("Metrics Comparison by Sample Group", fontsize=15, fontweight="bold")

    group_colors = plt.cm.tab10(np.linspace(0, 0.3, n_groups))

    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        x = np.arange(len(ROUNDS))
        width = 0.8 / n_groups

        for gi, group in enumerate(groups):
            group_data = results_df[results_df["sample_group"] == group]
            vals = []
            for rnd in ROUNDS:
                row = group_data[group_data["round_name"] == rnd]
                vals.append(row[metric].values[0] if len(row) > 0 else 0)
            bars = ax.bar(x + gi * width, vals, width, label=f"Group {group}",
                         color=group_colors[gi], edgecolor="gray", linewidth=0.5)

        ax.set_xticks(x + width * (n_groups - 1) / 2)
        ax.set_xticklabels([f"{r}→B" for r in ROUNDS])
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved group comparison: {output_path.name}")


# ── Main benchmark runner ─────────────────────────────────────────────────────
def _parse_int_list(value: str) -> list[int]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.")
    try:
        return [int(item) for item in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_round_list(value: str) -> list[str]:
    rounds = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not rounds:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of round names.")
    return rounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the cell-registration benchmark.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-groups", type=_parse_int_list, default=None, help="Comma-separated sample groups.")
    parser.add_argument("--rounds", type=_parse_round_list, default=None, help="Comma-separated rounds, e.g. A,C,D,E.")
    parser.add_argument("--position-weight", type=float, default=DEFAULT_REG_PARAMS["position_weight"])
    parser.add_argument("--distance-threshold", type=float, default=DEFAULT_REG_PARAMS["distance_threshold"])
    parser.add_argument("--spatial-window-size", type=float, default=DEFAULT_REG_PARAMS["spatial_window_size"])
    parser.add_argument("--top-k", type=int, default=DEFAULT_REG_PARAMS["top_k"])
    parser.add_argument("--cellpose-diameter", type=float, default=DEFAULT_REG_PARAMS["cellpose_diameter"])
    parser.add_argument(
        "--cellpose-flow-threshold",
        type=float,
        default=DEFAULT_REG_PARAMS["cellpose_flow_threshold"],
    )
    parser.add_argument(
        "--cellpose-cellprob-threshold",
        type=float,
        default=DEFAULT_REG_PARAMS["cellpose_cellprob_threshold"],
    )
    parser.add_argument("--cellpose-min-size", type=int, default=DEFAULT_REG_PARAMS["cellpose_min_size"])
    parser.add_argument("--min-cells-for-two-stage", type=int, default=DEFAULT_REG_PARAMS["min_cells_for_two_stage"])
    parser.add_argument("--coarse-top-k", type=int, default=DEFAULT_REG_PARAMS["coarse_top_k"])
    parser.add_argument(
        "--coarse-distance-threshold",
        type=float,
        default=DEFAULT_REG_PARAMS["coarse_distance_threshold"],
    )
    parser.add_argument("--neighbor-k", type=int, default=DEFAULT_REG_PARAMS["neighbor_k"])
    parser.add_argument("--neighbor-weight", type=float, default=DEFAULT_REG_PARAMS["neighbor_weight"])
    parser.add_argument("--landmark-weight", type=float, default=DEFAULT_REG_PARAMS["landmark_weight"])
    parser.add_argument("--max-theta-deg", type=float, default=DEFAULT_REG_PARAMS["max_theta_deg"])
    parser.add_argument("--max-translation", type=float, default=DEFAULT_REG_PARAMS["max_translation"])
    parser.add_argument("--max-scale-change", type=float, default=DEFAULT_REG_PARAMS["max_scale_change"])
    parser.add_argument(
        "--guided-min-position-weight",
        type=float,
        default=DEFAULT_REG_PARAMS["guided_min_position_weight"],
    )
    parser.add_argument(
        "--guided-spatial-window-cap",
        type=float,
        default=DEFAULT_REG_PARAMS["guided_spatial_window_cap"],
        help="Cap (pixels) for guided rematch spatial window. <=0 disables capping.",
    )
    parser.add_argument(
        "--validation-min-confidence",
        type=float,
        default=DEFAULT_REG_PARAMS["validation_min_confidence"],
    )
    parser.add_argument(
        "--validation-max-feature-diff",
        type=float,
        default=DEFAULT_REG_PARAMS["validation_max_feature_diff"],
    )
    parser.add_argument("--prefer-affine", action=argparse.BooleanOptionalAction, default=DEFAULT_REG_PARAMS["prefer_affine"])
    parser.add_argument("--ransac-max-trials", type=int, default=DEFAULT_REG_PARAMS["ransac_max_trials"])
    parser.add_argument("--ransac-residual-threshold", type=float, default=DEFAULT_REG_PARAMS["ransac_residual_threshold"])
    parser.add_argument(
        "--similarity-residual-threshold",
        type=float,
        default=DEFAULT_REG_PARAMS["similarity_residual_threshold"],
    )
    parser.add_argument("--fft-upsample-factor", type=int, default=DEFAULT_REG_PARAMS["fft_upsample_factor"])
    parser.add_argument("--fft-max-shift", type=float, default=DEFAULT_REG_PARAMS["fft_max_shift"])
    parser.add_argument("--fft-crop-ratio", type=float, default=DEFAULT_REG_PARAMS["fft_crop_ratio"])
    parser.add_argument("--fft-max-iterations", type=int, default=DEFAULT_REG_PARAMS["fft_max_iterations"])
    return parser.parse_args()


def main():
    global OUTPUT_DIR, SAMPLE_GROUPS, ROUNDS, REG_PARAMS
    args = parse_args()
    OUTPUT_DIR = args.output_dir.resolve()
    SAMPLE_GROUPS = args.sample_groups or [1, 2, 3]
    ROUNDS = args.rounds or ["A", "C", "D", "E"]
    REG_PARAMS = {
        "cellpose_diameter": args.cellpose_diameter,
        "cellpose_flow_threshold": args.cellpose_flow_threshold,
        "cellpose_cellprob_threshold": args.cellpose_cellprob_threshold,
        "cellpose_min_size": args.cellpose_min_size,
        "position_weight": args.position_weight,
        "distance_threshold": args.distance_threshold,
        "spatial_window_size": args.spatial_window_size,
        "top_k": args.top_k,
        "min_cells_for_two_stage": args.min_cells_for_two_stage,
        "coarse_top_k": args.coarse_top_k,
        "coarse_distance_threshold": args.coarse_distance_threshold,
        "neighbor_k": args.neighbor_k,
        "neighbor_weight": args.neighbor_weight,
        "landmark_weight": args.landmark_weight,
        "max_theta_deg": args.max_theta_deg,
        "max_translation": args.max_translation,
        "max_scale_change": args.max_scale_change,
        "guided_min_position_weight": args.guided_min_position_weight,
        "guided_spatial_window_cap": args.guided_spatial_window_cap,
        "validation_min_confidence": args.validation_min_confidence,
        "validation_max_feature_diff": args.validation_max_feature_diff,
        "prefer_affine": args.prefer_affine,
        "ransac_max_trials": args.ransac_max_trials,
        "ransac_residual_threshold": args.ransac_residual_threshold,
        "similarity_residual_threshold": args.similarity_residual_threshold,
        "fft_upsample_factor": args.fft_upsample_factor,
        "fft_max_shift": args.fft_max_shift,
        "fft_crop_ratio": args.fft_crop_ratio,
        "fft_max_iterations": args.fft_max_iterations,
    }
    print("=" * 70)
    print("  Cell Registration Benchmark")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    params_path = OUTPUT_DIR / "benchmark_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump({"sample_groups": SAMPLE_GROUPS, "rounds": ROUNDS, "reg_params": REG_PARAMS}, f, indent=2)
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Params saved to: {params_path.name}")
    print(f"  REG_PARAMS: {json.dumps(REG_PARAMS, indent=2)}")

    # Discover pairs
    print("\n[1/5] Discovering image pairs...")
    pairs = discover_image_pairs()
    print(f"  Found {len(pairs)} registration pairs")
    for p in pairs:
        print(f"    Group {p.sample_group}: {p.round_name}→B  |  src: {p.source_path.name}")

    if not pairs:
        print("  ✗ No image pairs found. Check directory structure.")
        return

    # Initialize segmenter (once, to avoid re-loading model)
    print("\n[2/5] Initializing Cellpose-SAM segmenter (GPU=True)...")
    cellpose_config = CellposeConfig(
        gpu=True,
        pretrained_model="cpsam",
        diameter=REG_PARAMS["cellpose_diameter"],
        flow_threshold=REG_PARAMS["cellpose_flow_threshold"],
        cellprob_threshold=REG_PARAMS["cellpose_cellprob_threshold"],
        min_size=REG_PARAMS["cellpose_min_size"],
    )
    segmenter = CellposeSegmenter(cellpose_config)
    print("  ✓ Segmenter ready")

    # Run registrations
    print("\n[3/5] Running registrations...")
    results: List[BenchmarkResult] = []

    for i, pair in enumerate(pairs):
        label = f"Group {pair.sample_group} - {pair.round_name}→B"
        print(f"\n  [{i+1}/{len(pairs)}] {label}")
        print(f"    Source:    {pair.source_path.name}")
        print(f"    Reference: {pair.reference_path.name}")

        # Load images
        source_img = imread(str(pair.source_path))
        reference_img = imread(str(pair.reference_path))
        print(f"    Image shapes: source={source_img.shape}, reference={reference_img.shape}")

        # Run registration (timed)
        registered, elapsed = run_registration(source_img, reference_img, segmenter)
        print(f"    ⏱ Registration time: {elapsed:.2f}s")

        # Save registered image
        reg_output_path = OUTPUT_DIR / f"registered_G{pair.sample_group}_{pair.round_name}_to_B.tif"
        imwrite(str(reg_output_path), registered)

        # Load ground truth and compute metrics
        gt_img = imread(str(pair.ground_truth_path))
        reg_aligned, gt_aligned = align_images_for_comparison(registered, gt_img)
        metrics = compute_all_metrics(reg_aligned, gt_aligned)

        print(f"    Metrics: PSNR={metrics['PSNR']:.2f} | SSIM={metrics['SSIM']:.4f} | NCC={metrics['NCC']:.4f}")

        results.append(BenchmarkResult(
            sample_group=pair.sample_group,
            round_name=pair.round_name,
            registration_time_sec=elapsed,
            metrics=metrics,
            registered_image=registered,
        ))

    # Build results DataFrame
    print("\n[4/5] Generating visualizations and reports...")
    rows = []
    for r in results:
        row = {
            "sample_group": r.sample_group,
            "round_name": r.round_name,
            "registration_time_sec": r.registration_time_sec,
        }
        row.update(r.metrics)
        rows.append(row)
    results_df = pd.DataFrame(rows)

    # Save CSV
    csv_path = OUTPUT_DIR / "benchmark_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved results CSV: {csv_path.name}")

    # Generate visualizations
    plot_metrics_bar_chart(results_df, OUTPUT_DIR / "metrics_bar_chart.png")
    plot_heatmap(results_df, OUTPUT_DIR / "metrics_heatmap.png")
    plot_timing_chart(results_df, OUTPUT_DIR / "timing_chart.png")
    plot_group_comparison(results_df, OUTPUT_DIR / "group_comparison.png")

    # Generate difference maps for each pair
    print("\n  Generating difference maps...")
    for pair, result in zip(pairs, results):
        gt_img = imread(str(pair.ground_truth_path))
        diff_path = OUTPUT_DIR / f"diff_G{pair.sample_group}_{pair.round_name}_to_B.png"
        plot_difference_maps(pair, result.registered_image, gt_img, diff_path)

    # Summary report
    print("\n[5/5] Summary")
    print("=" * 70)
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
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
