"""
Test how top_k affects registration F1 score using TPS warping.

Mimics the napari widget pipeline:
  1. Segmentation (cached)
  2. two_stage_match_cells (patch-balanced)
  3. RANSAC transform estimation
  4. Guided rematch under estimated transform
  5. TPS warp (instead of rigid)
  6. Object-level F1 evaluation
"""
from __future__ import annotations

import os, sys, time, functools
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_tl = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_tl):
    os.environ["PATH"] = _tl + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_tl)
try:
    import torch
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "napari-cell-registration" / "src"))

import numpy as np
import pandas as pd
from tifffile import imread
print = functools.partial(print, flush=True)

from napari_cell_registration.core import (
    CellFeaturesConfig,
    CellposeConfig,
    CellposeSegmenter,
    MatchingConfig,
    MIN_MATCHES_FOR_REFINEMENT,
    compute_cell_features,
    compute_match_residuals,
    estimate_rigid_transform_from_matches,
    estimate_rigid_transform_from_matches_ransac,
    greedy_match_cells,
    rigid_transform_to_affine,
    two_stage_match_cells,
)
from napari_cell_registration.core.matching import (
    apply_transform_to_features,
    _filter_candidate_matches_by_hard_constraints,
)
from napari_cell_registration.core.point_registration import (
    fit_tps_from_matches,
    warp_image_with_tps,
    warp_image_with_transform,
)
from cell_registration.evaluation import compute_prewarped_registration_metrics

# ── Paths ──
UNREG_ROOT = Path(r"F:\programme\nxy\Cell registration\benchmark\unregistration")
FIXED_PATH = UNREG_ROOT / "C2-1" / "fixed" / "C2-C-1.tif"
MOVING_PATH = UNREG_ROOT / "C2-1" / "moving" / "C2-A-1.tif"

TOP_K_VALUES = [40, 60, 80, 100]
RANSAC_RESIDUAL_THRESHOLD = 2.0
RANSAC_MAX_TRIALS = 1000
MAX_MATCH_DISTANCE_PX = 100
POSITION_WEIGHT = 1.0


def segment_image(path: Path, segmenter: CellposeSegmenter) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    img = imread(str(path))
    mask, _, _ = segmenter.segment_array(np.asarray(img))
    feat_config = CellFeaturesConfig(topology_neighbor_k=3)
    feats = compute_cell_features(mask, feat_config)
    return img, mask, feats


def run_widget_pipeline(
    img1, mask1, feats1,
    img2, mask2, feats2,
    top_k: int,
) -> dict:
    """Mimic the napari widget registration pipeline."""
    t0 = time.perf_counter()

    # Step 1: two_stage match (patch-balanced)
    max_dist = MAX_MATCH_DISTANCE_PX
    match_result = two_stage_match_cells(
        feats1, feats2, mask1.shape,
        feature_weight=1.0,
        topology_weight=0.0,
        position_weight=POSITION_WEIGHT,
        top_k=top_k,
        distance_threshold=None,
        spatial_window_size=float(max_dist),
        min_cells_for_two_stage=10,
        coarse_top_k=max(24, top_k),
        coarse_distance_threshold=2.0,
        coarse_matching_mode="morphology_guided",
        coarse_allow_scale=False,
        coarse_prefer_affine=False,
        coarse_residual_threshold=max(5.0, RANSAC_RESIDUAL_THRESHOLD * 2.0),
        coarse_max_trials=min(max(RANSAC_MAX_TRIALS, 200), 2000),
    )
    matches = match_result.matches.copy()

    coarse_info = (
        f"accepted, shift=({match_result.coarse_offset_xy[0]:.1f}, {match_result.coarse_offset_xy[1]:.1f})"
        if match_result.coarse_transform_accepted
        else "rejected"
    )

    if matches.empty or len(matches) < 3:
        return {"top_k": top_k, "error": "too few initial matches", "match_count": len(matches)}

    # Step 2: RANSAC transform
    transform, inlier_mask = estimate_rigid_transform_from_matches_ransac(
        feats1, feats2, matches,
        max_trials=RANSAC_MAX_TRIALS,
        residual_threshold=RANSAC_RESIDUAL_THRESHOLD,
        min_inliers=MIN_MATCHES_FOR_REFINEMENT,
    )
    inlier_count = int(inlier_mask.sum())
    all_residuals = compute_match_residuals(feats1, feats2, matches, transform)

    # Model consistency filter
    if inlier_count >= MIN_MATCHES_FOR_REFINEMENT:
        inlier_residuals = all_residuals[inlier_mask]
        inlier_median = float(np.median(inlier_residuals))
        inlier_mad = float(np.median(np.abs(inlier_residuals - inlier_median)))
        robust_scale = max(1.4826 * inlier_mad, 0.5)
        model_threshold = max(RANSAC_RESIDUAL_THRESHOLD * 2.0, inlier_median + 3.0 * robust_scale)
        keep_mask = all_residuals <= model_threshold
        keep_count = int(keep_mask.sum())
        if MIN_MATCHES_FOR_REFINEMENT <= keep_count < len(matches):
            matches = matches.loc[keep_mask].reset_index(drop=True)
            transform, _ = estimate_rigid_transform_from_matches_ransac(
                feats1, feats2, matches,
                max_trials=RANSAC_MAX_TRIALS,
                residual_threshold=RANSAC_RESIDUAL_THRESHOLD,
                min_inliers=MIN_MATCHES_FOR_REFINEMENT,
            )

    if len(matches) < MIN_MATCHES_FOR_REFINEMENT:
        return {"top_k": top_k, "error": "too few RANSAC matches", "match_count": len(matches)}

    # Step 3: Guided rematch
    initial_affine = rigid_transform_to_affine(transform)
    aligned_feats2 = apply_transform_to_features(feats2, initial_affine, mask1.shape)
    rematch_window = max(10.0, float(max_dist) * 0.3)
    rematch_config = MatchingConfig(
        feature_weight=1.0,
        topology_weight=0.0,
        position_weight=POSITION_WEIGHT,
        top_k=top_k,
        distance_threshold=None,
        spatial_window_size=rematch_window,
    )
    rematch_matches = greedy_match_cells(
        feats1, aligned_feats2, rematch_config, image_shape=mask1.shape,
    )
    if len(rematch_matches) >= MIN_MATCHES_FOR_REFINEMENT:
        transform = estimate_rigid_transform_from_matches(feats1, feats2, rematch_matches)
        matches = rematch_matches

    # Step 3b: Orientation filter (<=15°)
    n_before = len(matches)
    matches = _filter_candidate_matches_by_hard_constraints(
        matches,
        max_area_ratio=None,
        max_aspect_ratio_ratio=None,
        max_orientation_diff_deg=15.0,
        min_orientation_eccentricity=0.35,
    )
    n_after = len(matches)
    orient_filtered = n_before - n_after
    if n_after >= MIN_MATCHES_FOR_REFINEMENT and n_after < n_before:
        transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)

    final_match_count = len(matches)

    # Step 4: TPS warp
    affine_transform = rigid_transform_to_affine(transform)
    pts_fixed_xy = feats1.iloc[matches["idx1"].to_numpy(dtype=int)][
        ["centroid_x", "centroid_y"]
    ].to_numpy(dtype=float)
    pts_moving_xy = feats2.iloc[matches["idx2"].to_numpy(dtype=int)][
        ["centroid_x", "centroid_y"]
    ].to_numpy(dtype=float)

    tps = fit_tps_from_matches(
        pts_fixed_xy, pts_moving_xy,
        output_shape=mask1.shape[:2],
        rigid_transform=affine_transform,
        regularization=1e-3,
        n_boundary_per_side=4,
        add_boundary_anchors_flag=True,
    )

    # Warp mask with TPS
    mask2_warped = warp_image_with_tps(
        mask2.astype(np.int32), tps, mask1.shape[:2], order=0,
    )
    mask2_registered = np.rint(mask2_warped).astype(np.int32)

    elapsed = time.perf_counter() - t0

    # Step 5: F1 evaluation (using TPS-warped mask directly)
    metrics = compute_prewarped_registration_metrics(
        mask2_registered, mask1,
        instance_iou_threshold=0.3,
    )

    return {
        "top_k": top_k,
        "match_count": final_match_count,
        "orient_filtered": orient_filtered,
        "coarse": coarse_info,
        "ransac_inliers": inlier_count,
        "f1": metrics["match_f1"],
        "precision": metrics["match_precision"],
        "recall": metrics["match_recall"],
        "matched_cells": metrics["matched_cells"],
        "eligible_moving": metrics["eligible_moving_cells"],
        "eligible_fixed": metrics["eligible_fixed_cells"],
        "mean_iou": metrics["matched_mean_iou"],
        "elapsed_s": elapsed,
    }


def main():
    print("=" * 60)
    print("Top-K vs F1 Benchmark (TPS Pipeline)")
    print("=" * 60)
    print(f"Fixed: {FIXED_PATH.name}")
    print(f"Moving: {MOVING_PATH.name}")
    print()

    # Segment once, reuse
    print("Segmenting images...")
    segmenter = CellposeSegmenter(CellposeConfig(
        gpu=True,
        pretrained_model="cpsam",
        diameter=15,
        flow_threshold=-2.0,
        cellprob_threshold=1.0,
        min_size=5,
    ))

    img1, mask1, feats1 = segment_image(FIXED_PATH, segmenter)
    print(f"  Fixed: {len(feats1)} cells")
    img2, mask2, feats2 = segment_image(MOVING_PATH, segmenter)
    print(f"  Moving: {len(feats2)} cells")
    print()

    # Run for each top_k
    results = []
    for top_k in TOP_K_VALUES:
        print(f"--- top_k = {top_k} ---")
        result = run_widget_pipeline(img1, mask1, feats1, img2, mask2, feats2, top_k)
        results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Matches: {result['match_count']} (orient filtered: {result['orient_filtered']})")
            print(f"  F1: {result['f1']:.4f}  (P={result['precision']:.4f}, R={result['recall']:.4f})")
            print(f"  Matched cells: {result['matched_cells']} / {result['eligible_fixed']} fixed, {result['eligible_moving']} moving")
            print(f"  Mean IoU: {result['mean_iou']:.4f}")
            print(f"  Time: {result['elapsed_s']:.1f}s")
        print()

    # Summary table
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'top_k':>6} | {'matches':>8} | {'F1':>7} | {'Prec':>7} | {'Recall':>7} | {'matched':>8} | {'IoU':>6} | {'time':>6}")
    print("-" * 75)
    for r in results:
        if "error" in r:
            print(f"{r['top_k']:>6} | {'ERROR':>8} | {'-':>7} | {'-':>7} | {'-':>7} | {'-':>8} | {'-':>6} | {'-':>6}")
        else:
            print(
                f"{r['top_k']:>6} | {r['match_count']:>8} | {r['f1']:>7.4f} | {r['precision']:>7.4f} | "
                f"{r['recall']:>7.4f} | {r['matched_cells']:>8} | {r['mean_iou']:>6.4f} | {r['elapsed_s']:>5.1f}s"
            )


if __name__ == "__main__":
    main()
