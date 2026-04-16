"""Test F1 with current widget pipeline (TPS + consensus filter)."""
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
    CellFeaturesConfig, CellposeConfig, CellposeSegmenter, MatchingConfig,
    MIN_MATCHES_FOR_REFINEMENT, compute_cell_features, compute_match_residuals,
    estimate_rigid_transform_from_matches, estimate_rigid_transform_from_matches_ransac,
    greedy_match_cells, rigid_transform_to_affine, two_stage_match_cells,
)
from napari_cell_registration.core.matching import (
    apply_transform_to_features, _filter_candidate_matches_by_hard_constraints,
)
from napari_cell_registration.core.point_registration import (
    fit_tps_from_matches, warp_image_with_tps,
)
from cell_registration.evaluation import compute_prewarped_registration_metrics

UNREG_ROOT = Path(r"F:\programme\nxy\Cell registration\benchmark\unregistration")
FIXED_PATH = UNREG_ROOT / "C2-1" / "fixed" / "C2-C-1.tif"
MOVING_PATH = UNREG_ROOT / "C2-1" / "moving" / "C2-A-1.tif"

# Widget params
TOP_K = 320
MAX_DIST = 100
POSITION_WEIGHT = 1.0
RANSAC_RESIDUAL_THRESHOLD = 2.0
RANSAC_MAX_TRIALS = 1000


def main():
    print("Segmenting...")
    segmenter = CellposeSegmenter(CellposeConfig(
        gpu=True, pretrained_model="cpsam", diameter=15,
        flow_threshold=-2.0, cellprob_threshold=1.0, min_size=5,
    ))
    img1 = imread(str(FIXED_PATH))
    mask1, _, _ = segmenter.segment_array(np.asarray(img1))
    feats1 = compute_cell_features(mask1, CellFeaturesConfig(topology_neighbor_k=5))
    print(f"  Fixed: {len(feats1)} cells")

    img2 = imread(str(MOVING_PATH))
    mask2, _, _ = segmenter.segment_array(np.asarray(img2))
    feats2 = compute_cell_features(mask2, CellFeaturesConfig(topology_neighbor_k=5))
    print(f"  Moving: {len(feats2)} cells")

    # Step 1: two_stage match
    print("\nStep 1: two_stage match...")
    match_result = two_stage_match_cells(
        feats1, feats2, mask1.shape,
        feature_weight=1.0, topology_weight=0.0, position_weight=POSITION_WEIGHT,
        top_k=TOP_K, distance_threshold=None, spatial_window_size=float(MAX_DIST),
        min_cells_for_two_stage=10, coarse_top_k=max(24, TOP_K),
        coarse_distance_threshold=2.0, coarse_matching_mode="morphology_guided",
        coarse_allow_scale=False, coarse_prefer_affine=False,
        coarse_residual_threshold=max(5.0, RANSAC_RESIDUAL_THRESHOLD * 2.0),
        coarse_max_trials=min(max(RANSAC_MAX_TRIALS, 200), 2000),
    )
    matches = match_result.matches.copy()
    print(f"  Initial matches: {len(matches)}")
    print(f"  Coarse accepted: {match_result.coarse_transform_accepted}")

    # Step 2: RANSAC
    print("\nStep 2: RANSAC...")
    transform, inlier_mask = estimate_rigid_transform_from_matches_ransac(
        feats1, feats2, matches,
        max_trials=RANSAC_MAX_TRIALS, residual_threshold=RANSAC_RESIDUAL_THRESHOLD,
        min_inliers=MIN_MATCHES_FOR_REFINEMENT,
    )
    inlier_count = int(inlier_mask.sum())
    print(f"  RANSAC inliers: {inlier_count}/{len(matches)}")

    # Model consistency filter
    all_residuals = compute_match_residuals(feats1, feats2, matches, transform)
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
                max_trials=RANSAC_MAX_TRIALS, residual_threshold=RANSAC_RESIDUAL_THRESHOLD,
                min_inliers=MIN_MATCHES_FOR_REFINEMENT,
            )
            print(f"  Model consistency: kept {keep_count}/{len(all_residuals)}")

    # Step 3: Guided rematch
    print("\nStep 3: Guided rematch...")
    initial_affine = rigid_transform_to_affine(transform)
    aligned_feats2 = apply_transform_to_features(feats2, initial_affine, mask1.shape)
    rematch_window = max(10.0, float(MAX_DIST) * 0.6)
    rematch_config = MatchingConfig(
        feature_weight=1.0, topology_weight=0.0, position_weight=POSITION_WEIGHT,
        top_k=TOP_K, distance_threshold=None, spatial_window_size=rematch_window,
    )
    rematch_matches = greedy_match_cells(feats1, aligned_feats2, rematch_config, image_shape=mask1.shape, coverage_patch_grid=4)
    if len(rematch_matches) >= MIN_MATCHES_FOR_REFINEMENT:
        transform = estimate_rigid_transform_from_matches(feats1, feats2, rematch_matches)
        matches = rematch_matches
    print(f"  After rematch: {len(matches)}")

    # Step 4: Orientation filter (5°, eccentricity 0.15)
    n_before = len(matches)
    matches = _filter_candidate_matches_by_hard_constraints(
        matches, max_area_ratio=None, max_aspect_ratio_ratio=None,
        max_orientation_diff_deg=5.0, min_orientation_eccentricity=0.15,
    )
    if len(matches) < n_before:
        print(f"  Orientation filter: {n_before} -> {len(matches)}")
        if len(matches) >= MIN_MATCHES_FOR_REFINEMENT:
            transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)

    # Step 5: Displacement consensus filter
    print("\nStep 5: Displacement consensus filter...")
    n_before_disp = len(matches)
    pts_f = feats1.iloc[matches["idx1"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    pts_m = feats2.iloc[matches["idx2"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    disp_vectors = pts_f - pts_m

    h, w = mask1.shape[:2]
    grid = 4
    px = np.clip(np.floor(pts_f[:, 0] * grid / max(w, 1)).astype(int), 0, grid - 1)
    py = np.clip(np.floor(pts_f[:, 1] * grid / max(h, 1)).astype(int), 0, grid - 1)
    patch_ids = py * grid + px

    keep_mask_disp = np.ones(n_before_disp, dtype=bool)
    angle_threshold_deg = 5.0
    cos_thresh = np.cos(np.radians(angle_threshold_deg))
    for pid in range(grid * grid):
        in_patch = patch_ids == pid
        n_in = int(in_patch.sum())
        if n_in < 3:
            continue
        patch_disp = disp_vectors[in_patch]
        patch_indices = np.where(in_patch)[0]
        patch_norms = np.linalg.norm(patch_disp, axis=1)

        votes = np.zeros(n_in, dtype=int)
        for a in range(n_in):
            if patch_norms[a] < 1.0:
                continue
            for b in range(n_in):
                if patch_norms[b] < 1.0:
                    continue
                cos_ab = np.dot(patch_disp[a], patch_disp[b]) / (patch_norms[a] * patch_norms[b] + 1e-8)
                if cos_ab >= cos_thresh:
                    votes[a] += 1

        if votes.max() < 2:
            continue
        consensus_idx = int(np.argmax(votes))
        consensus_disp = patch_disp[consensus_idx]
        consensus_norm = patch_norms[consensus_idx]

        consensus_members = []
        for k in range(n_in):
            if patch_norms[k] < 1.0:
                keep_mask_disp[patch_indices[k]] = False
                continue
            cos_val = np.dot(patch_disp[k], consensus_disp) / (patch_norms[k] * consensus_norm + 1e-8)
            if cos_val < cos_thresh:
                keep_mask_disp[patch_indices[k]] = False
            else:
                consensus_members.append(k)

        if len(consensus_members) >= 3:
            member_lengths = np.array([patch_norms[k] for k in consensus_members])
            mean_len = float(np.mean(member_lengths))
            for k in consensus_members:
                if abs(patch_norms[k] - mean_len) > 5.0:
                    keep_mask_disp[patch_indices[k]] = False

    n_after_disp = int(keep_mask_disp.sum())
    if n_after_disp >= MIN_MATCHES_FOR_REFINEMENT and n_after_disp < n_before_disp:
        matches = matches.loc[keep_mask_disp].reset_index(drop=True)
        transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)
    print(f"  Consensus filter: {n_before_disp} -> {len(matches)}")

    # Step 6: TPS warp
    print(f"\nStep 6: TPS warp with {len(matches)} control points...")
    affine_transform = rigid_transform_to_affine(transform)
    pts_fixed_xy = feats1.iloc[matches["idx1"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    pts_moving_xy = feats2.iloc[matches["idx2"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)

    tps = fit_tps_from_matches(
        pts_fixed_xy, pts_moving_xy, output_shape=mask1.shape[:2],
        rigid_transform=affine_transform, regularization=1e-3,
        n_boundary_per_side=4, add_boundary_anchors_flag=True,
    )
    mask2_warped = warp_image_with_tps(mask2.astype(np.int32), tps, mask1.shape[:2], order=0)
    mask2_registered = np.rint(mask2_warped).astype(np.int32)

    # Step 7: F1
    print("\nStep 7: F1 evaluation...")
    metrics = compute_prewarped_registration_metrics(mask2_registered, mask1, instance_iou_threshold=0.3)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"  F1:       {metrics['match_f1']:.4f}")
    print(f"  Prec:     {metrics['match_precision']:.4f}")
    print(f"  Recall:   {metrics['match_recall']:.4f}")
    print(f"  Matched:  {metrics['matched_cells']} / {metrics['eligible_fixed_cells']} fixed, {metrics['eligible_moving_cells']} moving")
    print(f"  Mean IoU: {metrics['matched_mean_iou']:.4f}")


if __name__ == "__main__":
    main()
