"""
Batch benchmark: runs the CURRENT widget pipeline (TPS + consensus filter + Hu moments)
on all image pairs in the unregistration directory.

Usage:
    conda activate cell_registration
    python benchmark/run_tps_batch.py
"""
from __future__ import annotations

import functools
import json
import os
import sys
import time
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
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "napari-cell-registration" / "src"))

import numpy as np
import pandas as pd
from tifffile import imread, imwrite

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
)
from cell_registration.evaluation import compute_prewarped_registration_metrics

# ── Paths ──
UNREG_ROOT = PROJECT_ROOT / "benchmark" / "unregistration"
OUTPUT_ROOT = PROJECT_ROOT / "benchmark" / "results" / "full_registration_tps"

# ── Widget parameters (exact copy from _widget.py) ──
TOP_K = 320
MAX_DIST = 100
POSITION_WEIGHT = 1.0
RANSAC_RESIDUAL_THRESHOLD = 2.0
RANSAC_MAX_TRIALS = 1000
PATCH_GRID = 4
REMATCH_WINDOW_FACTOR = 0.6
ORIENTATION_MAX_DEG = 5.0
ORIENTATION_MIN_ECC = 0.15
CONSENSUS_ANGLE_DEG = 5.0
CONSENSUS_LENGTH_TOL = 5.0
IOU_THRESHOLD = 0.3

CELLPOSE_CFG = CellposeConfig(
    gpu=True, pretrained_model="cpsam", diameter=15,
    flow_threshold=-2.0, cellprob_threshold=1.0, min_size=5,
)
FEATURE_CFG = CellFeaturesConfig(topology_neighbor_k=5)


def discover_pairs():
    """Return list of (case_id, fixed_path, moving_path) for all pairs."""
    pairs = []
    for case_dir in sorted(UNREG_ROOT.iterdir()):
        if not case_dir.is_dir():
            continue
        fixed_dir = case_dir / "fixed"
        moving_dir = case_dir / "moving"
        if not fixed_dir.exists() or not moving_dir.exists():
            continue
        fixed_files = sorted(fixed_dir.glob("*.tif"))
        moving_files = sorted(moving_dir.glob("*.tif"))
        if len(fixed_files) != 1:
            continue
        fixed_path = fixed_files[0]
        for mov_path in moving_files:
            pairs.append((case_dir.name, fixed_path, mov_path))
    return pairs


def run_widget_pipeline(img1, mask1, feats1, img2, mask2, feats2):
    """Run the exact widget pipeline. Returns (registered_mask, matches, diagnostics)."""
    diag = {}

    # Step 1: two_stage match
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
    diag["initial_matches"] = len(matches)
    diag["coarse_accepted"] = bool(match_result.coarse_transform_accepted)

    # Step 2: RANSAC
    transform, inlier_mask = estimate_rigid_transform_from_matches_ransac(
        feats1, feats2, matches,
        max_trials=RANSAC_MAX_TRIALS, residual_threshold=RANSAC_RESIDUAL_THRESHOLD,
        min_inliers=MIN_MATCHES_FOR_REFINEMENT,
    )
    inlier_count = int(inlier_mask.sum())
    diag["ransac_inliers"] = inlier_count

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
            diag["model_consistency_kept"] = keep_count

    # Step 3: Guided rematch
    if len(matches) >= MIN_MATCHES_FOR_REFINEMENT:
        initial_affine = rigid_transform_to_affine(transform)
        aligned_feats2 = apply_transform_to_features(feats2, initial_affine, mask1.shape)
        rematch_window = max(10.0, float(MAX_DIST) * REMATCH_WINDOW_FACTOR)
        rematch_config = MatchingConfig(
            feature_weight=1.0, topology_weight=0.0, position_weight=POSITION_WEIGHT,
            top_k=TOP_K, distance_threshold=None, spatial_window_size=rematch_window,
        )
        rematch_matches = greedy_match_cells(
            feats1, aligned_feats2, rematch_config,
            image_shape=mask1.shape, coverage_patch_grid=PATCH_GRID,
        )
        if len(rematch_matches) >= MIN_MATCHES_FOR_REFINEMENT:
            transform = estimate_rigid_transform_from_matches(feats1, feats2, rematch_matches)
            matches = rematch_matches
    diag["after_rematch"] = len(matches)

    # Step 4: Orientation filter
    n_before_orient = len(matches)
    matches = _filter_candidate_matches_by_hard_constraints(
        matches, max_area_ratio=None, max_aspect_ratio_ratio=None,
        max_orientation_diff_deg=ORIENTATION_MAX_DEG,
        min_orientation_eccentricity=ORIENTATION_MIN_ECC,
    )
    if len(matches) < n_before_orient and len(matches) >= MIN_MATCHES_FOR_REFINEMENT:
        transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)
    diag["after_orientation"] = len(matches)

    # Step 5: Displacement consensus filter
    n_before_disp = len(matches)
    if n_before_disp >= MIN_MATCHES_FOR_REFINEMENT:
        pts_f = feats1.iloc[matches["idx1"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
        pts_m = feats2.iloc[matches["idx2"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
        disp_vectors = pts_f - pts_m
        h, w = mask1.shape[:2]
        grid = PATCH_GRID
        px = np.clip(np.floor(pts_f[:, 0] * grid / max(w, 1)).astype(int), 0, grid - 1)
        py = np.clip(np.floor(pts_f[:, 1] * grid / max(h, 1)).astype(int), 0, grid - 1)
        patch_ids = py * grid + px

        keep_mask_c = np.ones(n_before_disp, dtype=bool)
        cos_thresh = np.cos(np.radians(CONSENSUS_ANGLE_DEG))
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
                    keep_mask_c[patch_indices[k]] = False
                    continue
                cos_val = np.dot(patch_disp[k], consensus_disp) / (patch_norms[k] * consensus_norm + 1e-8)
                if cos_val < cos_thresh:
                    keep_mask_c[patch_indices[k]] = False
                else:
                    consensus_members.append(k)

            if len(consensus_members) >= 3:
                member_lengths = np.array([patch_norms[k] for k in consensus_members])
                mean_len = float(np.mean(member_lengths))
                for k in consensus_members:
                    if abs(patch_norms[k] - mean_len) > CONSENSUS_LENGTH_TOL:
                        keep_mask_c[patch_indices[k]] = False

        n_after_disp = int(keep_mask_c.sum())
        if n_after_disp >= MIN_MATCHES_FOR_REFINEMENT and n_after_disp < n_before_disp:
            matches = matches.loc[keep_mask_c].reset_index(drop=True)
            transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)
    diag["after_consensus"] = len(matches)

    # Step 6: TPS warp
    affine_transform = rigid_transform_to_affine(transform)
    pts_fixed_xy = feats1.iloc[matches["idx1"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    pts_moving_xy = feats2.iloc[matches["idx2"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)

    tps = fit_tps_from_matches(
        pts_fixed_xy, pts_moving_xy, output_shape=mask1.shape[:2],
        rigid_transform=affine_transform, regularization=1e-3,
        n_boundary_per_side=4, add_boundary_anchors_flag=True,
    )
    diag["tps_control_points"] = len(pts_fixed_xy)

    mask2_warped = warp_image_with_tps(mask2.astype(np.int32), tps, mask1.shape[:2], order=0)
    mask2_registered = np.rint(mask2_warped).astype(np.int32)

    return mask2_registered, matches, diag


def main():
    pairs = discover_pairs()
    print(f"Discovered {len(pairs)} pairs across {len(set(p[0] for p in pairs))} cases")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    segmenter = CellposeSegmenter(CELLPOSE_CFG)

    # Cache segmentation per image path
    seg_cache = {}
    all_results = []
    failures = []

    for i, (case_id, fixed_path, moving_path) in enumerate(pairs):
        pair_name = f"{moving_path.stem}_to_{fixed_path.stem}"
        pair_dir = OUTPUT_ROOT / case_id / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i+1}/{len(pairs)}] {case_id}: {moving_path.name} → {fixed_path.name}")

        t0 = time.perf_counter()
        try:
            # Segment (with cache)
            if str(fixed_path) not in seg_cache:
                img1 = imread(str(fixed_path))
                mask1, _, _ = segmenter.segment_array(np.asarray(img1))
                feats1 = compute_cell_features(mask1, FEATURE_CFG)
                seg_cache[str(fixed_path)] = (img1, mask1, feats1)
            img1, mask1, feats1 = seg_cache[str(fixed_path)]

            if str(moving_path) not in seg_cache:
                img2 = imread(str(moving_path))
                mask2, _, _ = segmenter.segment_array(np.asarray(img2))
                feats2 = compute_cell_features(mask2, FEATURE_CFG)
                seg_cache[str(moving_path)] = (img2, mask2, feats2)
            img2, mask2, feats2 = seg_cache[str(moving_path)]

            print(f"  Fixed: {len(feats1)} cells, Moving: {len(feats2)} cells")

            # Run pipeline
            mask2_reg, matches, diag = run_widget_pipeline(img1, mask1, feats1, img2, mask2, feats2)

            # Evaluate F1
            metrics = compute_prewarped_registration_metrics(mask2_reg, mask1, instance_iou_threshold=IOU_THRESHOLD)

            elapsed = time.perf_counter() - t0

            print(f"  F1={metrics['match_f1']:.4f}  Prec={metrics['match_precision']:.4f}  "
                  f"Recall={metrics['match_recall']:.4f}  Matched={metrics['matched_cells']}/{metrics['eligible_fixed_cells']}  "
                  f"TPS pts={diag['tps_control_points']}  {elapsed:.1f}s")

            # Save artifacts
            imwrite(str(pair_dir / "registered_mask.tif"), mask2_reg, compression="zlib")
            matches.to_csv(str(pair_dir / "registration_matches.csv"), index=False)
            feats1.to_csv(str(pair_dir / "fixed_features.csv"), index=False)
            feats2.to_csv(str(pair_dir / "moving_features.csv"), index=False)

            diag_full = {
                "case_id": case_id,
                "fixed_name": fixed_path.name,
                "moving_name": moving_path.name,
                "fixed_cells": len(feats1),
                "moving_cells": len(feats2),
                "pipeline": diag,
                "metrics": {k: (v if not isinstance(v, float) or np.isfinite(v) else None) for k, v in metrics.items()},
                "elapsed_seconds": round(elapsed, 2),
            }
            with open(pair_dir / "diagnostics.json", "w") as f:
                json.dump(diag_full, f, indent=2, default=str)

            row = {
                "case_id": case_id,
                "fixed_name": fixed_path.name,
                "moving_name": moving_path.name,
                "status": "success",
                "fixed_cells": len(feats1),
                "moving_cells": len(feats2),
                "initial_matches": diag["initial_matches"],
                "ransac_inliers": diag["ransac_inliers"],
                "after_rematch": diag["after_rematch"],
                "after_orientation": diag["after_orientation"],
                "after_consensus": diag["after_consensus"],
                "tps_control_points": diag["tps_control_points"],
                "elapsed_seconds": round(elapsed, 2),
                **{k: v for k, v in metrics.items()},
            }
            all_results.append(row)

        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  FAILED: {e}")
            failures.append({"case_id": case_id, "pair": pair_name, "error": str(e)})
            all_results.append({
                "case_id": case_id,
                "fixed_name": fixed_path.name,
                "moving_name": moving_path.name,
                "status": "failed",
                "error": str(e),
                "elapsed_seconds": round(elapsed, 2),
            })

    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(OUTPUT_ROOT / "summary.csv", index=False)

    with open(OUTPUT_ROOT / "failures.json", "w") as f:
        json.dump(failures, f, indent=2)

    # Print overall summary
    success_df = summary_df[summary_df["status"] == "success"]
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Success:     {len(success_df)}")
    print(f"  Failed:      {len(failures)}")
    if len(success_df) > 0:
        print(f"  Mean F1:     {success_df['match_f1'].mean():.4f}")
        print(f"  Median F1:   {success_df['match_f1'].median():.4f}")
        print(f"  Min F1:      {success_df['match_f1'].min():.4f}")
        print(f"  Max F1:      {success_df['match_f1'].max():.4f}")

    # Per-case F1
    case_f1 = success_df.groupby("case_id")["match_f1"].mean().sort_values()
    case_f1.to_csv(OUTPUT_ROOT / "f1_by_case.csv")
    print("\nPer-case mean F1:")
    for cid, f1 in case_f1.items():
        print(f"  {cid}: {f1:.4f}")

    # Save params
    params = {
        "top_k": TOP_K, "max_dist": MAX_DIST, "position_weight": POSITION_WEIGHT,
        "ransac_residual_threshold": RANSAC_RESIDUAL_THRESHOLD, "ransac_max_trials": RANSAC_MAX_TRIALS,
        "patch_grid": PATCH_GRID, "rematch_window_factor": REMATCH_WINDOW_FACTOR,
        "orientation_max_deg": ORIENTATION_MAX_DEG, "orientation_min_ecc": ORIENTATION_MIN_ECC,
        "consensus_angle_deg": CONSENSUS_ANGLE_DEG, "consensus_length_tol": CONSENSUS_LENGTH_TOL,
        "iou_threshold": IOU_THRESHOLD,
        "cellpose": {"diameter": 15, "flow_threshold": -2.0, "cellprob_threshold": 1.0, "min_size": 5},
    }
    with open(OUTPUT_ROOT / "params.json", "w") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()
