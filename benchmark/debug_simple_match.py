"""
Minimal pipeline for rigid_0049:
  1. Segment & extract features (existing code)
  2. Global greedy match (no patches, no spatial window, morphology only)
  3. RANSAC filter
  4. TPS warp
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_tl = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_tl):
    os.environ["PATH"] = _tl + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_tl)

from pathlib import Path
import numpy as np
import pandas as pd
from tifffile import imread

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "napari-cell-registration" / "src"))

from napari_cell_registration.core import (
    CellFeaturesConfig, CellposeConfig, CellposeSegmenter,
    MatchingConfig, compute_cell_features,
    estimate_rigid_transform_from_matches,
    estimate_rigid_transform_from_matches_ransac,
    rigid_transform_to_affine, greedy_match_cells,
    MIN_MATCHES_FOR_REFINEMENT,
)
from napari_cell_registration.core.point_registration import (
    fit_tps_from_matches, warp_image_with_tps,
)
from cell_registration.evaluation import compute_prewarped_registration_metrics

SYNTHETIC_ROOT = PROJECT_ROOT / "synthetic_dataset"
CELLPOSE_CFG = CellposeConfig(
    gpu=True, pretrained_model="cpsam", diameter=15,
    flow_threshold=-2.0, cellprob_threshold=1.0, min_size=5,
)
FEATURE_CFG = CellFeaturesConfig(topology_neighbor_k=5)

# ── Load & segment ──
print("Loading and segmenting rigid_0049...")
segmenter = CellposeSegmenter(CELLPOSE_CFG)

img1 = imread(str(SYNTHETIC_ROOT / "rigid" / "target" / "0049_target.tif"))
img2 = imread(str(SYNTHETIC_ROOT / "rigid" / "moving" / "0049_moving.tif"))
mask1, _, _ = segmenter.segment_array(np.asarray(img1))
mask2, _, _ = segmenter.segment_array(np.asarray(img2))
feats1 = compute_cell_features(mask1, FEATURE_CFG)
feats2 = compute_cell_features(mask2, FEATURE_CFG)

pts1 = feats1[["centroid_x", "centroid_y"]].to_numpy()
pts2 = feats2[["centroid_x", "centroid_y"]].to_numpy()

print(f"Target: {len(feats1)} cells, Moving: {len(feats2)} cells")

# ── Step 1: Global greedy match — morphology only, no spatial window ──
for top_k in [320, 500]:
    for ransac_thresh in [2.0, 5.0, 10.0, 20.0]:
        config = MatchingConfig(
            feature_weight=1.0,
            topology_weight=0.0,
            position_weight=0.0,
            top_k=top_k,
            distance_threshold=None,
            spatial_window_size=None,
        )
        matches = greedy_match_cells(feats1, feats2, config)

        # ── Step 2: RANSAC ──
        transform, inlier_mask = estimate_rigid_transform_from_matches_ransac(
            feats1, feats2, matches,
            max_trials=2000,
            residual_threshold=ransac_thresh,
            min_inliers=MIN_MATCHES_FOR_REFINEMENT,
        )
        inlier_count = int(inlier_mask.sum())
        matches_filtered = matches.loc[inlier_mask].reset_index(drop=True)

        if len(matches_filtered) < 10:
            print(f"\ntop_k={top_k}, ransac_thresh={ransac_thresh}: "
                  f"Only {len(matches_filtered)} inliers, skipping")
            continue

        # Check displacement stats of inliers
        idx1 = matches_filtered["idx1"].to_numpy(dtype=int)
        idx2 = matches_filtered["idx2"].to_numpy(dtype=int)
        pts_f = pts1[idx1]
        pts_m = pts2[idx2]
        disps = pts_f - pts_m
        norms = np.linalg.norm(disps, axis=1)

        # ── Step 3: TPS warp ──
        try:
            affine = rigid_transform_to_affine(transform)
            tps = fit_tps_from_matches(
                pts_f, pts_m, output_shape=mask1.shape[:2],
                rigid_transform=affine, regularization=1e-3,
                n_boundary_per_side=4, add_boundary_anchors_flag=True,
            )
            mask2_warped = warp_image_with_tps(
                mask2.astype(np.int32), tps, mask1.shape[:2], order=0)
            mask2_reg = np.rint(mask2_warped).astype(np.int32)

            metrics = compute_prewarped_registration_metrics(
                mask2_reg, mask1, instance_iou_threshold=0.3)

            print(f"\ntop_k={top_k}, ransac_thresh={ransac_thresh}: "
                  f"{inlier_count} inliers, "
                  f"disp mean={norms.mean():.1f} std={norms.std():.1f}")
            print(f"  F1={metrics['match_f1']:.4f}  "
                  f"Prec={metrics['match_precision']:.4f}  "
                  f"Recall={metrics['match_recall']:.4f}  "
                  f"Matched={metrics['matched_cells']}/{metrics['eligible_fixed_cells']}")
        except Exception as e:
            print(f"\ntop_k={top_k}, ransac_thresh={ransac_thresh}: TPS failed: {e}")
