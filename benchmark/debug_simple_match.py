"""
Simple patch-based morphology matching for rigid_0049.
No spatial window, no consensus, no RANSAC — pure feature similarity per patch.
Uses the existing matching API with position_weight=0.
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
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "napari-cell-registration" / "src"))

from napari_cell_registration.core import (
    CellFeaturesConfig, CellposeConfig, CellposeSegmenter,
    MatchingConfig, compute_cell_features, estimate_rigid_transform_from_matches,
    rigid_transform_to_affine,
)
from napari_cell_registration.core.matching import compute_match_distance_matrix
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
H, W = mask1.shape[:2]

print(f"Target: {len(feats1)} cells, Moving: {len(feats2)} cells")

# ── Ground truth (mutual nearest neighbor) ──
tree1 = cKDTree(pts1)
tree2 = cKDTree(pts2)
d12, i12 = tree2.query(pts1, k=1)
d21, i21 = tree1.query(pts2, k=1)
gt_pairs = set()
for i in range(len(pts1)):
    j = i12[i]
    if i21[j] == i and d12[i] < 30:
        gt_pairs.add((i, j))
print(f"Ground truth mutual NN pairs (<30px): {len(gt_pairs)}")


def patch_morphology_match(feats1, feats2, pts1, pts2, grid, H, W):
    """Match per patch: morphology-only, no spatial window, Hungarian assignment."""
    config = MatchingConfig(
        feature_weight=1.0,
        topology_weight=0.0,
        position_weight=0.0,  # pure morphology
        top_k=9999,
        distance_threshold=None,
        spatial_window_size=None,  # no window
    )

    all_matches = []
    used_2 = set()

    for py in range(grid):
        for px in range(grid):
            y0, y1 = py * H // grid, (py + 1) * H // grid
            x0, x1 = px * W // grid, (px + 1) * W // grid

            idx1_in = [i for i in range(len(pts1))
                       if x0 <= pts1[i, 0] < x1 and y0 <= pts1[i, 1] < y1]
            idx2_in = [j for j in range(len(pts2))
                       if x0 <= pts2[j, 0] < x1 and y0 <= pts2[j, 1] < y1]

            if not idx1_in or not idx2_in:
                continue

            # Build local feature DataFrames
            local_f1 = feats1.iloc[idx1_in].reset_index(drop=True)
            local_f2 = feats2.iloc[idx2_in].reset_index(drop=True)

            # Compute distance matrix using existing API
            dist = compute_match_distance_matrix(local_f1, local_f2, config)

            # Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(dist)

            for r, c in zip(row_ind, col_ind):
                if not np.isfinite(dist[r, c]):
                    continue
                j_global = idx2_in[c]
                if j_global not in used_2:
                    all_matches.append({
                        "idx1": idx1_in[r],
                        "idx2": j_global,
                        "distance": float(dist[r, c]),
                    })
                    used_2.add(j_global)

    return pd.DataFrame(all_matches)


# ── Try different grid sizes ──
for grid in [1, 2, 4, 8]:
    matches = patch_morphology_match(feats1, feats2, pts1, pts2, grid, H, W)
    matched_pairs = set(zip(matches["idx1"].values, matches["idx2"].values))
    correct = matched_pairs & gt_pairs

    disps = pts1[matches["idx1"].values] - pts2[matches["idx2"].values]
    norms = np.linalg.norm(disps, axis=1)

    print(f"\n=== Grid {grid}x{grid}: {len(matches)} matches, "
          f"{len(correct)} correct ({len(correct)/max(len(matches),1)*100:.1f}%) ===")
    print(f"  Displacement: mean={norms.mean():.1f}, median={np.median(norms):.1f}, "
          f"std={norms.std():.1f}")

    if len(matches) >= 20:
        # TPS warp & evaluate
        try:
            transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)
            affine = rigid_transform_to_affine(transform)
            pts_f = pts1[matches["idx1"].values]
            pts_m = pts2[matches["idx2"].values]
            tps = fit_tps_from_matches(
                pts_f, pts_m, output_shape=mask1.shape[:2],
                rigid_transform=affine, regularization=1e-3,
                n_boundary_per_side=4, add_boundary_anchors_flag=True,
            )
            mask2_warped = warp_image_with_tps(mask2.astype(np.int32), tps, mask1.shape[:2], order=0)
            mask2_reg = np.rint(mask2_warped).astype(np.int32)
            metrics = compute_prewarped_registration_metrics(mask2_reg, mask1, instance_iou_threshold=0.3)
            print(f"  → F1={metrics['match_f1']:.4f}  Prec={metrics['match_precision']:.4f}  "
                  f"Recall={metrics['match_recall']:.4f}  "
                  f"Matched={metrics['matched_cells']}/{metrics['eligible_fixed_cells']}")
        except Exception as e:
            print(f"  → TPS failed: {e}")
