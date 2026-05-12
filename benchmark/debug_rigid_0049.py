"""Debug rigid_0049: analyze why matching fails."""
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
    MatchingConfig, MIN_MATCHES_FOR_REFINEMENT,
    compute_cell_features, compute_match_residuals,
    estimate_rigid_transform_from_matches,
    estimate_rigid_transform_from_matches_ransac,
    greedy_match_cells, rigid_transform_to_affine, two_stage_match_cells,
)
from napari_cell_registration.core.matching import (
    apply_transform_to_features,
    _filter_candidate_matches_by_hard_constraints,
)

SYNTHETIC_ROOT = PROJECT_ROOT / "synthetic_dataset"
CELLPOSE_CFG = CellposeConfig(
    gpu=True, pretrained_model="cpsam", diameter=15,
    flow_threshold=-2.0, cellprob_threshold=1.0, min_size=5,
)
FEATURE_CFG = CellFeaturesConfig(topology_neighbor_k=5)

# Load images
target_path = SYNTHETIC_ROOT / "rigid" / "target" / "0049_target.tif"
moving_path = SYNTHETIC_ROOT / "rigid" / "moving" / "0049_moving.tif"
img1 = imread(str(target_path))
img2 = imread(str(moving_path))

print(f"Target image shape: {img1.shape}, dtype: {img1.dtype}")
print(f"Moving image shape: {img2.shape}, dtype: {img2.dtype}")
print(f"Target value range: [{img1.min()}, {img1.max()}]")
print(f"Moving value range: [{img2.min()}, {img2.max()}]")

# Check if images are identical or shifted
diff = img1.astype(float) - img2.astype(float)
print(f"\nPixel difference stats: mean={diff.mean():.4f}, std={diff.std():.4f}, "
      f"min={diff.min():.4f}, max={diff.max():.4f}")
print(f"Identical pixels: {(img1 == img2).sum()}/{img1.size} = {(img1 == img2).mean()*100:.1f}%")

# Segment
segmenter = CellposeSegmenter(CELLPOSE_CFG)
mask1, _, _ = segmenter.segment_array(np.asarray(img1))
mask2, _, _ = segmenter.segment_array(np.asarray(img2))

print(f"\nTarget cells: {mask1.max()}, Moving cells: {mask2.max()}")
print(f"Target mask nonzero: {(mask1 > 0).sum()}, Moving mask nonzero: {(mask2 > 0).sum()}")

feats1 = compute_cell_features(mask1, FEATURE_CFG)
feats2 = compute_cell_features(mask2, FEATURE_CFG)

print(f"\nTarget features: {len(feats1)} cells")
print(f"Moving features: {len(feats2)} cells")

# Compare centroid distributions
cx1 = feats1["centroid_x"].values
cy1 = feats1["centroid_y"].values
cx2 = feats2["centroid_x"].values
cy2 = feats2["centroid_y"].values

print(f"\nTarget centroid X: [{cx1.min():.1f}, {cx1.max():.1f}], mean={cx1.mean():.1f}")
print(f"Target centroid Y: [{cy1.min():.1f}, {cy1.max():.1f}], mean={cy1.mean():.1f}")
print(f"Moving centroid X: [{cx2.min():.1f}, {cx2.max():.1f}], mean={cx2.mean():.1f}")
print(f"Moving centroid Y: [{cy2.min():.1f}, {cy2.max():.1f}], mean={cy2.mean():.1f}")

# Estimate global displacement
dx = cx1.mean() - cx2.mean()
dy = cy1.mean() - cy2.mean()
print(f"\nEstimated global displacement: dx={dx:.1f}, dy={dy:.1f}, dist={np.sqrt(dx**2+dy**2):.1f}")

# Compare feature distributions (area, eccentricity, etc.)
print(f"\n--- Feature comparison ---")
for col in ["area", "eccentricity", "major_axis_length", "minor_axis_length"]:
    if col in feats1.columns and col in feats2.columns:
        v1 = feats1[col].values
        v2 = feats2[col].values
        print(f"  {col}: target mean={v1.mean():.2f} std={v1.std():.2f} | "
              f"moving mean={v2.mean():.2f} std={v2.std():.2f}")

# Run matching step by step
print("\n\n=== STEP-BY-STEP MATCHING ===\n")

# Step 1: two_stage match with default params
TOP_K = 320
MAX_DIST = 100
match_result = two_stage_match_cells(
    feats1, feats2, mask1.shape,
    feature_weight=1.0, topology_weight=0.0, position_weight=1.0,
    top_k=TOP_K, distance_threshold=None, spatial_window_size=float(MAX_DIST),
    min_cells_for_two_stage=10, coarse_top_k=max(24, TOP_K),
    coarse_distance_threshold=2.0, coarse_matching_mode="morphology_guided",
    coarse_allow_scale=False, coarse_prefer_affine=False,
    coarse_residual_threshold=max(5.0, 2.0 * 2.0),
    coarse_max_trials=min(max(1000, 200), 2000),
)
matches = match_result.matches.copy()
print(f"Step 1 - Initial matches: {len(matches)}")
print(f"  coarse_transform_accepted: {match_result.coarse_transform_accepted}")
if hasattr(match_result, 'coarse_transform') and match_result.coarse_transform is not None:
    print(f"  coarse_transform: {match_result.coarse_transform}")

# Analyze match displacements
pts_f = feats1.iloc[matches["idx1"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
pts_m = feats2.iloc[matches["idx2"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
disps = pts_f - pts_m
norms = np.linalg.norm(disps, axis=1)
angles = np.degrees(np.arctan2(disps[:, 1], disps[:, 0]))

print(f"\n  Match displacement stats:")
print(f"    norm: mean={norms.mean():.1f}, std={norms.std():.1f}, min={norms.min():.1f}, max={norms.max():.1f}")
print(f"    dx:   mean={disps[:,0].mean():.1f}, std={disps[:,0].std():.1f}")
print(f"    dy:   mean={disps[:,1].mean():.1f}, std={disps[:,1].std():.1f}")
print(f"    angle: mean={angles.mean():.1f}, std={angles.std():.1f}")

# Check how many matches have consistent direction
median_dx = np.median(disps[:, 0])
median_dy = np.median(disps[:, 1])
consistent = np.sum((np.abs(disps[:,0] - median_dx) < 10) & (np.abs(disps[:,1] - median_dy) < 10))
print(f"    Matches within 10px of median displacement ({median_dx:.1f},{median_dy:.1f}): {consistent}/{len(matches)}")

# Step 2: RANSAC
transform, inlier_mask = estimate_rigid_transform_from_matches_ransac(
    feats1, feats2, matches,
    max_trials=1000, residual_threshold=2.0,
    min_inliers=MIN_MATCHES_FOR_REFINEMENT,
)
inlier_count = int(inlier_mask.sum())
print(f"\nStep 2 - RANSAC inliers: {inlier_count}/{len(matches)}")
print(f"  Transform: {transform}")

# Residuals
all_residuals = compute_match_residuals(feats1, feats2, matches, transform)
print(f"  All residuals: mean={all_residuals.mean():.2f}, median={np.median(all_residuals):.2f}, "
      f"std={all_residuals.std():.2f}, max={all_residuals.max():.2f}")
inlier_residuals = all_residuals[inlier_mask]
outlier_residuals = all_residuals[~inlier_mask]
print(f"  Inlier residuals: mean={inlier_residuals.mean():.2f}, max={inlier_residuals.max():.2f}")
if len(outlier_residuals) > 0:
    print(f"  Outlier residuals: mean={outlier_residuals.mean():.2f}, min={outlier_residuals.min():.2f}")

# Check what the rigid transform looks like
affine = rigid_transform_to_affine(transform)
print(f"\n  Affine matrix:\n{affine}")

# Step 3: orientation filter  
ORIENTATION_MAX_DEG = 5.0
ORIENTATION_MIN_ECC = 0.15
n_before = len(matches)
filtered = _filter_candidate_matches_by_hard_constraints(
    matches, max_area_ratio=None, max_aspect_ratio_ratio=None,
    max_orientation_diff_deg=ORIENTATION_MAX_DEG,
    min_orientation_eccentricity=ORIENTATION_MIN_ECC,
)
print(f"\nStep 4 - Orientation filter: {n_before} → {len(filtered)}")

# Check orientation values
if "orientation_1" in matches.columns and "orientation_2" in matches.columns:
    o1 = matches["orientation_1"].values
    o2 = matches["orientation_2"].values
    odiff = np.abs(o1 - o2)
    odiff = np.minimum(odiff, 180 - odiff)  # wrap-around
    print(f"  Orientation diff: mean={odiff.mean():.2f}, std={odiff.std():.2f}, max={odiff.max():.2f}")
elif "orientation1" in feats1.columns:
    idx1 = matches["idx1"].to_numpy(dtype=int)
    idx2 = matches["idx2"].to_numpy(dtype=int)
    o1 = feats1.iloc[idx1]["orientation"].values if "orientation" in feats1.columns else None
    o2 = feats2.iloc[idx2]["orientation"].values if "orientation" in feats2.columns else None
    if o1 is not None and o2 is not None:
        odiff = np.abs(np.degrees(o1) - np.degrees(o2))
        print(f"  Orientation diff: mean={odiff.mean():.2f}°")

# Step 5: Consensus filter analysis
print(f"\n=== CONSENSUS FILTER ANALYSIS ===")
PATCH_GRID = 4
CONSENSUS_ANGLE_DEG = 5.0
CONSENSUS_LENGTH_TOL = 5.0

n_before_disp = len(matches)
pts_f = feats1.iloc[matches["idx1"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
pts_m = feats2.iloc[matches["idx2"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
disp_vectors = pts_f - pts_m
h, w = mask1.shape[:2]
grid = PATCH_GRID
px = np.clip(np.floor(pts_f[:, 0] * grid / max(w, 1)).astype(int), 0, grid - 1)
py = np.clip(np.floor(pts_f[:, 1] * grid / max(h, 1)).astype(int), 0, grid - 1)
patch_ids = py * grid + px

print(f"Image shape: {mask1.shape}")
print(f"Grid: {grid}x{grid}")

for pid in range(grid * grid):
    in_patch = patch_ids == pid
    n_in = int(in_patch.sum())
    if n_in == 0:
        continue
    patch_disp = disp_vectors[in_patch]
    patch_norms = np.linalg.norm(patch_disp, axis=1)
    patch_angles = np.degrees(np.arctan2(patch_disp[:, 1], patch_disp[:, 0]))
    py_idx = pid // grid
    px_idx = pid % grid
    print(f"  Patch [{py_idx},{px_idx}] (pid={pid}): {n_in} matches, "
          f"norm: mean={patch_norms.mean():.1f} std={patch_norms.std():.1f}, "
          f"angle: mean={patch_angles.mean():.1f}° std={patch_angles.std():.1f}°")

# Also check: what is the actual rigid shift in the synthetic data?
# Try to find ground truth
gt_dir = SYNTHETIC_ROOT / "rigid"
gt_files = list(gt_dir.glob("**/0049*"))
print(f"\nGround truth files for 0049: {[str(f) for f in gt_files]}")

# Check if there's a transforms file
for p in [SYNTHETIC_ROOT / "rigid" / "transforms.json",
          SYNTHETIC_ROOT / "rigid" / "params.json",
          SYNTHETIC_ROOT / "rigid" / "metadata.json",
          SYNTHETIC_ROOT / "transforms.json",
          SYNTHETIC_ROOT / "params.json"]:
    if p.exists():
        print(f"\nFound: {p}")
        import json
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, dict) and "0049" in str(data):
            print(f"  Contains 0049 entry")
        elif isinstance(data, list) and len(data) > 49:
            print(f"  Entry 49: {data[49]}")
        else:
            # Print first few keys
            if isinstance(data, dict):
                keys = list(data.keys())[:10]
                print(f"  Keys: {keys}")
                if "0049" in data:
                    print(f"  0049: {data['0049']}")
