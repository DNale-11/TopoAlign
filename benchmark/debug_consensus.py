"""Check if consensus filter is removing correct matches."""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_tl = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_tl):
    os.environ["PATH"] = _tl + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_tl)

from pathlib import Path
import numpy as np
from tifffile import imread

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "napari-cell-registration" / "src"))

from napari_cell_registration.core import (
    CellFeaturesConfig, CellposeConfig, CellposeSegmenter,
    compute_cell_features,
)

SYNTHETIC_ROOT = PROJECT_ROOT / "synthetic_dataset"
CELLPOSE_CFG = CellposeConfig(
    gpu=True, pretrained_model="cpsam", diameter=15,
    flow_threshold=-2.0, cellprob_threshold=1.0, min_size=5,
)
FEATURE_CFG = CellFeaturesConfig(topology_neighbor_k=5)

segmenter = CellposeSegmenter(CELLPOSE_CFG)

# ── Analyze rigid_0049 ──
target_path = SYNTHETIC_ROOT / "rigid" / "target" / "0049_target.tif"
moving_path = SYNTHETIC_ROOT / "rigid" / "moving" / "0049_moving.tif"
img1 = imread(str(target_path))
img2 = imread(str(moving_path))

mask1, _, _ = segmenter.segment_array(np.asarray(img1))
mask2, _, _ = segmenter.segment_array(np.asarray(img2))
feats1 = compute_cell_features(mask1, FEATURE_CFG)
feats2 = compute_cell_features(mask2, FEATURE_CFG)

# ── Ground-truth matching via nearest centroid ──
# For a rigid translation, the TRUE match is the nearest cell in the other image
# after applying the true displacement
pts1 = feats1[["centroid_x", "centroid_y"]].to_numpy()
pts2 = feats2[["centroid_x", "centroid_y"]].to_numpy()

from scipy.spatial import cKDTree

tree1 = cKDTree(pts1)
tree2 = cKDTree(pts2)

# Find nearest neighbor for each target cell in moving
dist_1to2, idx_1to2 = tree2.query(pts1, k=1)
# Find nearest neighbor for each moving cell in target
dist_2to1, idx_2to1 = tree1.query(pts2, k=1)

# Mutual nearest neighbors = high-confidence ground truth matches
mutual_matches = []
for i in range(len(pts1)):
    j = idx_1to2[i]
    if idx_2to1[j] == i:  # mutual nearest neighbor
        mutual_matches.append((i, j, dist_1to2[i]))

mutual_matches = np.array(mutual_matches)
print(f"=== Ground Truth Analysis (Mutual Nearest Neighbors) ===")
print(f"Mutual nearest neighbor pairs: {len(mutual_matches)}")
print(f"Distance stats: mean={mutual_matches[:,2].mean():.2f}, "
      f"std={mutual_matches[:,2].std():.2f}, "
      f"max={mutual_matches[:,2].max():.2f}")

# The displacement vectors for ground truth matches
gt_disps = pts1[mutual_matches[:,0].astype(int)] - pts2[mutual_matches[:,1].astype(int)]
gt_norms = np.linalg.norm(gt_disps, axis=1)
gt_angles = np.degrees(np.arctan2(gt_disps[:, 1], gt_disps[:, 0]))

print(f"\nGround truth displacement vectors:")
print(f"  dx: mean={gt_disps[:,0].mean():.2f}, std={gt_disps[:,0].std():.2f}")
print(f"  dy: mean={gt_disps[:,1].mean():.2f}, std={gt_disps[:,1].std():.2f}")
print(f"  norm: mean={gt_norms.mean():.2f}, std={gt_norms.std():.2f}, min={gt_norms.min():.2f}, max={gt_norms.max():.2f}")
print(f"  angle: mean={gt_angles.mean():.2f}°, std={gt_angles.std():.2f}°")

# How many GT matches are within consensus thresholds?
CONSENSUS_ANGLE_DEG = 5.0
CONSENSUS_LENGTH_TOL = 5.0
median_angle = np.median(gt_angles)
median_norm = np.median(gt_norms)
angle_diff = np.abs(gt_angles - median_angle)
angle_diff = np.minimum(angle_diff, 360 - angle_diff)
consistent = np.sum((angle_diff < CONSENSUS_ANGLE_DEG) & (np.abs(gt_norms - median_norm) < CONSENSUS_LENGTH_TOL))
print(f"\n  GT matches consistent with median (angle<{CONSENSUS_ANGLE_DEG}°, len_tol<{CONSENSUS_LENGTH_TOL}): "
      f"{consistent}/{len(mutual_matches)} = {consistent/len(mutual_matches)*100:.1f}%")

# Now check: how many of the 320 initial matches are actually correct?
# Load the initial matches from the pipeline
import pandas as pd
feat_dir = PROJECT_ROOT / "benchmark" / "results" / "synthetic_tps" / "features"
matches_df = pd.read_csv(feat_dir / "rigid_0049_matches.csv")
print(f"\n=== Pipeline Match Quality ===")
print(f"Pipeline produced {len(matches_df)} final matches")

# Check how many pipeline matches overlap with ground truth
pipeline_pairs = set(zip(matches_df["idx1"].values, matches_df["idx2"].values))
gt_pairs = set(zip(mutual_matches[:,0].astype(int), mutual_matches[:,1].astype(int)))
correct = pipeline_pairs & gt_pairs
print(f"Pipeline matches that are correct (in GT): {len(correct)}/{len(pipeline_pairs)}")

# ── Now run matching step-by-step and check correctness at each stage ──
from napari_cell_registration.core import (
    MatchingConfig, MIN_MATCHES_FOR_REFINEMENT,
    compute_match_residuals,
    estimate_rigid_transform_from_matches,
    estimate_rigid_transform_from_matches_ransac,
    greedy_match_cells, rigid_transform_to_affine, two_stage_match_cells,
)
from napari_cell_registration.core.matching import (
    apply_transform_to_features,
    _filter_candidate_matches_by_hard_constraints,
)

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
init_matches = match_result.matches.copy()

# Check correctness of initial 320 matches
init_pairs = set(zip(init_matches["idx1"].values, init_matches["idx2"].values))
init_correct = init_pairs & gt_pairs
print(f"\n=== Stage-by-stage correctness ===")
print(f"Initial 320 matches: {len(init_correct)} correct ({len(init_correct)/len(init_pairs)*100:.1f}%)")

# Check displacement distribution of the CORRECT vs INCORRECT initial matches
correct_mask = np.array([
    (row["idx1"], row["idx2"]) in gt_pairs 
    for _, row in init_matches.iterrows()
])
pts_f = feats1.iloc[init_matches["idx1"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy()
pts_m = feats2.iloc[init_matches["idx2"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy()
disps = pts_f - pts_m
norms = np.linalg.norm(disps, axis=1)

print(f"\n  Correct matches displacement: norm mean={norms[correct_mask].mean():.1f}, std={norms[correct_mask].std():.1f}")
if (~correct_mask).any():
    print(f"  Wrong matches displacement:   norm mean={norms[~correct_mask].mean():.1f}, std={norms[~correct_mask].std():.1f}")
print(f"  Correct matches dx: mean={disps[correct_mask,0].mean():.1f}, dy: mean={disps[correct_mask,1].mean():.1f}")
if (~correct_mask).any():
    print(f"  Wrong matches dx:   mean={disps[~correct_mask,0].mean():.1f}, dy: mean={disps[~correct_mask,1].mean():.1f}")

# Also do the same for a GOOD rigid case (rigid_0001) for comparison
print(f"\n\n=== COMPARISON: rigid_0001 (good case) ===")
target2 = SYNTHETIC_ROOT / "rigid" / "target" / "0001_target.tif"
moving2 = SYNTHETIC_ROOT / "rigid" / "moving" / "0001_moving.tif"
i1 = imread(str(target2))
i2 = imread(str(moving2))
m1, _, _ = segmenter.segment_array(np.asarray(i1))
m2, _, _ = segmenter.segment_array(np.asarray(i2))
f1 = compute_cell_features(m1, FEATURE_CFG)
f2 = compute_cell_features(m2, FEATURE_CFG)

p1 = f1[["centroid_x", "centroid_y"]].to_numpy()
p2 = f2[["centroid_x", "centroid_y"]].to_numpy()
t1 = cKDTree(p1)
t2 = cKDTree(p2)
d12, i12 = t2.query(p1, k=1)
d21, i21 = t1.query(p2, k=1)
gt2 = []
for i in range(len(p1)):
    j = i12[i]
    if i21[j] == i:
        gt2.append((i, j, d12[i]))
gt2 = np.array(gt2)
gt2_disps = p1[gt2[:,0].astype(int)] - p2[gt2[:,1].astype(int)]
gt2_norms = np.linalg.norm(gt2_disps, axis=1)
gt2_angles = np.degrees(np.arctan2(gt2_disps[:,1], gt2_disps[:,0]))

print(f"Target: {len(f1)} cells, Moving: {len(f2)} cells")
print(f"Mutual NN: {len(gt2)}, dist: mean={gt2[:,2].mean():.2f}")
print(f"GT displacement: dx={gt2_disps[:,0].mean():.2f}(std={gt2_disps[:,0].std():.2f}), "
      f"dy={gt2_disps[:,1].mean():.2f}(std={gt2_disps[:,1].std():.2f})")
print(f"GT norm: mean={gt2_norms.mean():.2f}, std={gt2_norms.std():.2f}")
print(f"GT angle: mean={gt2_angles.mean():.2f}°, std={gt2_angles.std():.2f}°")

# Run matching on good case too
mr2 = two_stage_match_cells(
    f1, f2, m1.shape,
    feature_weight=1.0, topology_weight=0.0, position_weight=1.0,
    top_k=TOP_K, distance_threshold=None, spatial_window_size=float(MAX_DIST),
    min_cells_for_two_stage=10, coarse_top_k=max(24, TOP_K),
    coarse_distance_threshold=2.0, coarse_matching_mode="morphology_guided",
    coarse_allow_scale=False, coarse_prefer_affine=False,
    coarse_residual_threshold=max(5.0, 2.0 * 2.0),
    coarse_max_trials=min(max(1000, 200), 2000),
)
im2 = mr2.matches.copy()
gt2_pairs = set(zip(gt2[:,0].astype(int), gt2[:,1].astype(int)))
im2_pairs = set(zip(im2["idx1"].values, im2["idx2"].values))
im2_correct = im2_pairs & gt2_pairs
print(f"coarse_accepted: {mr2.coarse_transform_accepted}")
print(f"Initial {len(im2)} matches: {len(im2_correct)} correct ({len(im2_correct)/len(im2_pairs)*100:.1f}%)")
