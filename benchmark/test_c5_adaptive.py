"""Quick test: C5-2 with adaptive window."""
from __future__ import annotations
import os, sys, functools
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_tl = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_tl):
    os.environ["PATH"] = _tl + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"): os.add_dll_directory(_tl)
try: import torch
except: pass
sys.path.insert(0, "napari-cell-registration/src")
sys.path.insert(0, ".")
import numpy as np
from tifffile import imread
print = functools.partial(print, flush=True)

from napari_cell_registration.core import *
from napari_cell_registration.core.matching import (
    apply_transform_to_features, _filter_candidate_matches_by_hard_constraints,
)
from napari_cell_registration.core.point_registration import fit_tps_from_matches, warp_image_with_tps
from cell_registration.evaluation import compute_prewarped_registration_metrics

seg = CellposeSegmenter(CellposeConfig(gpu=True, pretrained_model="cpsam", diameter=15, flow_threshold=-2.0, cellprob_threshold=1.0, min_size=5))
feat_cfg = CellFeaturesConfig(topology_neighbor_k=5)

TOP_K, MAX_DIST, POS_W = 320, 100, 1.0
RANSAC_RES, RANSAC_TRIALS = 2.0, 1000
MIN_CONSENSUS_FOR_TPS = 50

fixed_path = r"benchmark\unregistration\C5-2\fixed\C5-C-2.tif"
img1 = imread(fixed_path)
mask1, _, _ = seg.segment_array(np.asarray(img1))
feats1 = compute_cell_features(mask1, feat_cfg)
print(f"Fixed: {len(feats1)} cells")

for mov_name in ["C5-A-2.tif", "C5-B-2.tif", "C5-D-2.tif", "C5-E-2.tif"]:
    mov_path = f"benchmark\\unregistration\\C5-2\\moving\\{mov_name}"
    img2 = imread(mov_path)
    mask2, _, _ = seg.segment_array(np.asarray(img2))
    feats2 = compute_cell_features(mask2, feat_cfg)
    print(f"\n=== {mov_name}: {len(feats2)} cells ===")

    for window_scale in (1.0, 2.0):
        eff_max = MAX_DIST * window_scale
        print(f"  Window scale={window_scale}x  max_dist={eff_max:.0f}px")

        mr = two_stage_match_cells(
            feats1, feats2, mask1.shape, feature_weight=1.0, topology_weight=0.0,
            position_weight=POS_W, top_k=TOP_K, distance_threshold=None,
            spatial_window_size=float(eff_max), min_cells_for_two_stage=10,
            coarse_top_k=max(24, TOP_K), coarse_distance_threshold=2.0,
            coarse_matching_mode="morphology_guided", coarse_allow_scale=False,
            coarse_prefer_affine=False, coarse_residual_threshold=max(5.0, RANSAC_RES*2),
            coarse_max_trials=min(max(RANSAC_TRIALS,200),2000),
        )
        matches = mr.matches.copy()

        transform, inlier_mask = estimate_rigid_transform_from_matches_ransac(
            feats1, feats2, matches, max_trials=RANSAC_TRIALS,
            residual_threshold=RANSAC_RES, min_inliers=MIN_MATCHES_FOR_REFINEMENT,
        )
        inlier_count = int(inlier_mask.sum())
        print(f"    RANSAC: {inlier_count}/{len(matches)} inliers")

        # Model consistency
        res = compute_match_residuals(feats1, feats2, matches, transform)
        if inlier_count >= MIN_MATCHES_FOR_REFINEMENT:
            ir = res[inlier_mask]
            med = float(np.median(ir))
            mad = float(np.median(np.abs(ir - med)))
            thr = max(RANSAC_RES * 2, med + 3 * max(1.4826 * mad, 0.5))
            km = res <= thr
            kc = int(km.sum())
            if MIN_MATCHES_FOR_REFINEMENT <= kc < len(matches):
                matches = matches.loc[km].reset_index(drop=True)
                transform, _ = estimate_rigid_transform_from_matches_ransac(
                    feats1, feats2, matches, max_trials=RANSAC_TRIALS,
                    residual_threshold=RANSAC_RES, min_inliers=MIN_MATCHES_FOR_REFINEMENT,
                )

        # Guided rematch
        if len(matches) >= MIN_MATCHES_FOR_REFINEMENT:
            aff = rigid_transform_to_affine(transform)
            af2 = apply_transform_to_features(feats2, aff, mask1.shape)
            rw = max(10.0, float(eff_max) * 0.6)
            rc = MatchingConfig(feature_weight=1.0, topology_weight=0.0, position_weight=POS_W,
                                top_k=TOP_K, distance_threshold=None, spatial_window_size=rw)
            rm = greedy_match_cells(feats1, af2, rc, image_shape=mask1.shape, coverage_patch_grid=4)
            if len(rm) >= MIN_MATCHES_FOR_REFINEMENT:
                transform = estimate_rigid_transform_from_matches(feats1, feats2, rm)
                matches = rm

        # Orientation filter
        matches = _filter_candidate_matches_by_hard_constraints(
            matches, max_area_ratio=None, max_aspect_ratio_ratio=None,
            max_orientation_diff_deg=5.0, min_orientation_eccentricity=0.15,
        )
        if len(matches) >= MIN_MATCHES_FOR_REFINEMENT:
            transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)

        # Consensus filter
        n0 = len(matches)
        if n0 >= MIN_MATCHES_FOR_REFINEMENT:
            pf = feats1.iloc[matches["idx1"].to_numpy(int)][["centroid_x","centroid_y"]].to_numpy(float)
            pm = feats2.iloc[matches["idx2"].to_numpy(int)][["centroid_x","centroid_y"]].to_numpy(float)
            dv = pf - pm
            h, w = mask1.shape[:2]
            g = 4
            px = np.clip(np.floor(pf[:,0]*g/max(w,1)).astype(int), 0, g-1)
            py = np.clip(np.floor(pf[:,1]*g/max(h,1)).astype(int), 0, g-1)
            pids = py*g + px
            km = np.ones(n0, dtype=bool)
            ct = np.cos(np.radians(5.0))
            for pid in range(g*g):
                ip = pids == pid
                ni = int(ip.sum())
                if ni < 3: continue
                pd = dv[ip]; pi = np.where(ip)[0]; pn = np.linalg.norm(pd, axis=1)
                votes = np.zeros(ni, int)
                for a in range(ni):
                    if pn[a]<1: continue
                    for b in range(ni):
                        if pn[b]<1: continue
                        if np.dot(pd[a],pd[b])/(pn[a]*pn[b]+1e-8) >= ct: votes[a]+=1
                if votes.max()<2: continue
                ci = int(np.argmax(votes)); cd = pd[ci]; cn = pn[ci]
                cms = []
                for k in range(ni):
                    if pn[k]<1: km[pi[k]]=False; continue
                    if np.dot(pd[k],cd)/(pn[k]*cn+1e-8)<ct: km[pi[k]]=False
                    else: cms.append(k)
                if len(cms)>=3:
                    ml = float(np.mean([pn[k] for k in cms]))
                    for k in cms:
                        if abs(pn[k]-ml)>5: km[pi[k]]=False
            nad = int(km.sum())
            if nad >= MIN_MATCHES_FOR_REFINEMENT and nad < n0:
                matches = matches.loc[km].reset_index(drop=True)
                transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)

        n_pts = len(matches)
        print(f"    After consensus: {n_pts} control points")

        if n_pts >= MIN_CONSENSUS_FOR_TPS:
            # TPS + F1
            aff = rigid_transform_to_affine(transform)
            pf = feats1.iloc[matches["idx1"].to_numpy(int)][["centroid_x","centroid_y"]].to_numpy(float)
            pm = feats2.iloc[matches["idx2"].to_numpy(int)][["centroid_x","centroid_y"]].to_numpy(float)
            tps = fit_tps_from_matches(pf, pm, output_shape=mask1.shape[:2],
                rigid_transform=aff, regularization=1e-3, n_boundary_per_side=4, add_boundary_anchors_flag=True)
            m2w = warp_image_with_tps(mask2.astype(np.int32), tps, mask1.shape[:2], order=0)
            m2r = np.rint(m2w).astype(np.int32)
            met = compute_prewarped_registration_metrics(m2r, mask1, instance_iou_threshold=0.3)
            print(f"    F1={met['match_f1']:.4f}  Prec={met['match_precision']:.4f}  Recall={met['match_recall']:.4f}  Matched={met['matched_cells']}/{met['eligible_fixed_cells']}")
            break  # enough points
        else:
            print(f"    Not enough ({n_pts}<{MIN_CONSENSUS_FOR_TPS}), retrying wider...")
