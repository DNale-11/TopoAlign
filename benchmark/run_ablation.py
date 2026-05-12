"""
Ablation study runner — parameterized wrapper around run_tps_batch.run_widget_pipeline.
Each experiment writes results to benchmark/results/<experiment_name>/.
"""
from __future__ import annotations
import os, sys, time, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_tl = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_tl):
    os.environ["PATH"] = _tl + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"): os.add_dll_directory(_tl)
try: import torch
except: pass

from pathlib import Path
import numpy as np
import pandas as pd
from tifffile import imread, imwrite

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "napari-cell-registration" / "src"))

from napari_cell_registration.core import (
    CellposeSegmenter, CellposeConfig, CellFeaturesConfig,
    compute_cell_features, MatchingConfig, MIN_MATCHES_FOR_REFINEMENT,
    two_stage_match_cells, greedy_match_cells,
    estimate_rigid_transform_from_matches,
    estimate_rigid_transform_from_matches_ransac,
    rigid_transform_to_affine,
    compute_match_residuals,
)
from napari_cell_registration.core.matching import (
    apply_transform_to_features,
    _filter_candidate_matches_by_hard_constraints,
)
from napari_cell_registration.core.point_registration import (
    fit_tps_from_matches, warp_image_with_tps,
)
from cell_registration.evaluation import compute_prewarped_registration_metrics
from run_tps_batch import discover_pairs

# ── Fixed constants ──
IOU_THRESHOLD = 0.3
CELLPOSE_CFG = CellposeConfig(model_type="cpsam", diameter=15, flow_threshold=-2.0, cellprob_threshold=1.0, min_size=5)
FEATURE_CFG = CellFeaturesConfig(min_area=None, max_area=None, topology_neighbor_k=5)
RESULTS_ROOT = PROJECT_ROOT / "benchmark" / "results"

# ── Default parameters (baseline) ──
DEFAULTS = dict(
    top_k=320,
    max_dist=100,
    position_weight=1.0,
    ransac_residual_threshold=2.0,
    ransac_max_trials=1000,
    patch_grid=4,
    rematch_window_factor=0.6,
    orientation_max_deg=5.0,
    orientation_min_ecc=0.15,
    consensus_angle_deg=5.0,
    consensus_length_tol=5.0,
    min_consensus_for_tps=50,
    min_matches_for_refinement=5,
    # ablation flags
    use_retry=True,
    use_guided_rematch=True,
    use_consensus=True,
    use_tps=True,        # False = rigid-only for all
)


def run_pipeline(img1, mask1, feats1, img2, mask2, feats2, *, params: dict):
    """Parameterized pipeline mirroring the widget logic."""
    p = {**DEFAULTS, **params}
    TOP_K = p["top_k"]
    MAX_DIST = p["max_dist"]
    PW = p["position_weight"]
    RRT = p["ransac_residual_threshold"]
    RMT = p["ransac_max_trials"]
    PG = p["patch_grid"]
    RWF = p["rematch_window_factor"]
    O_DEG = p["orientation_max_deg"]
    O_ECC = p["orientation_min_ecc"]
    C_ANG = p["consensus_angle_deg"]
    C_LEN = p["consensus_length_tol"]
    MIN_C = p["min_consensus_for_tps"]
    MIN_M = p["min_matches_for_refinement"]

    window_scales = (1.0, 2.0) if p["use_retry"] else (1.0,)
    diag = {}

    for _ws in window_scales:
        eff_dist = MAX_DIST * _ws
        if _ws > 1.0:
            print(f"  Retrying with wider window: {eff_dist:.0f}px")

        # Step 1: Two-stage matching
        match_result = two_stage_match_cells(
            feats1, feats2, mask1.shape,
            feature_weight=1.0, topology_weight=0.0, position_weight=PW,
            top_k=TOP_K, distance_threshold=None, spatial_window_size=float(eff_dist),
            min_cells_for_two_stage=10, coarse_top_k=max(24, TOP_K),
            coarse_distance_threshold=2.0, coarse_matching_mode="morphology_guided",
            coarse_allow_scale=False, coarse_prefer_affine=False,
            coarse_residual_threshold=max(5.0, RRT * 2.0),
            coarse_max_trials=min(max(RMT, 200), 2000),
        )
        matches = match_result.matches.copy()
        diag["initial_matches"] = len(matches)

        # Step 2: RANSAC
        transform, inlier_mask = estimate_rigid_transform_from_matches_ransac(
            feats1, feats2, matches,
            max_trials=RMT, residual_threshold=RRT, min_inliers=MIN_M,
        )
        diag["ransac_inliers"] = int(inlier_mask.sum()) if inlier_mask is not None else 0

        # Step 3: Guided rematch
        if p["use_guided_rematch"] and len(matches) >= MIN_M:
            initial_affine = rigid_transform_to_affine(transform)
            aligned_feats2 = apply_transform_to_features(feats2, initial_affine, mask1.shape)
            rematch_window = max(10.0, float(eff_dist) * RWF)
            if _ws > 1.0:
                rematch_window = max(rematch_window, 160.0)
            rematch_config = MatchingConfig(
                feature_weight=1.0, topology_weight=0.0, position_weight=PW,
                top_k=TOP_K, distance_threshold=None, spatial_window_size=rematch_window,
            )
            rematch_matches = greedy_match_cells(
                feats1, aligned_feats2, rematch_config,
                image_shape=mask1.shape, coverage_patch_grid=PG,
            )
            if len(rematch_matches) >= MIN_M:
                transform = estimate_rigid_transform_from_matches(feats1, feats2, rematch_matches)
                matches = rematch_matches
        diag["after_rematch"] = len(matches)

        # Orientation filter
        n_before_orient = len(matches)
        matches = _filter_candidate_matches_by_hard_constraints(
            matches, max_area_ratio=None, max_aspect_ratio_ratio=None,
            max_orientation_diff_deg=O_DEG, min_orientation_eccentricity=O_ECC,
        )
        if len(matches) < n_before_orient and len(matches) >= MIN_M:
            transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)
        diag["after_orientation"] = len(matches)

        # Step 5: Displacement consensus
        if p["use_consensus"] and len(matches) >= MIN_M:
            affine_transform = rigid_transform_to_affine(transform)
            residuals = compute_match_residuals(feats1, feats2, matches, transform)
            matches = matches.copy()
            matches["residual_px"] = residuals

            pts_f = feats1.iloc[matches["idx1"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
            pts_m = feats2.iloc[matches["idx2"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
            disp_vectors = pts_f - pts_m
            h, w = mask1.shape[:2]
            grid = 4
            px = np.clip(np.floor(pts_f[:, 0] * grid / max(w, 1)).astype(int), 0, grid - 1)
            py = np.clip(np.floor(pts_f[:, 1] * grid / max(h, 1)).astype(int), 0, grid - 1)
            patch_ids = py * grid + px
            n_before = len(matches)
            keep_mask = np.ones(n_before, dtype=bool)
            angle_threshold_deg = C_ANG

            for pid in range(grid * grid):
                in_patch = patch_ids == pid
                n_in = int(in_patch.sum())
                if n_in < 3:
                    continue
                patch_disp = disp_vectors[in_patch]
                patch_indices = np.where(in_patch)[0]
                patch_norms = np.linalg.norm(patch_disp, axis=1)
                cos_thresh = np.cos(np.radians(angle_threshold_deg))
                votes = np.zeros(n_in, dtype=int)
                for a in range(n_in):
                    if patch_norms[a] < 1.0: continue
                    for b in range(n_in):
                        if patch_norms[b] < 1.0: continue
                        cos_ab = np.dot(patch_disp[a], patch_disp[b]) / (patch_norms[a] * patch_norms[b] + 1e-8)
                        if cos_ab >= cos_thresh: votes[a] += 1
                if votes.max() < 2: continue
                consensus_idx = int(np.argmax(votes))
                consensus_disp = patch_disp[consensus_idx]
                consensus_norm = patch_norms[consensus_idx]
                consensus_members = []
                for k in range(n_in):
                    if patch_norms[k] < 1.0:
                        keep_mask[patch_indices[k]] = False
                        continue
                    cos_val = np.dot(patch_disp[k], consensus_disp) / (patch_norms[k] * consensus_norm + 1e-8)
                    if cos_val < cos_thresh:
                        keep_mask[patch_indices[k]] = False
                    else:
                        consensus_members.append(k)
                if len(consensus_members) >= 3:
                    member_lengths = np.array([patch_norms[k] for k in consensus_members])
                    mean_len = float(np.mean(member_lengths))
                    for k in consensus_members:
                        if abs(patch_norms[k] - mean_len) > C_LEN:
                            keep_mask[patch_indices[k]] = False

            n_after = int(keep_mask.sum())
            if n_after >= MIN_M and n_after < n_before:
                matches = matches.loc[keep_mask].reset_index(drop=True)
                transform = estimate_rigid_transform_from_matches(feats1, feats2, matches)
        diag["after_consensus"] = len(matches)

        if len(matches) >= MIN_C:
            break
        print(f"  Only {len(matches)} control points (need {MIN_C})")

    # Rigid fallback with dominant direction
    if len(matches) < MIN_C:
        print(f"  Rigid fallback pass: relaxed matching at {MAX_DIST}px, finding dominant direction...")
        fb_result = two_stage_match_cells(
            feats1, feats2, mask1.shape,
            feature_weight=1.0, topology_weight=0.0, position_weight=PW,
            top_k=TOP_K, distance_threshold=None, spatial_window_size=float(MAX_DIST),
            min_cells_for_two_stage=10, coarse_top_k=max(24, TOP_K),
            coarse_distance_threshold=2.0, coarse_matching_mode="morphology_guided",
            coarse_allow_scale=False, coarse_prefer_affine=False,
            coarse_residual_threshold=max(5.0, RRT * 2.0),
            coarse_max_trials=min(max(RMT, 200), 2000),
        )
        fb_matches = fb_result.matches.copy()
        fb_transform, _ = estimate_rigid_transform_from_matches_ransac(
            feats1, feats2, fb_matches, max_trials=RMT, residual_threshold=RRT, min_inliers=MIN_M,
        )
        if len(fb_matches) >= MIN_M:
            fb_affine = rigid_transform_to_affine(fb_transform)
            fb_aligned = apply_transform_to_features(feats2, fb_affine, mask1.shape)
            fb_cfg = MatchingConfig(
                feature_weight=1.0, topology_weight=0.0, position_weight=PW,
                top_k=TOP_K, distance_threshold=None,
                spatial_window_size=max(10.0, float(MAX_DIST) * RWF),
            )
            fb_rematch = greedy_match_cells(feats1, fb_aligned, fb_cfg, image_shape=mask1.shape, coverage_patch_grid=PG)
            if len(fb_rematch) >= MIN_M:
                fb_transform = estimate_rigid_transform_from_matches(feats1, feats2, fb_rematch)
                fb_matches = fb_rematch

        fb_matches = _filter_candidate_matches_by_hard_constraints(
            fb_matches, max_area_ratio=None, max_aspect_ratio_ratio=None,
            max_orientation_diff_deg=O_DEG, min_orientation_eccentricity=O_ECC,
        )
        if len(fb_matches) >= MIN_M:
            fb_transform = estimate_rigid_transform_from_matches(feats1, feats2, fb_matches)

        if len(fb_matches) >= MIN_M:
            pts_f = feats1.iloc[fb_matches["idx1"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
            pts_m = feats2.iloc[fb_matches["idx2"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
            disps = pts_f - pts_m
            norms = np.linalg.norm(disps, axis=1)
            angles = np.arctan2(disps[:, 1], disps[:, 0])
            angle_tol = np.radians(C_ANG)
            length_tol = C_LEN
            n_fb = len(fb_matches)
            best_group = []
            for i in range(n_fb):
                group = []
                for j in range(n_fb):
                    a_diff = abs(angles[i] - angles[j])
                    a_diff = min(a_diff, 2 * np.pi - a_diff)
                    if a_diff <= angle_tol and abs(norms[i] - norms[j]) <= length_tol:
                        group.append(j)
                if len(group) > len(best_group):
                    best_group = group
            print(f"  Dominant direction group: {len(best_group)}/{n_fb} matches")
            if len(best_group) >= MIN_M:
                fb_matches = fb_matches.iloc[best_group].reset_index(drop=True)
                transform = estimate_rigid_transform_from_matches(feats1, feats2, fb_matches)
                matches = fb_matches

    # Warp
    affine_transform = rigid_transform_to_affine(transform)
    do_tps = p["use_tps"] and len(matches) >= MIN_C

    if do_tps:
        pts_fixed_xy = feats1.iloc[matches["idx1"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
        pts_moving_xy = feats2.iloc[matches["idx2"].to_numpy(dtype=int)][["centroid_x", "centroid_y"]].to_numpy(dtype=float)
        tps = fit_tps_from_matches(
            pts_fixed_xy, pts_moving_xy, output_shape=mask1.shape[:2],
            rigid_transform=affine_transform, regularization=1e-3,
            n_boundary_per_side=4, add_boundary_anchors_flag=True,
        )
        diag["tps_control_points"] = len(pts_fixed_xy)
        diag["warp_mode"] = "tps"
        mask2_warped = warp_image_with_tps(mask2.astype(np.int32), tps, mask1.shape[:2], order=0)
        mask2_registered = np.rint(mask2_warped).astype(np.int32)
    else:
        from scipy.ndimage import map_coordinates
        diag["tps_control_points"] = 0
        diag["warp_mode"] = "rigid_only" if not p["use_tps"] else "rigid_fallback"
        H, W = mask1.shape[:2]
        gy, gx = np.meshgrid(np.arange(H, dtype=float), np.arange(W, dtype=float), indexing="ij")
        queries_xy = np.stack([gx.ravel(), gy.ravel()], axis=1)
        inv_affine = np.linalg.inv(affine_transform)
        ones = np.ones((len(queries_xy), 1))
        src_xy = (inv_affine @ np.hstack([queries_xy, ones]).T).T[:, :2]
        src_row = src_xy[:, 1].reshape(H, W)
        src_col = src_xy[:, 0].reshape(H, W)
        mask2_warped = map_coordinates(mask2.astype(float), [src_row, src_col], order=0, mode="constant", cval=0.0)
        mask2_registered = np.rint(mask2_warped).astype(np.int32)

    return mask2_registered, matches, diag


# ── Experiment definitions ──
EXPERIMENTS = {
    # Ablation studies
    "ablation_rigid_only":       {"use_tps": False},
    "ablation_no_retry":         {"use_retry": False},
    "ablation_no_guided_rematch":{"use_guided_rematch": False, "min_consensus_for_tps": 3},
    "ablation_no_consensus":     {"use_consensus": False},
    # top_k sweep
    "topk_016":  {"top_k": 16},
    "topk_064":  {"top_k": 64},
    "topk_128":  {"top_k": 128},
    "topk_160":  {"top_k": 160},
    # topk_320 = baseline, skip
    "topk_640":  {"top_k": 640},
    # window sweep (TPS-only, no rigid fallback)
    "window_050": {"max_dist": 50, "min_consensus_for_tps": 3},
    # window_100 = baseline, skip
    "window_200": {"max_dist": 200, "min_consensus_for_tps": 3},
    # Patch grid sweep (full pipeline, only change patch coverage)
    "patch_0":  {"patch_grid": 0},
    "patch_2":  {"patch_grid": 2},
    "patch_3":  {"patch_grid": 3},
    # patch_4 = baseline (4x4=16), skip
    "patch_5":  {"patch_grid": 5},
}


def run_experiment(exp_name: str, exp_params: dict):
    output_root = RESULTS_ROOT / exp_name
    output_root.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs()
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"  Params override: {exp_params}")
    print(f"  Pairs: {len(pairs)}")
    print(f"{'='*60}")

    segmenter = CellposeSegmenter(CELLPOSE_CFG)
    seg_cache = {}
    all_results = []
    failures = []

    for i, (case_id, fixed_path, moving_path) in enumerate(pairs):
        pair_name = f"{moving_path.stem}_to_{fixed_path.stem}"
        pair_dir = output_root / case_id / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i+1}/{len(pairs)}] {case_id}: {moving_path.name} -> {fixed_path.name}")

        try:
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

            t0 = time.perf_counter()
            mask2_reg, matches, diag = run_pipeline(
                img1, mask1, feats1, img2, mask2, feats2, params=exp_params,
            )
            elapsed = time.perf_counter() - t0

            metrics = compute_prewarped_registration_metrics(mask2_reg, mask1, instance_iou_threshold=IOU_THRESHOLD)

            warp_mode = diag.get("warp_mode", "unknown")
            print(f"  F1={metrics['match_f1']:.4f}  Matched={metrics['matched_cells']}/{metrics['eligible_fixed_cells']}  "
                  f"warp={warp_mode}  reg_time={elapsed:.1f}s")

            imwrite(str(pair_dir / "registered_mask.tif"), mask2_reg, compression="zlib")
            matches.to_csv(str(pair_dir / "registration_matches.csv"), index=False)

            row = {
                "case_id": case_id, "fixed_name": fixed_path.name, "moving_name": moving_path.name,
                "status": "success", "fixed_cells": len(feats1), "moving_cells": len(feats2),
                "initial_matches": diag.get("initial_matches", 0),
                "after_rematch": diag.get("after_rematch", 0),
                "after_orientation": diag.get("after_orientation", 0),
                "after_consensus": diag.get("after_consensus", 0),
                "tps_control_points": diag.get("tps_control_points", 0),
                "warp_mode": warp_mode,
                "registration_time_sec": round(elapsed, 2),
                **{k: v for k, v in metrics.items()},
            }
            all_results.append(row)

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            failures.append({"case_id": case_id, "pair": pair_name, "error": str(e)})
            all_results.append({
                "case_id": case_id, "fixed_name": fixed_path.name, "moving_name": moving_path.name,
                "status": "failed", "error": str(e), "registration_time_sec": None,
            })

    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(output_root / "summary.csv", index=False)
    with open(output_root / "failures.json", "w") as f:
        json.dump(failures, f, indent=2)

    success_df = summary_df[summary_df["status"] == "success"]
    print(f"\n--- {exp_name} SUMMARY ---")
    print(f"  Success: {len(success_df)}/{len(pairs)}")
    if len(success_df) > 0:
        print(f"  Mean F1:   {success_df['match_f1'].mean():.4f}")
        print(f"  Median F1: {success_df['match_f1'].median():.4f}")
        print(f"  Mean reg time: {success_df['registration_time_sec'].mean():.2f}s")

    case_f1 = success_df.groupby("case_id")["match_f1"].mean().sort_values()
    case_f1.to_csv(output_root / "f1_by_case.csv")

    with open(output_root / "params.json", "w") as f:
        json.dump({**DEFAULTS, **exp_params, "experiment": exp_name}, f, indent=2, default=str)

    return success_df["match_f1"].mean() if len(success_df) > 0 else 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None, help="Run only this experiment name")
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.only:
        if args.only in experiments:
            experiments = {args.only: experiments[args.only]}
        else:
            print(f"Unknown experiment: {args.only}")
            print(f"Available: {list(EXPERIMENTS.keys())}")
            return

    overall = {}
    for name, params in experiments.items():
        mean_f1 = run_experiment(name, params)
        overall[name] = mean_f1

    # Final comparison
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPARISON")
    print("=" * 60)
    print(f"  {'Experiment':<35} {'Mean F1':>10}")
    print(f"  {'-'*35} {'-'*10}")
    # Add baseline reference
    print(f"  {'tps+rigid (baseline)':<35} {'0.8387':>10}")
    for name, f1 in sorted(overall.items(), key=lambda x: -x[1]):
        print(f"  {name:<35} {f1:>10.4f}")


if __name__ == "__main__":
    main()
