"""Batch ICP registration for all image pairs in benchmark/unregistration.

Outputs match the format of run_unregistration_batch.py:
  <case>/<moving_to_fixed>/
    registered_to_fixed.tif   – warped moving image
    registered_mask.tif       – warped moving mask (nearest-neighbor)
    valid_overlap_mask.tif    – valid overlap region
    fixed_mask.tif            – fixed segmentation mask
    moving_mask.tif           – moving segmentation mask (original space)
    fixed_features.csv        – fixed cell features
    moving_features.csv       – moving cell features (original space)
    registered_features.csv   – features of warped mask
    registration_matches.csv  – ICP match table
    diagnostics.json          – full diagnostics
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from skimage.transform import AffineTransform, warp
from tifffile import imread, imwrite

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cell_registration.io_utils import load_image, infer_image_mode, project_intensity_max
from cell_registration.segmentation import CellposeSegmenter
from cell_registration.config import DEFAULT_CELLPOSE_CONFIG, DEFAULT_FEATURE_CONFIG
from cell_registration.features import compute_cell_features
from cell_registration.robust_alignment import perform_global_registration
from cell_registration.icp_registration import ICPConfig, run_icp_pipeline
from cell_registration.evaluation import compute_object_registration_metrics

# ── Configuration ──────────────────────────────────────────────────────
BASE_DIR = Path(r"F:\programme\nxy\Cell registration\benchmark\unregistration")
OUTPUT_ROOT = Path(r"F:\programme\nxy\Cell registration\benchmark\icp_results")

ICP_CONFIG = ICPConfig(
    max_iterations=80,
    strict_match_threshold=10.0,
    w_spatial_init=0.3,
    w_spatial_final=1.0,
    w_feature_init=1.0,
    w_feature_final=0.3,
    transform_type="rigid",
)

OBJECT_EVAL_PARAMS = {
    "instance_iou_threshold": 0.3,
    "min_valid_instance_area_px": 20,
    "min_valid_fraction": 0.5,
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(o):
        if isinstance(o, Path): return str(o)
        if hasattr(o, "tolist"): return o.tolist()
        if hasattr(o, "item"): return o.item()
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=default)


def _features_for_export(features: pd.DataFrame) -> pd.DataFrame:
    exported = features.copy()
    if "centroid_x" in exported.columns and "x" not in exported.columns:
        exported["x"] = exported["centroid_x"]
    if "centroid_y" in exported.columns and "y" not in exported.columns:
        exported["y"] = exported["centroid_y"]
    return exported


def segment_image(segmenter, img):
    mode = infer_image_mode(img)
    if mode == "3d_zstack":
        _, mask, _, _ = segmenter.segment_zstack(img)
        overlay = project_intensity_max(img)
    else:
        mask, _, _ = segmenter.segment_array(img)
        overlay = img
    return mask, overlay


def run_single_pair(
    fixed_path: Path,
    moving_path: Path,
    output_dir: Path,
    segmenter: CellposeSegmenter,
    icp_config: ICPConfig,
    # Cached segmentations
    fixed_mask: np.ndarray,
    fixed_overlay: np.ndarray,
    feats_fixed: pd.DataFrame,
) -> dict:
    pair_name = f"{moving_path.stem}_to_{fixed_path.stem}"
    pair_dir = output_dir / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # Load & segment moving
    img_moving = load_image(moving_path)
    mask_moving, overlay_moving = segment_image(segmenter, img_moving)
    feats_moving = compute_cell_features(mask_moving, DEFAULT_FEATURE_CONFIG).reset_index(drop=True)

    n_fixed = len(feats_fixed)
    n_moving = len(feats_moving)
    print(f"    Cells: fixed={n_fixed}, moving={n_moving}")

    # ── Global coarse alignment with sanity check ──
    # Adjacent tissue sections should NOT have large rotations (>15°).
    # If RANSAC returns a big rotation it found wrong correspondences → reject.
    # Translation is NOT limited: real sections can have arbitrarily large shifts.
    MAX_GLOBAL_ROT_DEG = 15.0

    global_transform = perform_global_registration(feats_fixed, feats_moving)
    if global_transform is not None:
        try:
            m = np.asarray(global_transform.params, dtype=float)
            angle_deg = math.degrees(math.atan2(float(m[1, 0]), float(m[0, 0])))
            tx, ty = float(m[0, 2]), float(m[1, 2])
            trans_px = math.sqrt(tx**2 + ty**2)
            if abs(angle_deg) > MAX_GLOBAL_ROT_DEG:
                print(f"    ⚠ Global registration rejected "
                      f"(angle={angle_deg:.1f}°, trans={trans_px:.1f}px) → identity fallback")
                global_transform = None
            else:
                print(f"    ✔ Global registration accepted "
                      f"(angle={angle_deg:.1f}°, trans={trans_px:.1f}px)")
        except Exception:
            global_transform = None

    # ICP
    transform, matches = run_icp_pipeline(
        feats_fixed, feats_moving,
        initial_transform=global_transform,
        config=icp_config,
        verbose=False,
    )

    elapsed = time.perf_counter() - t0

    # ── Build affine and warp ──
    affine = transform.as_affine_transform()
    fixed_shape = fixed_mask.shape[:2]
    moving_shape = mask_moving.shape[:2]
    # skimage.warp(inverse_map=AffineTransform) automatically calls .inverse internally,
    # so we pass the FORWARD transform (moving→fixed) and let skimage invert it.
    fwd_affine = AffineTransform(matrix=transform.matrix)

    # Warp moving image → fixed space
    registered_img = warp(
        overlay_moving.astype(np.float64),
        inverse_map=fwd_affine,
        output_shape=fixed_shape,
        preserve_range=True,
        order=1,
    )
    if np.issubdtype(overlay_moving.dtype, np.integer):
        info = np.iinfo(overlay_moving.dtype)
        registered_img = np.clip(np.rint(registered_img), info.min, info.max).astype(overlay_moving.dtype)
    else:
        registered_img = registered_img.astype(overlay_moving.dtype)

    # Warp moving mask → fixed space (nearest-neighbor)
    registered_mask = warp(
        mask_moving.astype(np.float64),
        inverse_map=fwd_affine,
        output_shape=fixed_shape,
        preserve_range=True,
        order=0,
    )
    registered_mask = np.rint(registered_mask).astype(np.int32)

    # Valid overlap mask
    ones = np.ones(moving_shape, dtype=np.float64)
    valid_mask = warp(
        ones,
        inverse_map=fwd_affine,
        output_shape=fixed_shape,
        preserve_range=True,
        order=0,
    )
    valid_mask = (valid_mask > 0.5).astype(np.uint8)

    # Registered features (from warped mask)
    registered_features = compute_cell_features(registered_mask, DEFAULT_FEATURE_CONFIG).reset_index(drop=True)

    # ── Save all artifacts ──
    imwrite(str(pair_dir / "registered_to_fixed.tif"), registered_img)
    imwrite(str(pair_dir / "registered_mask.tif"), registered_mask)
    imwrite(str(pair_dir / "valid_overlap_mask.tif"), valid_mask)
    imwrite(str(pair_dir / "fixed_mask.tif"), fixed_mask.astype(np.int32))
    imwrite(str(pair_dir / "moving_mask.tif"), mask_moving.astype(np.int32))
    _features_for_export(feats_fixed).to_csv(pair_dir / "fixed_features.csv", index=False)
    _features_for_export(feats_moving).to_csv(pair_dir / "moving_features.csv", index=False)
    _features_for_export(registered_features).to_csv(pair_dir / "registered_features.csv", index=False)
    matches.to_csv(pair_dir / "registration_matches.csv", index=False)

    # ── Object-level evaluation ──
    metrics = {"case": fixed_path.parent.parent.name, "fixed": fixed_path.stem, "moving": moving_path.stem,
               "pair": pair_name, "n_cells_fixed": n_fixed, "n_cells_moving": n_moving,
               "icp_matched_pairs": len(matches), "time_seconds": round(elapsed, 1)}

    if not matches.empty and "spatial_dist_px" in matches.columns:
        dists = matches["spatial_dist_px"].to_numpy()
        metrics["icp_mean_L2"] = round(float(np.mean(dists)), 2)
        metrics["icp_median_L2"] = round(float(np.median(dists)), 2)
        metrics["icp_p95_L2"] = round(float(np.percentile(dists, 95)), 2)

    try:
        obj_metrics, match_table = compute_object_registration_metrics(
            moving_mask=mask_moving, fixed_mask=fixed_mask,
            moving_features=feats_moving, fixed_features=feats_fixed,
            transform=affine, valid_mask=valid_mask,
            instance_iou_threshold=OBJECT_EVAL_PARAMS["instance_iou_threshold"],
            min_valid_instance_area_px=OBJECT_EVAL_PARAMS["min_valid_instance_area_px"],
            min_valid_fraction=OBJECT_EVAL_PARAMS["min_valid_fraction"],
            return_match_table=True,
        )
        metrics.update({
            "F1": round(obj_metrics.get("match_f1", float("nan")), 4),
            "precision": round(obj_metrics.get("match_precision", float("nan")), 4),
            "recall": round(obj_metrics.get("match_recall", float("nan")), 4),
            "matched_cells": obj_metrics.get("matched_cells", 0),
            "eligible_moving": obj_metrics.get("eligible_moving_cells", 0),
            "eligible_fixed": obj_metrics.get("eligible_fixed_cells", 0),
            "matched_mean_iou": round(obj_metrics.get("matched_mean_iou", float("nan")), 4),
            "TRE_mean_px": round(obj_metrics.get("tre_mean_px", float("nan")), 2),
            "TRE_median_px": round(obj_metrics.get("tre_median_px", float("nan")), 2),
            "TRE_p95_px": round(obj_metrics.get("tre_p95_px", float("nan")), 2),
            "mask_dice": round(obj_metrics.get("mask_dice", float("nan")), 4),
            "mask_iou": round(obj_metrics.get("mask_iou", float("nan")), 4),
        })
        if not match_table.empty:
            match_table.to_csv(pair_dir / "object_match_table.csv", index=False)
    except Exception as e:
        print(f"    Warning: object metrics failed: {e}")

    # ── Save diagnostics JSON ──
    diag = {
        "case_id": metrics["case"],
        "fixed_path": str(fixed_path),
        "moving_path": str(moving_path),
        "icp_config": {
            "max_iterations": icp_config.max_iterations,
            "strict_match_threshold": icp_config.strict_match_threshold,
            "w_spatial_init": icp_config.w_spatial_init,
            "w_spatial_final": icp_config.w_spatial_final,
            "w_feature_init": icp_config.w_feature_init,
            "w_feature_final": icp_config.w_feature_final,
            "transform_type": icp_config.transform_type,
        },
        "cellpose_config": {
            "diameter": DEFAULT_CELLPOSE_CONFIG.diameter,
            "flow_threshold": DEFAULT_CELLPOSE_CONFIG.flow_threshold,
            "cellprob_threshold": DEFAULT_CELLPOSE_CONFIG.cellprob_threshold,
            "min_size": DEFAULT_CELLPOSE_CONFIG.min_size,
        },
        "transform_matrix": transform.matrix.tolist(),
        "metrics": metrics,
        "timestamp": datetime.now().astimezone().isoformat(),
    }
    _write_json(pair_dir / "diagnostics.json", diag)

    f1 = metrics.get("F1", float("nan"))
    tre = metrics.get("TRE_mean_px", float("nan"))
    n_match = metrics.get("icp_matched_pairs", 0)
    mean_l2 = metrics.get("icp_mean_L2", float("nan"))
    print(f"    ✓ {pair_name}: {n_match} pairs, mean_L2={mean_l2}px, F1={f1:.3f}, TRE={tre:.1f}px  ({elapsed:.0f}s)")

    return metrics


def main():
    print("=" * 70)
    print("  Batch ICP Registration (Full Artifact Export)")
    print(f"  Input: {BASE_DIR}")
    print(f"  Output: {OUTPUT_ROOT}")
    print(f"  Cellpose: diameter={DEFAULT_CELLPOSE_CONFIG.diameter}, "
          f"flow_threshold={DEFAULT_CELLPOSE_CONFIG.flow_threshold}, "
          f"cellprob_threshold={DEFAULT_CELLPOSE_CONFIG.cellprob_threshold}, "
          f"min_size={DEFAULT_CELLPOSE_CONFIG.min_size}")
    print("=" * 70)

    segmenter = CellposeSegmenter(DEFAULT_CELLPOSE_CONFIG)
    all_metrics: list[dict] = []
    errors: list[str] = []

    cases = sorted([d for d in BASE_DIR.iterdir() if d.is_dir()])
    total_pairs = 0
    for case_dir in cases:
        fixed_dir = case_dir / "fixed"
        moving_dir = case_dir / "moving"
        if not fixed_dir.exists() or not moving_dir.exists():
            continue
        moving_files = sorted(moving_dir.glob("*.tif"))
        total_pairs += len(moving_files)

    print(f"\n  Found {len(cases)} cases, {total_pairs} registration pairs total\n")

    pair_idx = 0
    for case_dir in cases:
        fixed_dir = case_dir / "fixed"
        moving_dir = case_dir / "moving"
        if not fixed_dir.exists() or not moving_dir.exists():
            continue

        fixed_files = sorted(fixed_dir.glob("*.tif"))
        moving_files = sorted(moving_dir.glob("*.tif"))
        if not fixed_files:
            continue
        fixed_path = fixed_files[0]

        # Segment fixed once per case (cache)
        print(f"\n[Case {case_dir.name}] fixed={fixed_path.stem}, {len(moving_files)} moving images")
        img_fixed = load_image(fixed_path)
        mask_fixed, overlay_fixed = segment_image(segmenter, img_fixed)
        feats_fixed = compute_cell_features(mask_fixed, DEFAULT_FEATURE_CONFIG).reset_index(drop=True)
        print(f"  Fixed: {len(feats_fixed)} cells")

        output_dir = OUTPUT_ROOT / case_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        for moving_path in moving_files:
            pair_idx += 1
            print(f"\n  [{pair_idx}/{total_pairs}] {moving_path.stem} → {fixed_path.stem}")
            try:
                metrics = run_single_pair(
                    fixed_path, moving_path, output_dir,
                    segmenter, ICP_CONFIG,
                    mask_fixed, overlay_fixed, feats_fixed,
                )
                all_metrics.append(metrics)
            except Exception as e:
                error_msg = f"{case_dir.name}/{moving_path.stem}: {e}"
                errors.append(error_msg)
                print(f"    ✗ ERROR: {e}")
                import traceback
                traceback.print_exc()

    # ── Save summary CSV ──
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "icp_batch_results.csv"
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        df.to_csv(summary_path, index=False)
        print(f"\n{'='*70}")
        print(f"  BATCH COMPLETE: {len(all_metrics)}/{total_pairs} pairs succeeded")
        if errors:
            print(f"  ERRORS: {len(errors)}")
            for e in errors:
                print(f"    - {e}")
        print(f"  Results saved to: {summary_path}")
        print(f"{'='*70}")

        # Aggregate stats
        print(f"\n  Aggregate Statistics:")
        for col in ["F1", "precision", "recall", "TRE_mean_px", "icp_mean_L2", "mask_dice", "matched_mean_iou"]:
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals) > 0:
                    print(f"    {col:20s}: mean={vals.mean():.4f}  median={vals.median():.4f}  "
                          f"min={vals.min():.4f}  max={vals.max():.4f}")

    _write_json(OUTPUT_ROOT / "run_manifest.json", {
        "started_at": datetime.now().astimezone().isoformat(),
        "input_root": str(BASE_DIR),
        "output_dir": str(OUTPUT_ROOT),
        "total_pairs": total_pairs,
        "success_count": len(all_metrics),
        "failure_count": len(errors),
        "icp_config": {
            "max_iterations": ICP_CONFIG.max_iterations,
            "strict_match_threshold": ICP_CONFIG.strict_match_threshold,
            "w_spatial_init": ICP_CONFIG.w_spatial_init,
            "w_spatial_final": ICP_CONFIG.w_spatial_final,
            "w_feature_init": ICP_CONFIG.w_feature_init,
            "w_feature_final": ICP_CONFIG.w_feature_final,
        },
        "cellpose_config": {
            "diameter": DEFAULT_CELLPOSE_CONFIG.diameter,
            "flow_threshold": DEFAULT_CELLPOSE_CONFIG.flow_threshold,
            "cellprob_threshold": DEFAULT_CELLPOSE_CONFIG.cellprob_threshold,
            "min_size": DEFAULT_CELLPOSE_CONFIG.min_size,
        },
    })


if __name__ == "__main__":
    main()
