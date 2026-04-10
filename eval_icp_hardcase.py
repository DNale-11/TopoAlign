"""Evaluate ICP registration on the A2-3 hard case with full metrics."""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, ".")

from cell_registration.io_utils import load_image, infer_image_mode, project_intensity_max
from cell_registration.segmentation import CellposeSegmenter
from cell_registration.config import DEFAULT_CELLPOSE_CONFIG, DEFAULT_FEATURE_CONFIG
from cell_registration.features import compute_cell_features
from cell_registration.robust_alignment import perform_global_registration
from cell_registration.icp_registration import ICPConfig, run_icp_pipeline
from cell_registration.evaluation import (
    compute_overlap_metrics,
    compute_object_registration_metrics,
)
from cell_registration.registration import RigidTransform
from skimage.transform import AffineTransform, warp

# Paths
fixed_path = Path(r"F:\programme\nxy\Cell registration\benchmark\unregistration\A2-3\fixed\A2-A-3.tif")
moving_path = Path(r"F:\programme\nxy\Cell registration\benchmark\unregistration\A2-3\moving\A2-C-3.tif")

print("="*70)
print("  Evaluating ICP Registration: A2-C-3 → A2-A-3 (Hard Case)")
print("="*70)

# 1. Load images
print("\n[1/5] Loading images...")
img_fixed = load_image(fixed_path)
img_moving = load_image(moving_path)

mode = infer_image_mode(img_fixed)
print(f"  Image mode: {mode}")

# 2. Segment
print("\n[2/5] Running Cellpose segmentation...")
segmenter = CellposeSegmenter(DEFAULT_CELLPOSE_CONFIG)

if mode == "3d_zstack":
    _, mask_fixed, _, _ = segmenter.segment_zstack(img_fixed)
    _, mask_moving, _, _ = segmenter.segment_zstack(img_moving)
    overlay_fixed = project_intensity_max(img_fixed)
    overlay_moving = project_intensity_max(img_moving)
else:
    mask_fixed, _, _ = segmenter.segment_array(img_fixed)
    mask_moving, _, _ = segmenter.segment_array(img_moving)
    overlay_fixed = img_fixed
    overlay_moving = img_moving

print(f"  Fixed: {mask_fixed.shape}, {len(np.unique(mask_fixed))-1} cells")
print(f"  Moving: {mask_moving.shape}, {len(np.unique(mask_moving))-1} cells")

# 3. Features + Global alignment + ICP
print("\n[3/5] Extracting features & running ICP...")
feats_fixed = compute_cell_features(mask_fixed, DEFAULT_FEATURE_CONFIG).reset_index(drop=True)
feats_moving = compute_cell_features(mask_moving, DEFAULT_FEATURE_CONFIG).reset_index(drop=True)

# Global coarse alignment
global_transform = perform_global_registration(feats_fixed, feats_moving)
if global_transform is not None:
    rot = getattr(global_transform, "rotation", np.nan)
    trans = getattr(global_transform, "translation", np.array([np.nan, np.nan]))
    if not (np.isfinite(rot) and np.all(np.isfinite(trans))):
        global_transform = None

print(f"  Coarse alignment: {'OK' if global_transform is not None else 'FAILED'}")

# ICP
icp_config = ICPConfig(
    max_iterations=80,                # more iterations for hard case
    strict_match_threshold=10.0,
    w_spatial_init=0.3,
    w_spatial_final=1.0,
    w_feature_init=1.0,
    w_feature_final=0.3,
    transform_type="rigid",
)

transform, matches = run_icp_pipeline(
    feats_fixed, feats_moving,
    initial_transform=global_transform,
    config=icp_config,
    verbose=True,
)

# 4. Compute all metrics
print("\n[4/5] Computing evaluation metrics...")

# Build affine transform for warping
affine = transform.as_affine_transform()

# Warp moving image onto fixed space
moving_2d = overlay_moving if overlay_moving.ndim == 2 else overlay_moving
fixed_2d = overlay_fixed if overlay_fixed.ndim == 2 else overlay_fixed

warped_moving = warp(
    moving_2d.astype(np.float64),
    inverse_map=AffineTransform(matrix=np.linalg.inv(transform.matrix)).inverse,
    output_shape=fixed_2d.shape[:2],
    preserve_range=True,
    order=1,
).astype(moving_2d.dtype)

# Valid mask: where the warped image actually has content
valid_mask = warped_moving > 0

# Pixel-level overlap metrics
print("\n--- Pixel-Level Metrics (Image Similarity) ---")
try:
    pixel_metrics = compute_overlap_metrics(warped_moving, fixed_2d, valid_mask)
    for k, v in pixel_metrics.items():
        print(f"  {k}: {v:.4f}")
except Exception as e:
    print(f"  Could not compute pixel metrics: {e}")
    pixel_metrics = {}

# Object-level metrics (F1, precision, recall, TRE)
print("\n--- Object-Level Metrics (Cell Matching) ---")
try:
    obj_metrics, match_table = compute_object_registration_metrics(
        moving_mask=mask_moving,
        fixed_mask=mask_fixed,
        moving_features=feats_moving,
        fixed_features=feats_fixed,
        transform=affine,
        valid_mask=valid_mask,
        instance_iou_threshold=0.3,
        return_match_table=True,
    )
    for k, v in obj_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
except Exception as e:
    print(f"  Could not compute object metrics: {e}")
    import traceback
    traceback.print_exc()
    obj_metrics = {}
    match_table = pd.DataFrame()

# 5. Summary
print("\n" + "="*70)
print("  SUMMARY: ICP Registration Quality")
print("="*70)

if not matches.empty:
    print(f"\n  Matched point pairs:     {len(matches)}")
    print(f"  Mean centroid L2:        {matches['residual_px'].mean():.2f} px")
    print(f"  Median centroid L2:      {matches['residual_px'].median():.2f} px")
    print(f"  P95 centroid L2:         {np.percentile(matches['residual_px'], 95):.2f} px")

if pixel_metrics:
    print(f"\n  SSIM:                    {pixel_metrics.get('SSIM', float('nan')):.4f}")
    print(f"  PSNR:                    {pixel_metrics.get('PSNR', float('nan')):.2f} dB")
    print(f"  NCC:                     {pixel_metrics.get('NCC', float('nan')):.4f}")
    print(f"  NRMSE:                   {pixel_metrics.get('NRMSE', float('nan')):.4f}")

if obj_metrics:
    print(f"\n  F1 Score:                {obj_metrics.get('match_f1', float('nan')):.4f}")
    print(f"  Precision:               {obj_metrics.get('match_precision', float('nan')):.4f}")
    print(f"  Recall:                  {obj_metrics.get('match_recall', float('nan')):.4f}")
    print(f"  Matched Cells:           {obj_metrics.get('matched_cells', 0)}")
    print(f"  Eligible Moving Cells:   {obj_metrics.get('eligible_moving_cells', 0)}")
    print(f"  Eligible Fixed Cells:    {obj_metrics.get('eligible_fixed_cells', 0)}")
    print(f"  Matched Mean IoU:        {obj_metrics.get('matched_mean_iou', float('nan')):.4f}")
    print(f"  TRE Mean:                {obj_metrics.get('tre_mean_px', float('nan')):.2f} px")
    print(f"  TRE Median:              {obj_metrics.get('tre_median_px', float('nan')):.2f} px")
    print(f"  TRE P95:                 {obj_metrics.get('tre_p95_px', float('nan')):.2f} px")
    print(f"  Mask Dice:               {obj_metrics.get('mask_dice', float('nan')):.4f}")
    print(f"  Mask IoU:                {obj_metrics.get('mask_iou', float('nan')):.4f}")

# Save match table
if not match_table.empty:
    out_path = Path("outputs/icp_evaluation_match_table.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    match_table.to_csv(out_path, index=False)
    print(f"\n  Match table saved to: {out_path}")

print("\n" + "="*70)
