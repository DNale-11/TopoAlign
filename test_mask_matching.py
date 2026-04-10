"""Test mask-based cell matching on a single pair of ELD masks.

Usage:
    python test_mask_matching.py
    python test_mask_matching.py --fixed-mask "eld/fixed mask/A2-A-1.tif" --moving-mask "eld/moving mask/A2-B-1_registered.tif"
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# Ensure DLL paths for torch
env_prefix = Path(sys.executable).resolve().parent
for p in [env_prefix, env_prefix / "Library" / "bin"]:
    if p.is_dir():
        os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(p))

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from cell_registration.mask_matching import (
    MaskMatchConfig,
    match_masks_pipeline,
    visualize_matches,
    visualize_match_pairs_grid,
)


def main():
    parser = argparse.ArgumentParser(description="Test mask-based cell matching.")
    parser.add_argument(
        "--fixed-mask",
        type=Path,
        default=Path("eld/fixed mask/A2-A-1.tif"),
        help="Path to fixed mask TIFF.",
    )
    parser.add_argument(
        "--moving-mask",
        type=Path,
        default=Path("eld/moving mask/A2-B-1_registered.tif"),
        help="Path to moving mask TIFF.",
    )
    parser.add_argument(
        "--fixed-features",
        type=Path,
        default=Path("eld/fixed feature/A2-A-1.csv"),
        help="Path to fixed features CSV.",
    )
    parser.add_argument(
        "--moving-features",
        type=Path,
        default=Path("eld/moving feature/A2-B-1_registered.csv"),
        help="Path to moving features CSV.",
    )
    parser.add_argument(
        "--search-radius",
        type=float,
        default=50.0,
        help="Search radius in pixels.",
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.1,
        help="Minimum IoU threshold.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mask_matching_test"),
        help="Output directory for results.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = MaskMatchConfig(
        search_radius=args.search_radius,
        min_iou=args.min_iou,
    )

    print("=" * 60)
    print("MASK-BASED CELL MATCHING TEST")
    print("=" * 60)
    print(f"Fixed mask:  {args.fixed_mask}")
    print(f"Moving mask: {args.moving_mask}")
    print(f"Config:      search_radius={config.search_radius}, min_iou={config.min_iou}")
    print()

    # Check feature files exist
    fixed_feat_path = args.fixed_features if args.fixed_features.exists() else None
    moving_feat_path = args.moving_features if args.moving_features.exists() else None

    result = match_masks_pipeline(
        fixed_mask_path=args.fixed_mask,
        moving_mask_path=args.moving_mask,
        fixed_features_path=fixed_feat_path,
        moving_features_path=moving_feat_path,
        output_csv=args.output_dir / "matches.csv",
        config=config,
        verbose=True,
    )

    if result.matches.empty:
        print("\nERROR: No matches found! Try increasing search_radius or decreasing min_iou.")
        return

    # Print IoU distribution
    print("\n" + "=" * 60)
    print("IoU DISTRIBUTION")
    print("=" * 60)
    matches = result.matches
    print(matches[["cell_id_fixed", "cell_id_moving", "iou", "hu_dist", "area_ratio", "score", "centroid_dist"]].describe())

    print("\nIoU histogram:")
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    counts, _ = np.histogram(matches["iou"], bins=bins)
    for i in range(len(bins) - 1):
        bar = "█" * counts[i]
        print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}): {counts[i]:4d} {bar}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    from tifffile import imread

    fixed_mask = imread(str(args.fixed_mask)).astype(np.int32)
    moving_mask = imread(str(args.moving_mask)).astype(np.int32)

    visualize_matches(
        fixed_mask, moving_mask, matches,
        output_path=args.output_dir / "match_overview.png",
        max_pairs=100,
    )

    visualize_match_pairs_grid(
        fixed_mask, moving_mask, matches,
        output_path=args.output_dir / "best_matches.png",
        n_pairs=20,
        sort_by="iou",
        ascending=False,
    )

    visualize_match_pairs_grid(
        fixed_mask, moving_mask, matches,
        output_path=args.output_dir / "worst_matches.png",
        n_pairs=20,
        sort_by="iou",
        ascending=True,
    )

    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
