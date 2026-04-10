"""Batch mask-based cell registration for all ELD fixed/moving mask pairs.

Usage:
    python batch_mask_registration.py
    python batch_mask_registration.py --search-radius 60 --min-iou 0.2
    python batch_mask_registration.py --fixed-mask-dir "eld/fixed mask" --moving-mask-dir "eld/moving mask"
"""
from __future__ import annotations

import argparse
import re
import sys
import os
from pathlib import Path
from collections import defaultdict

# Ensure DLL paths for torch
env_prefix = Path(sys.executable).resolve().parent
for p in [env_prefix, env_prefix / "Library" / "bin"]:
    if p.is_dir():
        os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(p))

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from cell_registration.mask_matching import (
    MaskMatchConfig,
    match_masks_pipeline,
    visualize_match_pairs_grid,
)


def _find_pairs(
    fixed_mask_dir: Path,
    moving_mask_dir: Path,
    fixed_feature_dir: Path | None,
    moving_feature_dir: Path | None,
) -> list[dict]:
    """
    Find matching fixed/moving mask pairs based on naming convention.

    Convention:
        Fixed:  {SAMPLE}-{FOV}.tif          e.g. A2-A-1.tif
        Moving: {SAMPLE}-{ROUND}-{FOV}_registered.tif  e.g. A2-B-1_registered.tif

    The fixed mask name prefix (e.g. "A2") and FOV number (e.g. "1") are used
    to match against moving masks from different rounds.
    """
    fixed_pattern = re.compile(r"^(.+?)\.tif$", re.IGNORECASE)
    moving_pattern = re.compile(r"^(.+?)(?:_registered)?\.tif$", re.IGNORECASE)

    fixed_masks = {}
    for f in sorted(fixed_mask_dir.glob("*.tif")):
        m = fixed_pattern.match(f.name)
        if m:
            fixed_masks[m.group(1)] = f

    moving_masks = {}
    for f in sorted(moving_mask_dir.glob("*.tif")):
        m = moving_pattern.match(f.name)
        if m:
            key = m.group(1)
            moving_masks[key] = f

    # Build pairs: for each fixed mask, find all its corresponding moving masks
    # by matching the sample name pattern
    pairs = []
    for fixed_name, fixed_path in sorted(fixed_masks.items()):
        # Parse fixed name: e.g. "A2-A-1" -> sample="A2", fixed_round="A", fov="1"
        parts = fixed_name.split("-")
        if len(parts) < 3:
            continue
        sample = parts[0]
        fov = parts[-1]

        for moving_key, moving_path in sorted(moving_masks.items()):
            # Parse moving name: e.g. "A2-B-1_registered" -> check if same sample + fov
            mparts = moving_key.split("-")
            if len(mparts) < 3:
                continue
            msample = mparts[0]
            mfov = mparts[-1]

            if msample == sample and mfov == fov:
                pair_info = {
                    "fixed_mask": fixed_path,
                    "moving_mask": moving_path,
                    "fixed_name": fixed_name,
                    "moving_name": moving_key,
                    "sample": sample,
                    "fov": fov,
                }

                # Try to find feature files
                if fixed_feature_dir:
                    feat_path = fixed_feature_dir / f"{fixed_name}.csv"
                    if feat_path.exists():
                        pair_info["fixed_features"] = feat_path

                if moving_feature_dir:
                    for suffix in [f"{moving_key}.csv", f"{moving_key}_registered.csv"]:
                        feat_path = moving_feature_dir / suffix
                        if feat_path.exists():
                            pair_info["moving_features"] = feat_path
                            break

                pairs.append(pair_info)

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Batch mask-based cell registration.")
    parser.add_argument(
        "--fixed-mask-dir",
        type=Path,
        default=Path("eld/fixed mask"),
        help="Directory containing fixed mask TIFFs.",
    )
    parser.add_argument(
        "--moving-mask-dir",
        type=Path,
        default=Path("eld/moving mask"),
        help="Directory containing moving mask TIFFs.",
    )
    parser.add_argument(
        "--fixed-feature-dir",
        type=Path,
        default=Path("eld/fixed feature"),
        help="Directory containing fixed feature CSVs.",
    )
    parser.add_argument(
        "--moving-feature-dir",
        type=Path,
        default=Path("eld/moving feature"),
        help="Directory containing moving feature CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mask_registration"),
        help="Output directory.",
    )
    parser.add_argument("--search-radius", type=float, default=50.0)
    parser.add_argument("--min-iou", type=float, default=0.2)
    parser.add_argument("--max-pairs", type=int, default=None, help="Limit number of pairs to process.")
    parser.add_argument("--visualize", action="store_true", help="Generate per-pair visualizations.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = MaskMatchConfig(
        search_radius=args.search_radius,
        min_iou=args.min_iou,
    )

    print("=" * 60)
    print("BATCH MASK-BASED CELL REGISTRATION")
    print("=" * 60)
    print(f"Config: search_radius={config.search_radius}, min_iou={config.min_iou}")

    pairs = _find_pairs(
        args.fixed_mask_dir,
        args.moving_mask_dir,
        args.fixed_feature_dir if args.fixed_feature_dir.exists() else None,
        args.moving_feature_dir if args.moving_feature_dir.exists() else None,
    )

    if args.max_pairs:
        pairs = pairs[:args.max_pairs]

    print(f"Found {len(pairs)} pairs to process.\n")

    summary_rows = []

    for i, pair in enumerate(pairs):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(pairs)}] {pair['fixed_name']} <-> {pair['moving_name']}")
        print(f"{'='*60}")

        pair_dir = args.output_dir / f"{pair['fixed_name']}_vs_{pair['moving_name']}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = match_masks_pipeline(
                fixed_mask_path=pair["fixed_mask"],
                moving_mask_path=pair["moving_mask"],
                fixed_features_path=pair.get("fixed_features"),
                moving_features_path=pair.get("moving_features"),
                output_csv=pair_dir / "matches.csv",
                config=config,
                verbose=True,
            )

            row = {
                "fixed_name": pair["fixed_name"],
                "moving_name": pair["moving_name"],
                "sample": pair["sample"],
                "fov": pair["fov"],
                "n_fixed": result.n_fixed,
                "n_moving": result.n_moving,
                "n_matched": len(result.matches),
                "match_rate_fixed": len(result.matches) / max(result.n_fixed, 1),
                "match_rate_moving": len(result.matches) / max(result.n_moving, 1),
            }

            if not result.matches.empty:
                row["iou_mean"] = result.matches["iou"].mean()
                row["iou_median"] = result.matches["iou"].median()
                row["iou_gt_0.3"] = (result.matches["iou"] > 0.3).sum()
                row["iou_gt_0.5"] = (result.matches["iou"] > 0.5).sum()

                if args.visualize:
                    from tifffile import imread
                    fm = imread(str(pair["fixed_mask"])).astype(np.int32)
                    mm = imread(str(pair["moving_mask"])).astype(np.int32)
                    visualize_match_pairs_grid(
                        fm, mm, result.matches,
                        output_path=pair_dir / "best_matches.png",
                        n_pairs=12, sort_by="iou", ascending=False,
                    )
            else:
                row["iou_mean"] = 0.0
                row["iou_median"] = 0.0
                row["iou_gt_0.3"] = 0
                row["iou_gt_0.5"] = 0

            summary_rows.append(row)

        except Exception as e:
            print(f"  ERROR: {e}")
            summary_rows.append({
                "fixed_name": pair["fixed_name"],
                "moving_name": pair["moving_name"],
                "error": str(e),
            })

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "batch_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n\n{'='*60}")
    print("BATCH SUMMARY")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
