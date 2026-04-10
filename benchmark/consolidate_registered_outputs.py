"""Consolidate registered masks and registered features into flat output folders."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from tifffile import imread

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
NAPARI_SRC = PROJECT_ROOT / "napari-cell-registration" / "src"
if str(NAPARI_SRC) not in sys.path:
    sys.path.insert(0, str(NAPARI_SRC))

from napari_cell_registration.core.config import CellFeaturesConfig
from napari_cell_registration.core.features import compute_cell_features


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten registered masks/features into shared folders.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "unregistration_full",
        help="Benchmark result root that contains per-pair output directories.",
    )
    parser.add_argument(
        "--mask-folder-name",
        default="moving",
        help="Flat folder name used for consolidated registered masks.",
    )
    parser.add_argument(
        "--feature-folder-name",
        default="moving feature",
        help="Flat folder name used for consolidated registered feature CSVs.",
    )
    return parser.parse_args()


def _prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()


def _features_for_export(features: pd.DataFrame) -> pd.DataFrame:
    exported = features.copy()
    if "centroid_x" in exported.columns and "x" not in exported.columns:
        exported["x"] = exported["centroid_x"]
    if "centroid_y" in exported.columns and "y" not in exported.columns:
        exported["y"] = exported["centroid_y"]
    return exported


def _discover_pair_dirs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("diagnostics.json"))


def main() -> None:
    args = _parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input result directory not found: {input_dir}")

    moving_dir = input_dir / str(args.mask_folder_name)
    moving_feature_dir = input_dir / str(args.feature_folder_name)
    _prepare_output_dir(moving_dir)
    _prepare_output_dir(moving_feature_dir)

    pair_dirs = _discover_pair_dirs(input_dir)
    if not pair_dirs:
        raise RuntimeError(f"No pair directories found under {input_dir}")

    feature_config = CellFeaturesConfig()
    exported = 0
    for pair_dir in pair_dirs:
        base_name = pair_dir.name
        registered_mask_path = pair_dir / "registered_mask.tif"
        if not registered_mask_path.is_file():
            raise FileNotFoundError(f"Missing registered mask: {registered_mask_path}")

        consolidated_mask_path = moving_dir / f"{base_name}.tif"
        shutil.copy2(registered_mask_path, consolidated_mask_path)

        registered_mask = imread(str(registered_mask_path))
        registered_features = _features_for_export(compute_cell_features(registered_mask, feature_config))

        pair_registered_features_path = pair_dir / "registered_features.csv"
        registered_features.to_csv(pair_registered_features_path, index=False)
        consolidated_feature_path = moving_feature_dir / f"{base_name}.csv"
        registered_features.to_csv(consolidated_feature_path, index=False)
        exported += 1

    print(f"Exported pairs: {exported}")
    print(f"Mask folder: {moving_dir}")
    print(f"Feature folder: {moving_feature_dir}")


if __name__ == "__main__":
    main()
