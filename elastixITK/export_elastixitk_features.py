"""Segment elastixITK fixed/moving images and export masks plus cell features."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _prepare_torch_runtime() -> None:
    env_prefix = Path(sys.executable).resolve().parent
    candidate_dirs = [
        env_prefix,
        env_prefix / "Library" / "bin",
        env_prefix / "Lib" / "site-packages" / "torch" / "lib",
        env_prefix / "lib" / "site-packages" / "torch" / "lib",
    ]
    for path in candidate_dirs:
        if not path.is_dir():
            continue
        os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(path))


_prepare_torch_runtime()

import numpy as np
import torch  # noqa: F401
from tifffile import imread
from tifffile import imwrite

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "napari-cell-registration" / "src"))

from napari_cell_registration.core import CellFeaturesConfig, CellposeConfig, CellposeSegmenter, compute_cell_features

ELASTIX_DIR = Path(__file__).resolve().parent
FIXED_DIR = ELASTIX_DIR / "fixed_images"
MOVING_DIR = ELASTIX_DIR / "moving"
FIXED_FEATURE_DIR = ELASTIX_DIR / "fixed feature"
MOVING_FEATURE_DIR = ELASTIX_DIR / "moving feature"
FIXED_MASK_DIR = ELASTIX_DIR / "fixed_mask"
MOVING_MASK_DIR = ELASTIX_DIR / "moving_mask"
DEFAULT_REG_PARAMS = {
    "cellpose_diameter": 10.0,
    "cellpose_flow_threshold": 0.0,
    "cellpose_cellprob_threshold": 0.0,
    "cellpose_min_size": 15,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export elastixITK masks and feature CSVs.")
    parser.add_argument(
        "--moving-only",
        action="store_true",
        help="Only export masks/features for files under elastixITK/moving.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask/feature outputs instead of skipping them.",
    )
    return parser.parse_args()


def _load_reg_params() -> dict[str, float | int]:
    return DEFAULT_REG_PARAMS


def _make_segmenter(reg_params: dict[str, float | int]) -> CellposeSegmenter:
    gpu = bool(torch.cuda.is_available())
    print(f"Using Cellpose GPU={gpu}")
    return CellposeSegmenter(
        CellposeConfig(
            gpu=gpu,
            pretrained_model="cpsam",
            diameter=reg_params["cellpose_diameter"],
            flow_threshold=reg_params["cellpose_flow_threshold"],
            cellprob_threshold=reg_params["cellpose_cellprob_threshold"],
            min_size=reg_params["cellpose_min_size"],
        )
    )


def _iter_tif_files(folder: Path) -> list[Path]:
    direct_files = sorted(folder.glob("*.tif"))
    if direct_files:
        return direct_files
    return sorted(path for path in folder.glob("*/*.tif") if path.is_file())


def _export_outputs(
    image_path: Path,
    mask_dir: Path,
    feature_dir: Path,
    segmenter: CellposeSegmenter,
    overwrite: bool = False,
) -> None:
    mask_path = mask_dir / f"{image_path.stem}.tif"
    feature_path = feature_dir / f"{image_path.stem}.csv"
    if not overwrite and mask_path.exists() and feature_path.exists():
        print(f"skip {image_path.name} -> {mask_path.name}, {feature_path.name}")
        return

    mask_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    image = imread(str(image_path))
    mask, _, _ = segmenter.segment_array(image)
    features = compute_cell_features(mask, CellFeaturesConfig())
    if "centroid_x" in features.columns and "x" not in features.columns:
        features["x"] = features["centroid_x"]
    if "centroid_y" in features.columns and "y" not in features.columns:
        features["y"] = features["centroid_y"]
    imwrite(str(mask_path), np.asarray(mask, dtype=np.int32))
    features.to_csv(feature_path, index=False)
    print(f"{image_path.name} -> {mask_path.name}, {feature_path.name} ({len(features)} cells)")


def main() -> None:
    args = _parse_args()
    reg_params = _load_reg_params()
    segmenter = _make_segmenter(reg_params)

    FIXED_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    MOVING_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    FIXED_MASK_DIR.mkdir(parents=True, exist_ok=True)
    MOVING_MASK_DIR.mkdir(parents=True, exist_ok=True)

    fixed_files = _iter_tif_files(FIXED_DIR)
    moving_files = _iter_tif_files(MOVING_DIR)

    if not args.moving_only:
        for image_path in fixed_files:
            _export_outputs(
                image_path=image_path,
                mask_dir=FIXED_MASK_DIR,
                feature_dir=FIXED_FEATURE_DIR,
                segmenter=segmenter,
                overwrite=args.overwrite,
            )

    for image_path in moving_files:
        _export_outputs(
            image_path=image_path,
            mask_dir=MOVING_MASK_DIR,
            feature_dir=MOVING_FEATURE_DIR,
            segmenter=segmenter,
            overwrite=args.overwrite,
        )

    if not args.moving_only:
        print(f"Exported fixed masks/features: {len(fixed_files)}")
    print(f"Exported moving masks/features: {len(moving_files)}")


if __name__ == "__main__":
    main()
