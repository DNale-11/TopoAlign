"""Segment BigWarp fixed/moving images and export masks plus cell features."""

from __future__ import annotations

import json
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
from tifffile import imread, imwrite

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "napari-cell-registration" / "src"))

from napari_cell_registration.core import CellFeaturesConfig, CellposeConfig, CellposeSegmenter, compute_cell_features

BIGWARP_DIR = Path(__file__).resolve().parent
FIXED_IMAGE_DIR = BIGWARP_DIR / "fixed"
MOVING_IMAGE_DIR = BIGWARP_DIR / "moving"
FIXED_MASK_DIR = BIGWARP_DIR / "fixed mask"
MOVING_MASK_DIR = BIGWARP_DIR / "moving mask"
FIXED_FEATURE_DIR = BIGWARP_DIR / "fixed feature"
MOVING_FEATURE_DIR = BIGWARP_DIR / "moving feature"
RUN_MANIFEST_PATH = PROJECT_ROOT / "benchmark" / "results" / "unregistration_full" / "run_manifest.json"
DEFAULT_REG_PARAMS = {
    "cellpose_diameter": 10.0,
    "cellpose_flow_threshold": 0.0,
    "cellpose_cellprob_threshold": 0.0,
    "cellpose_min_size": 15,
}


def _load_reg_params() -> dict[str, float | int]:
    if RUN_MANIFEST_PATH.is_file():
        with open(RUN_MANIFEST_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        reg_params = payload.get("reg_params", {})
        return {
            "cellpose_diameter": reg_params.get("cellpose_diameter", DEFAULT_REG_PARAMS["cellpose_diameter"]),
            "cellpose_flow_threshold": reg_params.get(
                "cellpose_flow_threshold",
                DEFAULT_REG_PARAMS["cellpose_flow_threshold"],
            ),
            "cellpose_cellprob_threshold": reg_params.get(
                "cellpose_cellprob_threshold",
                DEFAULT_REG_PARAMS["cellpose_cellprob_threshold"],
            ),
            "cellpose_min_size": reg_params.get("cellpose_min_size", DEFAULT_REG_PARAMS["cellpose_min_size"]),
        }
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


def _export_one(
    image_path: Path,
    mask_path: Path,
    feature_path: Path,
    segmenter: CellposeSegmenter,
) -> None:
    if mask_path.exists() and feature_path.exists():
        print(f"skip {image_path.name}")
        return

    image = imread(str(image_path))
    mask, _, _ = segmenter.segment_array(image)
    features = compute_cell_features(mask, CellFeaturesConfig())
    if "centroid_x" in features.columns and "x" not in features.columns:
        features["x"] = features["centroid_x"]
    if "centroid_y" in features.columns and "y" not in features.columns:
        features["y"] = features["centroid_y"]

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(str(mask_path), np.asarray(mask, dtype=np.int32))
    features.to_csv(feature_path, index=False)
    print(f"{image_path.name} -> {mask_path.name}, {feature_path.name} ({len(features)} cells)")


def main() -> None:
    reg_params = _load_reg_params()
    segmenter = _make_segmenter(reg_params)

    FIXED_MASK_DIR.mkdir(parents=True, exist_ok=True)
    FIXED_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    MOVING_MASK_DIR.mkdir(parents=True, exist_ok=True)
    MOVING_FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    fixed_files = sorted(FIXED_IMAGE_DIR.glob("*.tif"))
    moving_files = sorted(MOVING_IMAGE_DIR.glob("*/*.tif"))

    for image_path in fixed_files:
        mask_path = FIXED_MASK_DIR / image_path.name
        feature_path = FIXED_FEATURE_DIR / f"{image_path.stem}.csv"
        _export_one(image_path, mask_path, feature_path, segmenter)

    for image_path in moving_files:
        group_name = image_path.parent.name
        mask_path = MOVING_MASK_DIR / group_name / image_path.name
        feature_path = MOVING_FEATURE_DIR / group_name / f"{image_path.stem}.csv"
        _export_one(image_path, mask_path, feature_path, segmenter)

    print(f"Exported fixed images: {len(fixed_files)}")
    print(f"Exported moving images: {len(moving_files)}")


if __name__ == "__main__":
    main()
