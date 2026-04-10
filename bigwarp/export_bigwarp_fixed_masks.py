"""Export BigWarp fixed-image masks only."""

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

from napari_cell_registration.core import CellposeConfig, CellposeSegmenter

BIGWARP_DIR = Path(__file__).resolve().parent
FIXED_IMAGE_DIR = BIGWARP_DIR / "fixed_images"
FIXED_MASK_DIR = BIGWARP_DIR / "fixed mask"
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


def main() -> None:
    if not FIXED_IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Fixed image directory not found: {FIXED_IMAGE_DIR}")

    fixed_files = sorted(FIXED_IMAGE_DIR.glob("*.tif"))
    if not fixed_files:
        raise FileNotFoundError(f"No TIFF images found in {FIXED_IMAGE_DIR}")

    FIXED_MASK_DIR.mkdir(parents=True, exist_ok=True)
    segmenter = _make_segmenter(_load_reg_params())

    for image_path in fixed_files:
        mask_path = FIXED_MASK_DIR / image_path.name
        image = np.asarray(imread(str(image_path)))
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D image at {image_path}, got shape {image.shape}.")
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        mask, _, _ = segmenter.segment_array(image)
        imwrite(str(mask_path), np.asarray(mask, dtype=np.int32))
        print(f"{image_path.name} -> {mask_path.name}")

    print(f"Exported fixed masks: {len(fixed_files)}")
    print(f"Mask output: {FIXED_MASK_DIR}")


if __name__ == "__main__":
    main()
