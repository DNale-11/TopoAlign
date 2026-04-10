"""Export masks and feature CSVs for ELD registration outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


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

from benchmark.run_benchmark import DEFAULT_OUTPUT_DIR
from napari_cell_registration.core import CellFeaturesConfig, CellposeConfig, CellposeSegmenter, compute_cell_features

ELD_DIR = Path(__file__).resolve().parent
FIXED_DIR = ELD_DIR / "fixed"
MOVING_DIR = ELD_DIR / "moving"
FIXED_PATTERN = re.compile(r"C(\d+)-Bfixed_image\.tif$", re.IGNORECASE)
MOVING_PATTERN = re.compile(r"C(\d+)\s+([A-Z])-Bmoving_registered\.tif$", re.IGNORECASE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ELD masks and feature CSVs.")
    parser.add_argument(
        "--params-json",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "benchmark_params.json",
        help="Path to benchmark params JSON. sample_groups / rounds / cellpose params will be reused.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ELD_DIR / "masks_features",
        help="Directory to store mask TIFF files and feature CSVs.",
    )
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force Cellpose GPU on/off. Default: auto-detect from torch.",
    )
    return parser.parse_args()


def _load_params(path: Path) -> tuple[list[int], list[str], dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["sample_groups"], payload["rounds"], payload["reg_params"]


def _resolve_gpu_flag(requested: bool | None) -> bool:
    if requested is not None:
        return requested
    return bool(torch.cuda.is_available())


def _load_image(path: Path) -> np.ndarray:
    image = imread(str(path))
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image at {path}, got shape {image.shape}.")
    return np.asarray(image, dtype=np.uint8)


def _find_fixed_image(group: int) -> Path:
    for path in sorted(FIXED_DIR.glob("*.tif")):
        match = FIXED_PATTERN.search(path.name)
        if match and int(match.group(1)) == group:
            return path
    raise FileNotFoundError(f"Fixed B image not found for group {group} in {FIXED_DIR}")


def _find_moving_images(group: int) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(MOVING_DIR.glob("*.tif")):
        match = MOVING_PATTERN.search(path.name)
        if not match:
            continue
        sample_group = int(match.group(1))
        round_name = match.group(2).upper()
        if sample_group == group:
            result[round_name] = path
    return result


def main() -> None:
    args = _parse_args()
    sample_groups, rounds, reg_params = _load_params(args.params_json.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu = _resolve_gpu_flag(args.gpu)
    print(f"Using Cellpose GPU={gpu}")
    segmenter = CellposeSegmenter(
        CellposeConfig(
            gpu=gpu,
            pretrained_model="cpsam",
            diameter=reg_params["cellpose_diameter"],
            flow_threshold=reg_params["cellpose_flow_threshold"],
            cellprob_threshold=reg_params["cellpose_cellprob_threshold"],
            min_size=reg_params["cellpose_min_size"],
        )
    )

    for group in sample_groups:
        fixed_path = _find_fixed_image(group)
        fixed_image = _load_image(fixed_path)
        fixed_mask, _, _ = segmenter.segment_array(fixed_image)
        fixed_features = compute_cell_features(fixed_mask, CellFeaturesConfig())

        fixed_mask_path = output_dir / f"fixed_mask_G{group}_B.tif"
        fixed_features_path = output_dir / f"fixed_features_G{group}_B.csv"
        imwrite(str(fixed_mask_path), fixed_mask.astype(np.int32))
        fixed_features.to_csv(fixed_features_path, index=False)
        print(f"[G{group}] saved fixed: {fixed_mask_path.name}")
        print(f"[G{group}] saved fixed features: {fixed_features_path.name}")

        moving_images = _find_moving_images(group)
        for round_name in rounds:
            if round_name == "B":
                continue
            moving_path = moving_images.get(round_name)
            if moving_path is None:
                print(f"[G{group}] missing moving image for round {round_name}, skipped")
                continue

            moving_image = _load_image(moving_path)
            moving_mask, _, _ = segmenter.segment_array(moving_image)
            moving_features = compute_cell_features(moving_mask, CellFeaturesConfig())

            moving_mask_path = output_dir / f"moving_mask_G{group}_{round_name}_to_B.tif"
            moving_features_path = output_dir / f"moving_features_G{group}_{round_name}_to_B.csv"
            imwrite(str(moving_mask_path), moving_mask.astype(np.int32))
            moving_features.to_csv(moving_features_path, index=False)
            print(f"[G{group}] saved moving {round_name}: {moving_mask_path.name}")
            print(f"[G{group}] saved moving {round_name} features: {moving_features_path.name}")


if __name__ == "__main__":
    main()
