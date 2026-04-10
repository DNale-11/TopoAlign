"""Export masks from manually registered BigWarp images using channel 5 only."""

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

REGISTRATION_DIR = Path(__file__).resolve().parent / "registration"
ROUND_PATTERN = re.compile(r"C3-([A-Z])-63-(\d+)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export moving/fixed masks from BigWarp manual registration results.")
    parser.add_argument(
        "--params-json",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "benchmark_params.json",
        help="Path to benchmark params JSON. sample_groups / rounds / cellpose params will be reused.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "bigwarp_manual_channel5_masks",
        help="Directory to store mask TIFF files.",
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


def _extract_channel5(image_path: Path) -> np.ndarray:
    image = imread(str(image_path))
    if image.ndim != 3 or image.shape[0] < 5:
        raise ValueError(f"Expected a 5-channel TIFF at {image_path}, got shape {image.shape}.")
    return np.asarray(image[4], dtype=np.uint8)


def _find_group_files(group: int) -> tuple[Path, dict[str, Path]]:
    folder = REGISTRATION_DIR / f"C1-{group}"
    if not folder.is_dir():
        raise FileNotFoundError(f"Registration folder not found: {folder}")

    fixed_path: Path | None = None
    moving_paths: dict[str, Path] = {}
    for image_path in sorted(folder.glob("*.tif")):
        match = ROUND_PATTERN.search(image_path.name)
        if match is None:
            continue
        round_name, sample_group = match.group(1), int(match.group(2))
        if sample_group != group:
            continue
        if round_name == "B":
            fixed_path = image_path
        else:
            moving_paths[round_name] = image_path

    if fixed_path is None:
        raise FileNotFoundError(f"Fixed B image not found in {folder}")
    return fixed_path, moving_paths


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
        fixed_path, moving_paths = _find_group_files(group)
        fixed_channel5 = _extract_channel5(fixed_path)
        fixed_mask, _, _ = segmenter.segment_array(fixed_channel5)
        fixed_features = compute_cell_features(fixed_mask, CellFeaturesConfig())
        fixed_output_path = output_dir / f"fixed_mask_G{group}_B_ch05.tif"
        fixed_features_path = output_dir / f"fixed_features_G{group}_B_ch05.csv"
        imwrite(str(fixed_output_path), fixed_mask.astype(np.int32))
        fixed_features.to_csv(fixed_features_path, index=False)
        print(f"[G{group}] saved fixed: {fixed_output_path.name}")
        print(f"[G{group}] saved fixed features: {fixed_features_path.name}")

        for round_name in rounds:
            if round_name == "B":
                continue
            moving_path = moving_paths.get(round_name)
            if moving_path is None:
                print(f"[G{group}] missing moving image for round {round_name}, skipped")
                continue
            moving_channel5 = _extract_channel5(moving_path)
            moving_mask, _, _ = segmenter.segment_array(moving_channel5)
            moving_features = compute_cell_features(moving_mask, CellFeaturesConfig())
            moving_output_path = output_dir / f"moving_mask_G{group}_{round_name}_to_B_ch05.tif"
            moving_features_path = output_dir / f"moving_features_G{group}_{round_name}_to_B_ch05.csv"
            imwrite(str(moving_output_path), moving_mask.astype(np.int32))
            moving_features.to_csv(moving_features_path, index=False)
            print(f"[G{group}] saved moving {round_name}: {moving_output_path.name}")
            print(f"[G{group}] saved moving {round_name} features: {moving_features_path.name}")


if __name__ == "__main__":
    main()
