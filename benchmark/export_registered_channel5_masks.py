"""Export registered channel-5 masks without running benchmark comparisons."""

from __future__ import annotations

import argparse
import json
import os
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
from tifffile import imwrite

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "napari-cell-registration" / "src"))

from benchmark.run_benchmark import DEFAULT_OUTPUT_DIR, discover_image_pairs, get_or_create_segmentation, run_registration
from napari_cell_registration.core import CellposeConfig, CellposeSegmenter
from napari_cell_registration.core.point_registration import warp_image_with_transform


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export registered channel-5 masks only.")
    parser.add_argument(
        "--params-json",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "benchmark_params.json",
        help="Path to benchmark params JSON. sample_groups / rounds / reg_params will be reused.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "registered_channel5_masks",
        help="Directory to store registered mask TIFF files.",
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
    import torch

    return bool(torch.cuda.is_available())


def _warp_mask(mask: np.ndarray, transform: Any, output_shape: tuple[int, int]) -> np.ndarray:
    warped = warp_image_with_transform(mask.astype(np.int32), transform, output_shape, order=0)
    return np.rint(warped).astype(np.int32)


def main() -> None:
    args = _parse_args()
    sample_groups, rounds, reg_params = _load_params(args.params_json.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_image_pairs(
        sample_groups,
        rounds,
        require_secondary_comparison=False,
    )
    if not pairs:
        raise RuntimeError("No registration pairs found.")

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

    segmentation_cache: dict[Path, Any] = {}
    exported_reference_masks: set[Path] = set()
    exported_reference_features: set[Path] = set()

    for index, pair in enumerate(pairs, start=1):
        print(f"[{index}/{len(pairs)}] Group {pair.sample_group} {pair.round_name} -> B")
        source_artifacts = get_or_create_segmentation(pair.source_path, segmenter, segmentation_cache)
        reference_artifacts = get_or_create_segmentation(pair.reference_path, segmenter, segmentation_cache)
        registration = run_registration(source_artifacts, reference_artifacts, reg_params)

        output_shape = tuple(int(v) for v in reference_artifacts.mask.shape)
        registered_mask = _warp_mask(source_artifacts.mask, registration.transform, output_shape)

        output_path = output_dir / f"registered_mask_G{pair.sample_group}_{pair.round_name}_to_B_ch05.tif"
        moving_features_path = output_dir / f"moving_features_G{pair.sample_group}_{pair.round_name}_to_B_ch05.csv"
        imwrite(str(output_path), registered_mask)
        source_artifacts.features.to_csv(moving_features_path, index=False)
        print(f"  saved: {output_path.name}")
        print(f"  saved: {moving_features_path.name}")

        if reference_artifacts.image_path not in exported_reference_masks:
            reference_output_path = output_dir / f"reference_mask_G{pair.sample_group}_B_ch05.tif"
            imwrite(str(reference_output_path), reference_artifacts.mask.astype(np.int32))
            exported_reference_masks.add(reference_artifacts.image_path)
            print(f"  saved: {reference_output_path.name}")
        if reference_artifacts.image_path not in exported_reference_features:
            reference_features_path = output_dir / f"fixed_features_G{pair.sample_group}_B_ch05.csv"
            reference_artifacts.features.to_csv(reference_features_path, index=False)
            exported_reference_features.add(reference_artifacts.image_path)
            print(f"  saved: {reference_features_path.name}")


if __name__ == "__main__":
    main()
