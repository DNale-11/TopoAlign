"""
Object-level benchmark for the cell registration pipeline.

The benchmark evaluates the final estimated transform in fixed-image space using
precomputed segmentations from the original raw images. Warped images are never
re-segmented for the main benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from registration_runtime import (
    CellposeConfig as SharedCellposeConfig,
    CellposeSegmenter as SharedCellposeSegmenter,
    DEFAULT_OBJECT_EVAL_PARAMS as SHARED_DEFAULT_OBJECT_EVAL_PARAMS,
    DEFAULT_REG_PARAMS as SHARED_DEFAULT_REG_PARAMS,
    NAPARI_CORE_IMPORT_ERROR as SHARED_NAPARI_CORE_IMPORT_ERROR,
    compute_secondary_intensity_metrics as shared_compute_secondary_intensity_metrics,
    get_or_create_segmentation as shared_get_or_create_segmentation,
    prepare_torch_runtime as shared_prepare_torch_runtime,
    run_registration as shared_run_registration,
)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.transform import AffineTransform
from tifffile import imread, imwrite

matplotlib.use("Agg")

# Preload torch on Windows before importing cellpose-dependent modules.
_torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if not os.path.isdir(_torch_lib):
    _torch_lib = os.path.join(sys.prefix, "lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_lib):
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_torch_lib)
try:
    import torch  # noqa: F401
except Exception:
    torch = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "napari-cell-registration" / "src"))

from cell_registration.evaluation import compute_object_registration_metrics, compute_overlap_metrics

NAPARI_CORE_IMPORT_ERROR: Exception | None = None
try:
    from napari_cell_registration.core import (
        CellFeaturesConfig,
        CellposeConfig,
        CellposeSegmenter,
        MatchingConfig,
        compute_cell_features,
        greedy_match_cells,
    )
    from napari_cell_registration.core.matching import apply_transform_to_features, two_stage_match_cells
    from napari_cell_registration.core.point_registration import (
        compute_valid_overlap_mask,
        estimate_robust_transform,
        refine_transform_with_neighbors,
        warp_image_with_transform,
    )
    from napari_cell_registration.core.validation import validate_matches
except ImportError as exc:
    NAPARI_CORE_IMPORT_ERROR = exc
    CellFeaturesConfig = object
    CellposeConfig = object
    CellposeSegmenter = object
    MatchingConfig = object

    def _missing_dependency(*args, **kwargs):
        raise ImportError(
            "Benchmark runtime dependencies are missing. Install cellpose and related registration dependencies."
        ) from NAPARI_CORE_IMPORT_ERROR

    compute_cell_features = _missing_dependency
    greedy_match_cells = _missing_dependency
    apply_transform_to_features = _missing_dependency
    two_stage_match_cells = _missing_dependency
    compute_valid_overlap_mask = _missing_dependency
    estimate_robust_transform = _missing_dependency
    refine_transform_with_neighbors = _missing_dependency
    warp_image_with_transform = _missing_dependency
    validate_matches = _missing_dependency


BENCHMARK_DIR = Path(__file__).resolve().parent
UNREG_DIR = BENCHMARK_DIR / "unregistration"
REG_DIR = BENCHMARK_DIR / "registration"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results"

DEFAULT_SAMPLE_GROUPS = [1, 2, 3]
DEFAULT_ROUNDS = ["A", "C", "D", "E"]

DEFAULT_REG_PARAMS = {
    "cellpose_diameter": 15.0,
    "cellpose_flow_threshold": -2.0,
    "cellpose_cellprob_threshold": 1.0,
    "cellpose_min_size": 5,
    "position_weight": 50.0,
    "distance_threshold": 2.0,
    "spatial_window_size": 100.0,
    "top_k": 100,
    "min_cells_for_two_stage": 20,
    "coarse_top_k": 50,
    "coarse_distance_threshold": 2.0,
    "neighbor_k": 5,
    "neighbor_weight": 1.0,
    "landmark_weight": 10.0,
    "max_theta_deg": 10.0,
    "max_translation": 30.0,
    "max_scale_change": 0.05,
    "guided_min_position_weight": 4.0,
    "guided_spatial_window_cap": 60.0,
    "validation_min_confidence": 0.2,
    "validation_max_feature_diff": 0.7,
    "allow_scale": False,
    "prefer_affine": False,
    "ransac_max_trials": 500,
    "ransac_residual_threshold": 3.0,
    "similarity_residual_threshold": 5.0,
}

DEFAULT_OBJECT_EVAL_PARAMS = {
    "instance_iou_threshold": 0.3,
    "min_valid_instance_area_px": 20,
    "min_valid_fraction": 0.5,
}

OBJECT_HEADLINE_METRICS = [
    "match_f1",
    "matched_mean_iou",
    "centroid_error_median_px",
    "centroid_error_p95_px",
    "mask_dice",
]
OBJECT_SUPPORTING_METRICS = [
    "mask_iou",
    "matched_cells",
    "match_precision",
    "match_recall",
    "tre_mean_px",
    "tre_median_px",
    "tre_p95_px",
    "valid_overlap_ratio",
    "eligible_moving_cells",
    "eligible_fixed_cells",
]
OBJECT_ALL_METRICS = OBJECT_HEADLINE_METRICS + OBJECT_SUPPORTING_METRICS + [
    "centroid_error_mean_px",
]
OBJECT_PLOT_METRICS = [
    "match_f1",
    "matched_mean_iou",
    "centroid_error_median_px",
    "centroid_error_p95_px",
    "mask_dice",
]
SECONDARY_INTENSITY_METRICS = ["PSNR", "SSIM", "MSE", "NRMSE", "NCC", "MI", "valid_overlap_ratio"]


def _prepare_torch_runtime() -> None:
    """Load torch lazily so --help works without initializing the full runtime."""
    torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if not os.path.isdir(torch_lib):
        torch_lib = os.path.join(sys.prefix, "lib", "site-packages", "torch", "lib")
    if os.path.isdir(torch_lib):
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(torch_lib)
    __import__("torch")


@dataclass
class ImagePair:
    sample_group: int
    round_name: str
    source_path: Path
    reference_path: Path
    comparison_path: Path | None = None


@dataclass
class SegmentationArtifacts:
    image_path: Path
    image: np.ndarray = field(repr=False)
    mask: np.ndarray = field(repr=False)
    features: pd.DataFrame = field(repr=False)


@dataclass
class RegistrationArtifacts:
    registered_image: np.ndarray = field(repr=False)
    export_image: np.ndarray = field(repr=False)
    valid_mask: np.ndarray = field(repr=False)
    elapsed_seconds: float
    transform: AffineTransform = field(repr=False)
    moving_mask: np.ndarray = field(repr=False)
    fixed_mask: np.ndarray = field(repr=False)
    moving_features: pd.DataFrame = field(repr=False)
    fixed_features: pd.DataFrame = field(repr=False)
    diagnostics: dict[str, object] = field(default_factory=dict, repr=False)


@dataclass
class BenchmarkResult:
    sample_group: int
    round_name: str
    registration_time_sec: float
    object_metrics: dict[str, float]
    diagnostics: dict[str, object] = field(default_factory=dict)
    secondary_intensity_metrics: dict[str, float] | None = None
    registered_image: np.ndarray | None = field(default=None, repr=False)
    valid_mask: np.ndarray | None = field(default=None, repr=False)
    match_table: pd.DataFrame | None = field(default=None, repr=False)


def find_unreg_image(sample_group: int, round_name: str) -> Path | None:
    folder = UNREG_DIR / f"C-{sample_group}"
    if not folder.exists():
        return None
    pattern = f"C3-{round_name}-63-{sample_group}"
    for file_path in folder.iterdir():
        if file_path.suffix.lower() in (".tif", ".tiff") and pattern in file_path.name:
            return file_path
    return None


def find_b_merged_reference(sample_group: int) -> Path | None:
    folder = REG_DIR / f"C1-{sample_group}"
    if not folder.exists():
        return None
    pattern = f"C3-B-63-{sample_group}"
    for file_path in folder.iterdir():
        if file_path.suffix.lower() in (".tif", ".tiff") and pattern in file_path.name:
            return file_path
    return None


def load_b_reference_channel5(path: Path) -> np.ndarray:
    merged = imread(str(path))
    if merged.ndim != 3:
        raise ValueError(f"Expected a 3D merged TIFF at {path}, got shape {merged.shape}.")
    if merged.shape[0] < 5:
        raise ValueError(f"Expected at least 5 channels in {path}, got shape {merged.shape}.")
    return merged[4]


def discover_image_pairs(
    sample_groups: list[int],
    rounds: list[str],
    *,
    require_secondary_comparison: bool,
) -> list[ImagePair]:
    pairs: list[ImagePair] = []
    for sample_group in sample_groups:
        reference_path = find_unreg_image(sample_group, "B")
        if reference_path is None:
            print(f"  Reference image B not found for sample group {sample_group}, skipping")
            continue

        comparison_path = find_b_merged_reference(sample_group)
        if require_secondary_comparison and comparison_path is None:
            print(f"  Secondary comparison image B not found for sample group {sample_group}, skipping")
            continue

        for round_name in rounds:
            source_path = find_unreg_image(sample_group, round_name)
            if source_path is None:
                print(f"  Source image {round_name} not found for sample group {sample_group}, skipping")
                continue
            pairs.append(
                ImagePair(
                    sample_group=sample_group,
                    round_name=round_name,
                    source_path=source_path,
                    reference_path=reference_path,
                    comparison_path=comparison_path,
                )
            )
    return pairs


def to_2d_gray(img: np.ndarray) -> np.ndarray:
    """Use the last channel for multichannel DAPI-like images."""
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[0] <= 10 and img.shape[1] > 10 and img.shape[2] > 10:
            return img[-1]
        if img.shape[2] <= 10:
            return img[..., -1]
        return np.max(img, axis=0)
    return np.asarray(img)


def align_images_for_comparison(
    registered: np.ndarray,
    comparison_image: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reg_2d = np.asarray(to_2d_gray(registered), dtype=np.float64)
    ref_2d = np.asarray(to_2d_gray(comparison_image), dtype=np.float64)
    h = min(reg_2d.shape[0], ref_2d.shape[0])
    w = min(reg_2d.shape[1], ref_2d.shape[1])
    if valid_mask is None:
        mask = np.ones((h, w), dtype=bool)
    else:
        mask = np.asarray(valid_mask)[:h, :w] > 0.5
    return reg_2d[:h, :w], ref_2d[:h, :w], mask


def compute_secondary_intensity_metrics(
    registered: np.ndarray,
    comparison_image: np.ndarray,
    valid_mask: np.ndarray | None,
) -> dict[str, float]:
    reg_2d, ref_2d, mask_2d = align_images_for_comparison(
        registered,
        comparison_image,
        valid_mask=valid_mask,
    )
    return compute_overlap_metrics(reg_2d, ref_2d, valid_mask=mask_2d)


def _cast_warped_image(warped: np.ndarray, source_dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(source_dtype, np.integer):
        dtype_info = np.iinfo(source_dtype)
        return np.clip(np.rint(warped), dtype_info.min, dtype_info.max).astype(source_dtype)
    return warped.astype(source_dtype, copy=False)


def _finalize_registration(
    *,
    source_img: np.ndarray,
    reference_img: np.ndarray,
    source_mask: np.ndarray,
    reference_mask: np.ndarray,
    source_features: pd.DataFrame,
    reference_features: pd.DataFrame,
    transform: AffineTransform,
    elapsed_seconds: float,
    diagnostics: dict[str, object],
) -> RegistrationArtifacts:
    reference_shape = tuple(int(v) for v in to_2d_gray(reference_img).shape[:2])
    source_shape = tuple(int(v) for v in to_2d_gray(source_img).shape[:2])

    registered_f = warp_image_with_transform(
        source_img,
        transform,
        reference_shape,
        order=1,
    )
    export_f = warp_image_with_transform(
        source_img,
        transform,
        source_shape,
        order=1,
    )
    registered = _cast_warped_image(registered_f, source_img.dtype)
    export_image = _cast_warped_image(export_f, source_img.dtype)
    valid_mask = compute_valid_overlap_mask(source_shape, transform, reference_shape)

    return RegistrationArtifacts(
        registered_image=registered,
        export_image=export_image,
        valid_mask=valid_mask,
        elapsed_seconds=elapsed_seconds,
        transform=transform,
        moving_mask=source_mask,
        fixed_mask=reference_mask,
        moving_features=source_features,
        fixed_features=reference_features,
        diagnostics=diagnostics,
    )


def run_registration(
    source_artifacts: SegmentationArtifacts,
    reference_artifacts: SegmentationArtifacts,
    reg_params: dict[str, Any],
) -> RegistrationArtifacts:
    """
    Run the registration pipeline using precomputed source/reference masks and features.
    """
    source_img = source_artifacts.image
    reference_img = reference_artifacts.image
    mask_src = source_artifacts.mask
    mask_ref = reference_artifacts.mask
    feats_src = source_artifacts.features
    feats_ref = reference_artifacts.features

    t_start = time.perf_counter()
    n_src = len(feats_src)
    n_ref = len(feats_ref)
    print(f"    Cells found: source={n_src}, reference={n_ref}")

    reference_shape = tuple(int(v) for v in to_2d_gray(reference_img).shape[:2])

    if n_src < 3 or n_ref < 3:
        print("    Too few cells for registration, falling back to identity transform")
        elapsed = time.perf_counter() - t_start
        return _finalize_registration(
            source_img=source_img,
            reference_img=reference_img,
            source_mask=mask_src,
            reference_mask=mask_ref,
            source_features=feats_src,
            reference_features=feats_ref,
            transform=AffineTransform(),
            elapsed_seconds=elapsed,
            diagnostics={
                "transform_method": "insufficient_cells",
                "inlier_count": 0,
                "match_count": 0,
                "median_inlier_residual": float("inf"),
                "mean_inlier_residual": float("inf"),
            },
        )

    match_result = two_stage_match_cells(
        feats_ref,
        feats_src,
        mask_ref.shape,
        position_weight=reg_params["position_weight"],
        top_k=reg_params["top_k"],
        distance_threshold=reg_params["distance_threshold"],
        spatial_window_size=reg_params["spatial_window_size"],
        min_cells_for_two_stage=reg_params["min_cells_for_two_stage"],
        coarse_top_k=reg_params["coarse_top_k"],
        coarse_distance_threshold=reg_params["coarse_distance_threshold"],
    )
    if not match_result.coarse_matches.empty:
        print(f"    Coarse matches: {len(match_result.coarse_matches)}")
        print(
            "    Coarse offset: "
            f"dx={match_result.coarse_offset_xy[0]:.2f}, dy={match_result.coarse_offset_xy[1]:.2f}"
        )

    matches = match_result.matches
    print(f"    Fine matches: {len(matches)}")

    config_val = MatchingConfig(
        feature_weight=reg_params["feature_weight"],
        position_weight=reg_params["position_weight"],
        distance_threshold=reg_params["distance_threshold"],
        min_confidence=reg_params["validation_min_confidence"],
        max_feature_diff=reg_params["validation_max_feature_diff"],
        validation_neighbor_k=reg_params["validation_neighbor_k"],
        validation_max_neighbor_profile_diff=reg_params["validation_max_neighbor_profile_diff"],
        validation_ambiguity_ratio=reg_params["validation_ambiguity_ratio"],
        validation_ambiguity_min_gap=reg_params["validation_ambiguity_min_gap"],
        spatial_window_size=reg_params["spatial_window_size"],
    )
    matches = validate_matches(matches, feats_ref, match_result.aligned_df2, config_val)
    print(f"    Validated matches: {len(matches)}")

    if len(matches) < 3:
        print("    Too few matches for transform estimation, falling back to identity transform")
        elapsed = time.perf_counter() - t_start
        return _finalize_registration(
            source_img=source_img,
            reference_img=reference_img,
            source_mask=mask_src,
            reference_mask=mask_ref,
            source_features=feats_src,
            reference_features=feats_ref,
            transform=AffineTransform(),
            elapsed_seconds=elapsed,
            diagnostics={
                "transform_method": "insufficient_matches",
                "inlier_count": 0,
                "match_count": int(len(matches)),
                "median_inlier_residual": float("inf"),
                "mean_inlier_residual": float("inf"),
            },
        )

    pts_ref_yx = feats_ref.loc[matches["idx1"], ["centroid_y", "centroid_x"]].to_numpy()
    pts_src_yx = feats_src.loc[matches["idx2"], ["centroid_y", "centroid_x"]].to_numpy()
    pts_ref_xy = pts_ref_yx[:, ::-1]
    pts_src_xy = pts_src_yx[:, ::-1]

    robust = estimate_robust_transform(
        pts_ref_xy,
        pts_src_xy,
        allow_scale=bool(reg_params["allow_scale"]),
        prefer_affine=bool(reg_params["prefer_affine"]),
        residual_threshold=reg_params["ransac_residual_threshold"],
        similarity_residual_threshold=reg_params["similarity_residual_threshold"],
        max_trials=reg_params["ransac_max_trials"],
    )
    affine = robust.transform
    print(
        f"    RANSAC model: {robust.method} | "
        f"inliers: {robust.inlier_count}/{len(matches)} | "
        f"median residual: {robust.median_inlier_residual:.2f}px"
    )

    guided_feats_src = apply_transform_to_features(feats_src, affine, reference_shape)
    guided_spatial_cap = reg_params["guided_spatial_window_cap"]
    spatial_window = reg_params["spatial_window_size"]
    if guided_spatial_cap is None or guided_spatial_cap <= 0:
        guided_window = spatial_window
    else:
        guided_window = spatial_window if spatial_window is None else min(spatial_window, guided_spatial_cap)
    guided_config = MatchingConfig(
        position_weight=max(reg_params["position_weight"], reg_params["guided_min_position_weight"]),
        top_k=reg_params["top_k"],
        distance_threshold=None if reg_params["distance_threshold"] is None else reg_params["distance_threshold"] + 0.5,
        spatial_window_size=guided_window,
    )
    guided_matches = greedy_match_cells(feats_ref, guided_feats_src, guided_config)
    guided_matches = validate_matches(guided_matches, feats_ref, guided_feats_src, config_val)
    if len(guided_matches) >= 3:
        guided_ref_xy = feats_ref.loc[guided_matches["idx1"], ["centroid_x", "centroid_y"]].to_numpy()
        guided_src_xy = feats_src.loc[guided_matches["idx2"], ["centroid_x", "centroid_y"]].to_numpy()
        guided = estimate_robust_transform(
            guided_ref_xy,
            guided_src_xy,
            allow_scale=bool(reg_params["allow_scale"]),
            prefer_affine=bool(reg_params["prefer_affine"]),
            residual_threshold=reg_params["ransac_residual_threshold"],
            similarity_residual_threshold=reg_params["similarity_residual_threshold"],
            max_trials=reg_params["ransac_max_trials"],
        )
        if guided.score() > robust.score():
            matches = guided_matches
            robust = guided
            affine = robust.transform
            pts_ref_xy = guided_ref_xy
            pts_src_xy = guided_src_xy
            print(
                f"    Guided rematch improved support: {robust.method}, "
                f"{robust.inlier_count}/{len(matches)} inliers"
            )

    if robust.inlier_count >= 3:
        pts_ref_xy_inliers = pts_ref_xy[robust.inliers]
        pts_src_xy_inliers = pts_src_xy[robust.inliers]
    else:
        pts_ref_xy_inliers = np.empty((0, 2), dtype=float)
        pts_src_xy_inliers = np.empty((0, 2), dtype=float)

    if len(pts_ref_xy_inliers) >= 3:
        print(f"    Refining with k={reg_params['neighbor_k']} neighbors (KNN)...")
        affine = refine_transform_with_neighbors(
            pts_ref_xy_inliers,
            pts_src_xy_inliers,
            initial=affine,
            k=reg_params["neighbor_k"],
            neighbor_weight=reg_params["neighbor_weight"],
            landmark_weight=reg_params["landmark_weight"],
            max_theta_deg=reg_params["max_theta_deg"],
            max_translation=reg_params["max_translation"],
            max_scale_change=reg_params["max_scale_change"],
            allow_scale=bool(reg_params["allow_scale"]),
        )
        if robust.method != "similarity":
            affine = refine_transform_with_neighbors(
                pts_ref_xy_inliers,
                pts_src_xy_inliers,
                initial=affine,
                k=reg_params["neighbor_k"],
                neighbor_weight=reg_params["neighbor_weight"],
                landmark_weight=reg_params["landmark_weight"],
                max_theta_deg=reg_params["max_theta_deg"],
                max_translation=reg_params["max_translation"],
                max_scale_change=reg_params["max_scale_change"],
                allow_scale=bool(reg_params["allow_scale"]),
                optimize_translation_only=True,
            )
        print("    KNN refinement complete")

    elapsed = time.perf_counter() - t_start
    return _finalize_registration(
        source_img=source_img,
        reference_img=reference_img,
        source_mask=mask_src,
        reference_mask=mask_ref,
        source_features=feats_src,
        reference_features=feats_ref,
        transform=affine,
        elapsed_seconds=elapsed,
        diagnostics={
            "transform_method": robust.method,
            "inlier_count": int(robust.inlier_count),
            "match_count": int(len(matches)),
            "median_inlier_residual": float(robust.median_inlier_residual),
            "mean_inlier_residual": float(robust.mean_inlier_residual),
        },
    )


def get_or_create_segmentation(
    image_path: Path,
    segmenter: CellposeSegmenter,
    cache: dict[Path, SegmentationArtifacts],
) -> SegmentationArtifacts:
    if image_path in cache:
        return cache[image_path]

    image = imread(str(image_path))
    mask, _, _ = segmenter.segment_array(image)
    features = compute_cell_features(mask, CellFeaturesConfig())
    artifacts = SegmentationArtifacts(
        image_path=image_path,
        image=image,
        mask=mask,
        features=features,
    )
    cache[image_path] = artifacts
    return artifacts


# Keep the benchmark entrypoint on the shared runtime implementation used by the
# unregistration batch exporter so both scripts stay behaviorally aligned.
DEFAULT_REG_PARAMS = SHARED_DEFAULT_REG_PARAMS
DEFAULT_OBJECT_EVAL_PARAMS = SHARED_DEFAULT_OBJECT_EVAL_PARAMS
NAPARI_CORE_IMPORT_ERROR = SHARED_NAPARI_CORE_IMPORT_ERROR
CellposeConfig = SharedCellposeConfig
CellposeSegmenter = SharedCellposeSegmenter
_prepare_torch_runtime = shared_prepare_torch_runtime
compute_secondary_intensity_metrics = shared_compute_secondary_intensity_metrics
get_or_create_segmentation = shared_get_or_create_segmentation
run_registration = shared_run_registration


def _fmt_metric(value: Any, precision: int = 4) -> str:
    if value is None:
        return "nan"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(numeric):
        return "nan"
    return f"{numeric:.{precision}f}"


def build_results_tables(
    results: list[BenchmarkResult],
    object_eval_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    main_rows: list[dict[str, Any]] = []
    secondary_rows: list[dict[str, Any]] = []

    for result in results:
        row: dict[str, Any] = {
            "sample_group": result.sample_group,
            "round_name": result.round_name,
            "registration_time_sec": result.registration_time_sec,
            "instance_iou_threshold": object_eval_params["instance_iou_threshold"],
            "min_valid_instance_area_px": object_eval_params["min_valid_instance_area_px"],
            "min_valid_fraction": object_eval_params["min_valid_fraction"],
        }
        row.update(result.object_metrics)
        row.update(result.diagnostics)
        main_rows.append(row)

        if result.secondary_intensity_metrics is not None:
            secondary_row = {
                "sample_group": result.sample_group,
                "round_name": result.round_name,
                "registration_time_sec": result.registration_time_sec,
            }
            secondary_row.update(result.secondary_intensity_metrics)
            secondary_rows.append(secondary_row)

    main_df = pd.DataFrame(main_rows)
    secondary_df = pd.DataFrame(secondary_rows) if secondary_rows else None
    return main_df, secondary_df


def write_results_tables(
    output_dir: Path,
    main_df: pd.DataFrame,
    secondary_df: pd.DataFrame | None = None,
) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / "benchmark_results.csv"
    main_df.to_csv(main_path, index=False)

    secondary_path: Path | None = None
    if secondary_df is not None:
        secondary_path = output_dir / "benchmark_secondary_intensity_results.csv"
        secondary_df.to_csv(secondary_path, index=False)

    return main_path, secondary_path


def compute_summary_frame(results_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        series = pd.to_numeric(results_df[metric], errors="coerce")
        rows.append(
            {
                "metric": metric,
                "median": float(series.median(skipna=True)),
                "p25": float(series.quantile(0.25)),
                "p75": float(series.quantile(0.75)),
                "mean": float(series.mean(skipna=True)),
            }
        )
    return pd.DataFrame(rows)


def plot_object_metric_panels(results_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes_flat = axes.ravel()
    labels = [f"G{row['sample_group']}-{row['round_name']}->B" for _, row in results_df.iterrows()]
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_df)))

    for index, metric in enumerate(OBJECT_PLOT_METRICS):
        ax = axes_flat[index]
        values = pd.to_numeric(results_df[metric], errors="coerce").to_numpy(dtype=float)
        bars = ax.bar(range(len(values)), np.nan_to_num(values, nan=0.0), color=colors, edgecolor="gray", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, value in zip(bars, values):
            label = "nan" if np.isnan(value) else f"{value:.3f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=7,
            )

    summary_metrics = compute_summary_frame(results_df, OBJECT_PLOT_METRICS)
    ax = axes_flat[-1]
    ax.axis("off")
    lines = [
        f"{row.metric}: median={row.median:.3f}, p25={row.p25:.3f}, p75={row.p75:.3f}, mean={row.mean:.3f}"
        for row in summary_metrics.itertuples(index=False)
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_object_heatmap(results_df: pd.DataFrame, output_path: Path) -> None:
    metrics = OBJECT_PLOT_METRICS
    labels = [f"G{row['sample_group']}-{row['round_name']}->B" for _, row in results_df.iterrows()]
    data = results_df[metrics].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    data_normalized = np.zeros_like(data)
    for column_index in range(data.shape[1]):
        column = data[:, column_index]
        finite = np.isfinite(column)
        if not np.any(finite):
            continue
        col_min = np.nanmin(column)
        col_max = np.nanmax(column)
        if abs(col_max - col_min) < 1e-10:
            data_normalized[:, column_index] = 0.5
        else:
            data_normalized[:, column_index] = (np.nan_to_num(column, nan=col_min) - col_min) / (col_max - col_min)

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5 + 2)))
    im = ax.imshow(data_normalized, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=10, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(len(labels)):
        for j in range(len(metrics)):
            value = data[i, j]
            text = "nan" if np.isnan(value) else f"{value:.3f}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if data_normalized[i, j] > 0.6 else "black",
            )

    ax.set_title("Object-Level Registration Metrics", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Normalized value")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_timing_chart(results_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [f"G{row['sample_group']}-{row['round_name']}->B" for _, row in results_df.iterrows()]
    times = pd.to_numeric(results_df["registration_time_sec"], errors="coerce").to_numpy(dtype=float)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(times)))
    bars = ax.bar(range(len(times)), np.nan_to_num(times, nan=0.0), color=colors, edgecolor="gray", linewidth=0.5)

    for bar, value in zip(bars, times):
        label = "nan" if np.isnan(value) else f"{value:.1f}s"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Registration Time per Image Pair", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    avg_time = float(np.nanmean(times)) if len(times) else float("nan")
    if np.isfinite(avg_time):
        ax.axhline(avg_time, color="red", linestyle="--", alpha=0.7, label=f"Mean: {avg_time:.1f}s")
        ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_secondary_intensity_panels(results_df: pd.DataFrame, output_path: Path) -> None:
    metrics = ["SSIM", "NCC", "MI", "PSNR"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.ravel()
    labels = [f"G{row['sample_group']}-{row['round_name']}->B" for _, row in results_df.iterrows()]
    colors = plt.cm.Accent(np.linspace(0, 1, len(results_df)))

    for index, metric in enumerate(metrics):
        ax = axes_flat[index]
        values = pd.to_numeric(results_df[metric], errors="coerce").to_numpy(dtype=float)
        ax.bar(range(len(values)), np.nan_to_num(values, nan=0.0), color=colors, edgecolor="gray", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _parse_int_list(value: str) -> list[int]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.")
    try:
        return [int(item) for item in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_round_list(value: str) -> list[str]:
    rounds = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not rounds:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of round names.")
    return rounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the object-level cell-registration benchmark.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-groups", type=_parse_int_list, default=None, help="Comma-separated sample groups.")
    parser.add_argument("--rounds", type=_parse_round_list, default=None, help="Comma-separated rounds, e.g. A,C,D,E.")
    parser.add_argument("--feature-weight", type=float, default=DEFAULT_REG_PARAMS["feature_weight"])
    parser.add_argument("--topology-weight", type=float, default=DEFAULT_REG_PARAMS["topology_weight"])
    parser.add_argument("--position-weight", type=float, default=DEFAULT_REG_PARAMS["position_weight"])
    parser.add_argument("--distance-threshold", type=float, default=DEFAULT_REG_PARAMS["distance_threshold"])
    parser.add_argument("--spatial-window-size", type=float, default=DEFAULT_REG_PARAMS["spatial_window_size"])
    parser.add_argument("--top-k", type=int, default=DEFAULT_REG_PARAMS["top_k"])
    parser.add_argument("--cellpose-diameter", type=float, default=DEFAULT_REG_PARAMS["cellpose_diameter"])
    parser.add_argument("--cellpose-flow-threshold", type=float, default=DEFAULT_REG_PARAMS["cellpose_flow_threshold"])
    parser.add_argument(
        "--cellpose-cellprob-threshold",
        type=float,
        default=DEFAULT_REG_PARAMS["cellpose_cellprob_threshold"],
    )
    parser.add_argument("--cellpose-min-size", type=int, default=DEFAULT_REG_PARAMS["cellpose_min_size"])
    parser.add_argument("--min-cells-for-two-stage", type=int, default=DEFAULT_REG_PARAMS["min_cells_for_two_stage"])
    parser.add_argument("--coarse-top-k", type=int, default=DEFAULT_REG_PARAMS["coarse_top_k"])
    parser.add_argument(
        "--coarse-distance-threshold",
        type=float,
        default=DEFAULT_REG_PARAMS["coarse_distance_threshold"],
    )
    parser.add_argument(
        "--coarse-matching-mode",
        choices=("global", "patch"),
        default=DEFAULT_REG_PARAMS["coarse_matching_mode"],
        help="Matcher used only during coarse matching before offset estimation.",
    )
    parser.add_argument(
        "--coarse-patch-rows",
        type=int,
        default=DEFAULT_REG_PARAMS["coarse_patch_rows"],
    )
    parser.add_argument(
        "--coarse-patch-cols",
        type=int,
        default=DEFAULT_REG_PARAMS["coarse_patch_cols"],
    )
    parser.add_argument(
        "--coarse-patch-top-k-per-patch",
        type=int,
        default=DEFAULT_REG_PARAMS["coarse_patch_top_k_per_patch"],
        help="Optional per-patch cap for coarse patch matching; <=0 falls back to automatic allocation.",
    )
    parser.add_argument(
        "--coarse-image-enabled",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REG_PARAMS["coarse_image_enabled"],
        help="Run image-level coarse rigid initialization before cell matching.",
    )
    parser.add_argument(
        "--coarse-image-target-max-dim",
        type=int,
        default=DEFAULT_REG_PARAMS["coarse_image_target_max_dim"],
    )
    parser.add_argument(
        "--coarse-image-crop-ratio",
        type=float,
        default=DEFAULT_REG_PARAMS["coarse_image_crop_ratio"],
    )
    parser.add_argument(
        "--coarse-image-upsample-factor",
        type=int,
        default=DEFAULT_REG_PARAMS["coarse_image_upsample_factor"],
    )
    parser.add_argument(
        "--coarse-image-min-score",
        type=float,
        default=DEFAULT_REG_PARAMS["coarse_image_min_score"],
    )
    parser.add_argument("--neighbor-k", type=int, default=DEFAULT_REG_PARAMS["neighbor_k"])
    parser.add_argument("--neighbor-weight", type=float, default=DEFAULT_REG_PARAMS["neighbor_weight"])
    parser.add_argument("--landmark-weight", type=float, default=DEFAULT_REG_PARAMS["landmark_weight"])
    parser.add_argument("--max-theta-deg", type=float, default=DEFAULT_REG_PARAMS["max_theta_deg"])
    parser.add_argument("--max-translation", type=float, default=DEFAULT_REG_PARAMS["max_translation"])
    parser.add_argument("--max-scale-change", type=float, default=DEFAULT_REG_PARAMS["max_scale_change"])
    parser.add_argument(
        "--guided-min-position-weight",
        type=float,
        default=DEFAULT_REG_PARAMS["guided_min_position_weight"],
    )
    parser.add_argument(
        "--guided-spatial-window-cap",
        type=float,
        default=DEFAULT_REG_PARAMS["guided_spatial_window_cap"],
        help="Cap (pixels) for guided rematch spatial window. <=0 disables capping.",
    )
    parser.add_argument(
        "--guided-top-k",
        type=int,
        default=DEFAULT_REG_PARAMS["guided_top_k"],
        help="Optional top-k cap used only during guided rematch; <=0 falls back to --top-k.",
    )
    parser.add_argument(
        "--guided-distance-relaxation",
        type=float,
        default=DEFAULT_REG_PARAMS["guided_distance_relaxation"],
        help="Extra distance threshold slack added only during guided rematch.",
    )
    parser.add_argument(
        "--guided-matching-mode",
        choices=("global", "patch"),
        default=DEFAULT_REG_PARAMS["guided_matching_mode"],
        help="Matcher used during guided rematch only.",
    )
    parser.add_argument(
        "--guided-patch-rows",
        type=int,
        default=DEFAULT_REG_PARAMS["guided_patch_rows"],
    )
    parser.add_argument(
        "--guided-patch-cols",
        type=int,
        default=DEFAULT_REG_PARAMS["guided_patch_cols"],
    )
    parser.add_argument(
        "--guided-patch-top-k-per-patch",
        type=int,
        default=DEFAULT_REG_PARAMS["guided_patch_top_k_per_patch"],
        help="Optional per-patch cap for guided patch matching; <=0 falls back to automatic allocation.",
    )
    parser.add_argument(
        "--guided-residual-clip-mad-factor",
        type=float,
        default=DEFAULT_REG_PARAMS["guided_residual_clip_mad_factor"],
        help="MAD multiplier used to clip high guided inlier residuals before refit; <=0 disables.",
    )
    parser.add_argument(
        "--guided-residual-clip-min-inliers",
        type=int,
        default=DEFAULT_REG_PARAMS["guided_residual_clip_min_inliers"],
    )
    parser.add_argument(
        "--guided-residual-clip-max-drop-fraction",
        type=float,
        default=DEFAULT_REG_PARAMS["guided_residual_clip_max_drop_fraction"],
    )
    parser.add_argument(
        "--guided-residual-clip-min-median-gain-px",
        type=float,
        default=DEFAULT_REG_PARAMS["guided_residual_clip_min_median_gain_px"],
    )
    parser.add_argument(
        "--validation-min-confidence",
        type=float,
        default=DEFAULT_REG_PARAMS["validation_min_confidence"],
    )
    parser.add_argument(
        "--validation-max-feature-diff",
        type=float,
        default=DEFAULT_REG_PARAMS["validation_max_feature_diff"],
    )
    parser.add_argument(
        "--validation-neighbor-k",
        type=int,
        default=DEFAULT_REG_PARAMS["validation_neighbor_k"],
        help="Matched-neighbor profile size used for local geometry validation; <=0 disables.",
    )
    parser.add_argument(
        "--validation-max-neighbor-profile-diff",
        type=float,
        default=DEFAULT_REG_PARAMS["validation_max_neighbor_profile_diff"],
        help="Maximum median relative difference allowed between local matched-neighbor distance profiles.",
    )
    parser.add_argument(
        "--validation-ambiguity-ratio",
        type=float,
        default=DEFAULT_REG_PARAMS["validation_ambiguity_ratio"],
        help="Maximum allowed ratio between a chosen match distance and its next-best feasible alternative; <=0 disables.",
    )
    parser.add_argument(
        "--validation-ambiguity-min-gap",
        type=float,
        default=DEFAULT_REG_PARAMS["validation_ambiguity_min_gap"],
        help="Minimum distance gap required against the next-best feasible alternative; <=0 disables.",
    )
    parser.add_argument(
        "--allow-scale",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REG_PARAMS["allow_scale"],
        help="Allow similarity-scale estimation instead of rigid rotation+translation only.",
    )
    parser.add_argument(
        "--prefer-affine",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REG_PARAMS["prefer_affine"],
    )
    parser.add_argument("--ransac-max-trials", type=int, default=DEFAULT_REG_PARAMS["ransac_max_trials"])
    parser.add_argument("--ransac-residual-threshold", type=float, default=DEFAULT_REG_PARAMS["ransac_residual_threshold"])
    parser.add_argument(
        "--similarity-residual-threshold",
        type=float,
        default=DEFAULT_REG_PARAMS["similarity_residual_threshold"],
    )
    parser.add_argument(
        "--instance-iou-threshold",
        type=float,
        default=DEFAULT_OBJECT_EVAL_PARAMS["instance_iou_threshold"],
    )
    parser.add_argument(
        "--min-valid-instance-area-px",
        type=int,
        default=DEFAULT_OBJECT_EVAL_PARAMS["min_valid_instance_area_px"],
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=DEFAULT_OBJECT_EVAL_PARAMS["min_valid_fraction"],
    )
    parser.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-registered-images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-match-tables", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--secondary-intensity-eval", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _save_diagnostic_manifest(
    output_path: Path,
    pair: ImagePair,
    object_metrics: dict[str, float],
    diagnostics: dict[str, object],
    object_eval_params: dict[str, Any],
    secondary_intensity_metrics: dict[str, float] | None,
) -> None:
    payload = {
        "sample_group": pair.sample_group,
        "round_name": pair.round_name,
        "source_path": str(pair.source_path),
        "reference_path": str(pair.reference_path),
        "comparison_path": str(pair.comparison_path) if pair.comparison_path is not None else None,
        "object_eval_params": object_eval_params,
        "object_metrics": object_metrics,
        "secondary_intensity_metrics": secondary_intensity_metrics,
        "diagnostics": diagnostics,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    _prepare_torch_runtime()
    if NAPARI_CORE_IMPORT_ERROR is not None:
        raise ImportError(
            "Benchmark runtime dependencies are missing. Install cellpose and related registration dependencies."
        ) from NAPARI_CORE_IMPORT_ERROR

    output_dir = args.output_dir.resolve()
    sample_groups = args.sample_groups or DEFAULT_SAMPLE_GROUPS
    rounds = args.rounds or DEFAULT_ROUNDS
    reg_params = {
        "cellpose_diameter": args.cellpose_diameter,
        "cellpose_flow_threshold": args.cellpose_flow_threshold,
        "cellpose_cellprob_threshold": args.cellpose_cellprob_threshold,
        "cellpose_min_size": args.cellpose_min_size,
        "feature_weight": args.feature_weight,
        "topology_weight": args.topology_weight,
        "position_weight": args.position_weight,
        "distance_threshold": args.distance_threshold,
        "spatial_window_size": args.spatial_window_size,
        "top_k": args.top_k,
        "min_cells_for_two_stage": args.min_cells_for_two_stage,
        "coarse_top_k": args.coarse_top_k,
        "coarse_distance_threshold": args.coarse_distance_threshold,
        "coarse_matching_mode": args.coarse_matching_mode,
        "coarse_patch_rows": args.coarse_patch_rows,
        "coarse_patch_cols": args.coarse_patch_cols,
        "coarse_patch_top_k_per_patch": args.coarse_patch_top_k_per_patch,
        "coarse_image_enabled": args.coarse_image_enabled,
        "coarse_image_target_max_dim": args.coarse_image_target_max_dim,
        "coarse_image_crop_ratio": args.coarse_image_crop_ratio,
        "coarse_image_upsample_factor": args.coarse_image_upsample_factor,
        "coarse_image_min_score": args.coarse_image_min_score,
        "neighbor_k": args.neighbor_k,
        "neighbor_weight": args.neighbor_weight,
        "landmark_weight": args.landmark_weight,
        "max_theta_deg": args.max_theta_deg,
        "max_translation": args.max_translation,
        "max_scale_change": args.max_scale_change,
        "guided_min_position_weight": args.guided_min_position_weight,
        "guided_spatial_window_cap": args.guided_spatial_window_cap,
        "guided_top_k": args.guided_top_k,
        "guided_distance_relaxation": args.guided_distance_relaxation,
        "guided_matching_mode": args.guided_matching_mode,
        "guided_patch_rows": args.guided_patch_rows,
        "guided_patch_cols": args.guided_patch_cols,
        "guided_patch_top_k_per_patch": args.guided_patch_top_k_per_patch,
        "guided_residual_clip_mad_factor": args.guided_residual_clip_mad_factor,
        "guided_residual_clip_min_inliers": args.guided_residual_clip_min_inliers,
        "guided_residual_clip_max_drop_fraction": args.guided_residual_clip_max_drop_fraction,
        "guided_residual_clip_min_median_gain_px": args.guided_residual_clip_min_median_gain_px,
        "validation_min_confidence": args.validation_min_confidence,
        "validation_max_feature_diff": args.validation_max_feature_diff,
        "validation_neighbor_k": args.validation_neighbor_k,
        "validation_max_neighbor_profile_diff": args.validation_max_neighbor_profile_diff,
        "validation_ambiguity_ratio": args.validation_ambiguity_ratio,
        "validation_ambiguity_min_gap": args.validation_ambiguity_min_gap,
        "allow_scale": args.allow_scale,
        "prefer_affine": bool(args.allow_scale and args.prefer_affine),
        "ransac_max_trials": args.ransac_max_trials,
        "ransac_residual_threshold": args.ransac_residual_threshold,
        "similarity_residual_threshold": args.similarity_residual_threshold,
    }
    object_eval_params = {
        "instance_iou_threshold": args.instance_iou_threshold,
        "min_valid_instance_area_px": args.min_valid_instance_area_px,
        "min_valid_fraction": args.min_valid_fraction,
    }

    if args.prefer_affine and not args.allow_scale:
        print("  Note: --prefer-affine ignored because --allow-scale is disabled.")

    print("=" * 70)
    print("  Cell Registration Object-Level Benchmark")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = output_dir / "benchmark_params.json"
    with open(params_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "sample_groups": sample_groups,
                "rounds": rounds,
                "reg_params": reg_params,
                "object_eval_params": object_eval_params,
                "secondary_intensity_eval": bool(args.secondary_intensity_eval),
            },
            handle,
            indent=2,
        )
    print(f"  Output dir: {output_dir}")
    print(f"  Params saved to: {params_path.name}")

    print("\n[1/5] Discovering image pairs...")
    pairs = discover_image_pairs(
        sample_groups,
        rounds,
        require_secondary_comparison=bool(args.secondary_intensity_eval),
    )
    print(f"  Found {len(pairs)} registration pairs")
    for pair in pairs:
        print(f"    Group {pair.sample_group}: {pair.round_name} -> B | src: {pair.source_path.name}")
    if not pairs:
        print("  No image pairs found. Check directory structure.")
        return

    print("\n[2/5] Initializing Cellpose-SAM segmenter (GPU=True)...")
    segmenter = CellposeSegmenter(
        CellposeConfig(
            gpu=True,
            pretrained_model="cpsam",
            diameter=reg_params["cellpose_diameter"],
            flow_threshold=reg_params["cellpose_flow_threshold"],
            cellprob_threshold=reg_params["cellpose_cellprob_threshold"],
            min_size=reg_params["cellpose_min_size"],
        )
    )
    print("  Segmenter ready")

    segmentation_cache: dict[Path, SegmentationArtifacts] = {}
    results: list[BenchmarkResult] = []

    print("\n[3/5] Running registrations...")
    for index, pair in enumerate(pairs, start=1):
        print(f"\n  [{index}/{len(pairs)}] Group {pair.sample_group} - {pair.round_name} -> B")
        print(f"    Source:    {pair.source_path.name}")
        print(f"    Reference: {pair.reference_path.name}")
        if pair.comparison_path is not None:
            print(f"    Secondary: {pair.comparison_path.name} [channel 5]")

        source_artifacts = get_or_create_segmentation(pair.source_path, segmenter, segmentation_cache)
        reference_artifacts = get_or_create_segmentation(pair.reference_path, segmenter, segmentation_cache)
        print(
            "    Image shapes: "
            f"source={source_artifacts.image.shape}, reference={reference_artifacts.image.shape}"
        )

        registration = run_registration(source_artifacts, reference_artifacts, reg_params)
        print(f"    Registration time: {registration.elapsed_seconds:.2f}s")

        object_metrics_result = compute_object_registration_metrics(
            moving_mask=registration.moving_mask,
            fixed_mask=registration.fixed_mask,
            moving_features=registration.moving_features,
            fixed_features=registration.fixed_features,
            transform=registration.transform,
            valid_mask=registration.valid_mask,
            instance_iou_threshold=object_eval_params["instance_iou_threshold"],
            min_valid_instance_area_px=object_eval_params["min_valid_instance_area_px"],
            min_valid_fraction=object_eval_params["min_valid_fraction"],
            return_match_table=bool(args.save_match_tables),
        )
        if args.save_match_tables:
            object_metrics, match_table = object_metrics_result  # type: ignore[misc]
        else:
            object_metrics = object_metrics_result  # type: ignore[assignment]
            match_table = None

        print(
            "    Object metrics: "
            f"match_f1={_fmt_metric(object_metrics['match_f1'])} | "
            f"matched_mean_iou={_fmt_metric(object_metrics['matched_mean_iou'])} | "
            f"centroid_median={_fmt_metric(object_metrics['centroid_error_median_px'])} | "
            f"mask_dice={_fmt_metric(object_metrics['mask_dice'])}"
        )

        secondary_intensity_metrics: dict[str, float] | None = None
        if args.secondary_intensity_eval:
            if pair.comparison_path is None:
                raise ValueError("secondary_intensity_eval is enabled but comparison_path is missing.")
            reference_b_ch5 = load_b_reference_channel5(pair.comparison_path)
            secondary_intensity_metrics = compute_secondary_intensity_metrics(
                registration.registered_image,
                reference_b_ch5,
                valid_mask=registration.valid_mask,
            )
            print(
                "    Secondary intensity: "
                f"SSIM={_fmt_metric(secondary_intensity_metrics['SSIM'])} | "
                f"MI={_fmt_metric(secondary_intensity_metrics['MI'])} | "
                f"overlap={_fmt_metric(secondary_intensity_metrics['valid_overlap_ratio'])}"
            )

        if args.save_registered_images:
            reg_output_path = output_dir / f"registered_G{pair.sample_group}_{pair.round_name}_to_B.tif"
            imwrite(str(reg_output_path), registration.registered_image)

        if match_table is not None:
            match_table_path = output_dir / f"matches_G{pair.sample_group}_{pair.round_name}_to_B.csv"
            match_table.to_csv(match_table_path, index=False)

        if args.save_diagnostics:
            diag_path = output_dir / f"diagnostic_G{pair.sample_group}_{pair.round_name}_to_B.json"
            _save_diagnostic_manifest(
                diag_path,
                pair,
                object_metrics=object_metrics,
                diagnostics=registration.diagnostics,
                object_eval_params=object_eval_params,
                secondary_intensity_metrics=secondary_intensity_metrics,
            )

        results.append(
            BenchmarkResult(
                sample_group=pair.sample_group,
                round_name=pair.round_name,
                registration_time_sec=registration.elapsed_seconds,
                object_metrics=object_metrics,
                diagnostics=registration.diagnostics,
                secondary_intensity_metrics=secondary_intensity_metrics,
                registered_image=registration.registered_image if (args.save_registered_images or args.save_plots) else None,
                valid_mask=registration.valid_mask if args.save_plots else None,
                match_table=match_table,
            )
        )

    print("\n[4/5] Writing reports...")
    main_df, secondary_df = build_results_tables(results, object_eval_params)
    main_csv_path, secondary_csv_path = write_results_tables(output_dir, main_df, secondary_df)
    print(f"  Saved main results CSV: {main_csv_path.name}")
    if secondary_csv_path is not None:
        print(f"  Saved secondary intensity CSV: {secondary_csv_path.name}")

    summary_df = compute_summary_frame(main_df, OBJECT_PLOT_METRICS)
    summary_path = output_dir / "benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved summary CSV: {summary_path.name}")

    if args.save_plots:
        plot_object_metric_panels(main_df, output_dir / "object_metrics_panel.png")
        plot_object_heatmap(main_df, output_dir / "object_metrics_heatmap.png")
        plot_timing_chart(main_df, output_dir / "timing_chart.png")
        if secondary_df is not None:
            plot_secondary_intensity_panels(secondary_df, output_dir / "secondary_intensity_panel.png")

    print("\n[5/5] Summary")
    print("=" * 90)
    print(
        f"{'Pair':<15} {'F1':<10} {'IoU':<10} {'CentMed':<10} "
        f"{'CentP95':<10} {'Dice':<10} {'Time(s)':<10}"
    )
    print("-" * 90)
    for _, row in main_df.iterrows():
        label = f"G{int(row['sample_group'])}-{row['round_name']}->B"
        print(
            f"{label:<15} {_fmt_metric(row['match_f1']):<10} {_fmt_metric(row['matched_mean_iou']):<10} "
            f"{_fmt_metric(row['centroid_error_median_px']):<10} {_fmt_metric(row['centroid_error_p95_px']):<10} "
            f"{_fmt_metric(row['mask_dice']):<10} {_fmt_metric(row['registration_time_sec'], precision=2):<10}"
        )
    print("-" * 90)
    for summary_row in summary_df.itertuples(index=False):
        print(
            f"{summary_row.metric}: median={_fmt_metric(summary_row.median)} | "
            f"p25={_fmt_metric(summary_row.p25)} | p75={_fmt_metric(summary_row.p75)} | "
            f"mean={_fmt_metric(summary_row.mean)}"
        )
    print("=" * 90)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
