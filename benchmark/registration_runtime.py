"""
Shared runtime helpers for benchmark registration scripts.

This module centralizes dependency bootstrapping, segmentation caching, and the
object-level registration pipeline so multiple benchmark entrypoints stay in
sync.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
NAPARI_SRC = PROJECT_ROOT / "napari-cell-registration" / "src"
if str(NAPARI_SRC) not in sys.path:
    sys.path.insert(0, str(NAPARI_SRC))


def _configure_torch_runtime_path() -> None:
    torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if not os.path.isdir(torch_lib):
        torch_lib = os.path.join(sys.prefix, "lib", "site-packages", "torch", "lib")
    if os.path.isdir(torch_lib):
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(torch_lib)


_configure_torch_runtime_path()
try:
    import torch
except Exception:
    torch = None  # type: ignore[assignment]

import numpy as np
import pandas as pd
from scipy.ndimage import shift as ndimage_shift
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, estimate_transform, resize
from tifffile import imread

from cell_registration.evaluation import compute_overlap_metrics

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
    from napari_cell_registration.core.matching import (
        apply_transform_to_features,
        match_cells_per_patch,
        two_stage_match_cells,
    )
    from napari_cell_registration.core.point_registration import (
        RobustTransformResult,
        compute_valid_overlap_mask,
        estimate_robust_transform,
        refine_transform_with_neighbors,
        warp_image_with_transform,
    )
    from napari_cell_registration.core.validation import validate_matches
except ImportError as exc:
    NAPARI_CORE_IMPORT_ERROR = exc
    CellFeaturesConfig = object  # type: ignore[assignment]
    CellposeConfig = object  # type: ignore[assignment]
    CellposeSegmenter = object  # type: ignore[assignment]
    MatchingConfig = object  # type: ignore[assignment]

    def _missing_dependency(*args, **kwargs):
        raise ImportError(
            "Benchmark runtime dependencies are missing. Install cellpose and related registration dependencies."
        ) from NAPARI_CORE_IMPORT_ERROR

    compute_cell_features = _missing_dependency  # type: ignore[assignment]
    greedy_match_cells = _missing_dependency  # type: ignore[assignment]
    apply_transform_to_features = _missing_dependency  # type: ignore[assignment]
    match_cells_per_patch = _missing_dependency  # type: ignore[assignment]
    two_stage_match_cells = _missing_dependency  # type: ignore[assignment]
    compute_valid_overlap_mask = _missing_dependency  # type: ignore[assignment]
    RobustTransformResult = object  # type: ignore[assignment]
    estimate_robust_transform = _missing_dependency  # type: ignore[assignment]
    refine_transform_with_neighbors = _missing_dependency  # type: ignore[assignment]
    warp_image_with_transform = _missing_dependency  # type: ignore[assignment]
    validate_matches = _missing_dependency  # type: ignore[assignment]


DEFAULT_REG_PARAMS = {
    "cellpose_diameter": 10.0,
    "cellpose_flow_threshold": 0.0,
    "cellpose_cellprob_threshold": 0.0,
    "cellpose_min_size": 15,
    "feature_weight": 1.0,
    "topology_weight": 0.0,
    "position_weight": 3.0,
    "distance_threshold": 2.5,
    "spatial_window_size": 100.0,
    "top_k": 100,
    "min_cells_for_two_stage": 20,
    "coarse_top_k": 50,
    "coarse_distance_threshold": 2.0,
    "coarse_matching_mode": "global",
    "coarse_patch_rows": 3,
    "coarse_patch_cols": 3,
    "coarse_patch_top_k_per_patch": None,
    "coarse_image_enabled": False,
    "coarse_image_target_max_dim": 256,
    "coarse_image_crop_ratio": 0.85,
    "coarse_image_upsample_factor": 10,
    "coarse_image_min_score": 0.05,
    "neighbor_k": 5,
    "neighbor_weight": 1.0,
    "landmark_weight": 10.0,
    "max_theta_deg": 10.0,
    "max_translation": 30.0,
    "max_scale_change": 0.05,
    "guided_min_position_weight": 4.0,
    "guided_spatial_window_cap": 60.0,
    "guided_top_k": None,
    "guided_distance_relaxation": 0.5,
    "guided_matching_mode": "global",
    "guided_patch_rows": 4,
    "guided_patch_cols": 4,
    "guided_patch_top_k_per_patch": None,
    "guided_residual_clip_mad_factor": 0.0,
    "guided_residual_clip_min_inliers": 12,
    "guided_residual_clip_max_drop_fraction": 0.2,
    "guided_residual_clip_min_median_gain_px": 0.1,
    "validation_min_confidence": 0.2,
    "validation_max_feature_diff": 0.7,
    "validation_neighbor_k": 0,
    "validation_max_neighbor_profile_diff": 0.35,
    "validation_ambiguity_ratio": 0.0,
    "validation_ambiguity_min_gap": 0.0,
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


@dataclass
class SegmentationArtifacts:
    image_path: Path
    image: np.ndarray = field(repr=False)
    mask: np.ndarray = field(repr=False)
    features: pd.DataFrame = field(repr=False)
    segmentation_seconds: float = 0.0


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
    match_table: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)


@dataclass
class ImageCoarseAlignmentResult:
    transform: AffineTransform = field(repr=False)
    used: bool
    rotation_deg: float
    translation_xy: np.ndarray = field(repr=False)
    score: float
    error: float
    evaluated_angles: int
    elapsed_seconds: float = 0.0


def prepare_torch_runtime() -> None:
    """Load torch lazily so CLI --help works without initializing the full runtime."""
    _configure_torch_runtime_path()
    __import__("torch")


def ensure_registration_dependencies() -> None:
    if NAPARI_CORE_IMPORT_ERROR is not None:
        raise ImportError(
            "Benchmark runtime dependencies are missing. Install cellpose and related registration dependencies."
        ) from NAPARI_CORE_IMPORT_ERROR


def get_environment_snapshot() -> dict[str, Any]:
    try:
        import cellpose
    except Exception:
        cellpose = None  # type: ignore[assignment]

    try:
        import skimage
    except Exception:
        skimage = None  # type: ignore[assignment]

    try:
        import tifffile
    except Exception:
        tifffile = None  # type: ignore[assignment]

    try:
        import pandas as _pandas
    except Exception:
        _pandas = None  # type: ignore[assignment]

    cuda_available = False
    if torch is not None:
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False

    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "sys_prefix": sys.prefix,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "torch_version": getattr(torch, "__version__", None) if torch is not None else None,
        "cellpose_version": getattr(cellpose, "__version__", None) if cellpose is not None else None,
        "skimage_version": getattr(skimage, "__version__", None) if skimage is not None else None,
        "tifffile_version": getattr(tifffile, "__version__", None) if tifffile is not None else None,
        "pandas_version": getattr(_pandas, "__version__", None) if _pandas is not None else None,
        "cuda_available": cuda_available,
    }


def build_segmenter(reg_params: dict[str, Any], *, gpu: bool = True) -> CellposeSegmenter:
    ensure_registration_dependencies()
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


def _normalize_for_image_coarse_alignment(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(to_2d_gray(img), dtype=np.float64)
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float64)
    lo = float(np.percentile(finite, 5.0))
    hi = float(np.percentile(finite, 99.5))
    if hi <= lo + 1e-8:
        hi = float(np.max(finite))
    arr = arr - lo
    scale = max(hi - lo, 1e-8)
    arr = np.clip(arr / scale, 0.0, 1.0)
    return arr


def _resize_for_image_coarse_alignment(
    fixed_img: np.ndarray,
    moving_img: np.ndarray,
    target_max_dim: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    max_dim = max(fixed_img.shape[0], fixed_img.shape[1], moving_img.shape[0], moving_img.shape[1])
    if target_max_dim <= 0 or max_dim <= target_max_dim:
        return fixed_img, moving_img, 1.0

    scale = float(target_max_dim) / float(max_dim)
    fixed_shape = (
        max(32, int(round(fixed_img.shape[0] * scale))),
        max(32, int(round(fixed_img.shape[1] * scale))),
    )
    moving_shape = (
        max(32, int(round(moving_img.shape[0] * scale))),
        max(32, int(round(moving_img.shape[1] * scale))),
    )
    fixed_small = resize(
        fixed_img,
        fixed_shape,
        order=1,
        anti_aliasing=True,
        preserve_range=True,
    )
    moving_small = resize(
        moving_img,
        moving_shape,
        order=1,
        anti_aliasing=True,
        preserve_range=True,
    )
    return np.asarray(fixed_small, dtype=np.float64), np.asarray(moving_small, dtype=np.float64), scale


def _crop_center(img: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    crop_h = max(1, min(int(crop_h), h))
    crop_w = max(1, min(int(crop_w), w))
    y0 = max((h - crop_h) // 2, 0)
    x0 = max((w - crop_w) // 2, 0)
    return img[y0 : y0 + crop_h, x0 : x0 + crop_w]


def _central_overlap_crops(
    fixed_img: np.ndarray,
    moving_img: np.ndarray,
    crop_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    h = min(fixed_img.shape[0], moving_img.shape[0])
    w = min(fixed_img.shape[1], moving_img.shape[1])
    if crop_ratio <= 0.0 or crop_ratio >= 1.0:
        crop_h = h
        crop_w = w
    else:
        crop_h = max(32, int(round(h * crop_ratio)))
        crop_w = max(32, int(round(w * crop_ratio)))
    return _crop_center(fixed_img, crop_h, crop_w), _crop_center(moving_img, crop_h, crop_w)


def _normalized_cross_correlation(fixed_img: np.ndarray, moving_img: np.ndarray) -> float:
    mask = (fixed_img > 0.05) | (moving_img > 0.05)
    if int(np.count_nonzero(mask)) < 128:
        mask = np.ones_like(fixed_img, dtype=bool)

    fixed_vals = fixed_img[mask].astype(np.float64, copy=False)
    moving_vals = moving_img[mask].astype(np.float64, copy=False)
    if fixed_vals.size == 0 or moving_vals.size == 0:
        return -1.0

    fixed_vals = fixed_vals - float(np.mean(fixed_vals))
    moving_vals = moving_vals - float(np.mean(moving_vals))
    denom = float(np.linalg.norm(fixed_vals) * np.linalg.norm(moving_vals))
    if denom <= 1e-8:
        return -1.0
    return float(np.dot(fixed_vals, moving_vals) / denom)


def _add_patch_coordinates(
    df: pd.DataFrame,
    image_shape: tuple[int, int],
    patch_rows: int,
    patch_cols: int,
) -> pd.DataFrame:
    out = df.copy()
    h, w = image_shape[:2]
    x = out["centroid_x"].to_numpy(dtype=float)
    y = out["centroid_y"].to_numpy(dtype=float)
    out["patch_x"] = np.clip(
        np.floor(x * float(max(patch_cols, 1)) / float(max(w, 1))).astype(int),
        0,
        max(patch_cols - 1, 0),
    )
    out["patch_y"] = np.clip(
        np.floor(y * float(max(patch_rows, 1)) / float(max(h, 1))).astype(int),
        0,
        max(patch_rows - 1, 0),
    )
    return out


def _resolve_patch_top_k(
    global_top_k: int,
    patch_rows: int,
    patch_cols: int,
    patch_top_k_per_patch: int | None,
) -> int:
    if patch_top_k_per_patch is not None and patch_top_k_per_patch > 0:
        return int(patch_top_k_per_patch)
    patches = max(int(patch_rows) * int(patch_cols), 1)
    return max(1, int(np.ceil(float(global_top_k) / float(patches))))


def _point_residuals(transform: AffineTransform, pts_moving_xy: np.ndarray, pts_fixed_xy: np.ndarray) -> np.ndarray:
    pts_reg = transform(pts_moving_xy)
    return np.linalg.norm(pts_reg - pts_fixed_xy, axis=1)


def _refit_transform_from_mask(
    pts_fixed_xy: np.ndarray,
    pts_moving_xy: np.ndarray,
    inlier_mask: np.ndarray,
    method: str,
) -> AffineTransform | None:
    keep = np.asarray(inlier_mask, dtype=bool)
    if int(np.count_nonzero(keep)) < 3:
        return None

    method_key = str(method).strip().lower()
    if method_key == "rigid":
        estimate_method = "euclidean"
    elif method_key == "similarity":
        estimate_method = "similarity"
    elif method_key == "affine":
        estimate_method = "affine"
    elif method_key == "translation":
        translation = np.median(pts_fixed_xy[keep] - pts_moving_xy[keep], axis=0)
        return AffineTransform(translation=(float(translation[0]), float(translation[1])))
    else:
        return None

    try:
        refit = estimate_transform(estimate_method, pts_moving_xy[keep], pts_fixed_xy[keep])
    except Exception:
        return None
    return AffineTransform(matrix=np.asarray(refit.params, dtype=float))


def _maybe_apply_guided_residual_clip(
    robust: RobustTransformResult,
    pts_fixed_xy: np.ndarray,
    pts_moving_xy: np.ndarray,
    reg_params: dict[str, Any],
) -> tuple[RobustTransformResult, dict[str, Any]]:
    diagnostics = {
        "guided_residual_clip_used": False,
        "guided_residual_clip_limit": float("nan"),
        "guided_residual_clip_kept_inliers": int(robust.inlier_count),
        "guided_residual_clip_median_before": float(robust.median_inlier_residual),
        "guided_residual_clip_median_after": float(robust.median_inlier_residual),
    }

    mad_factor = float(reg_params.get("guided_residual_clip_mad_factor", 0.0) or 0.0)
    min_inliers = max(3, int(reg_params.get("guided_residual_clip_min_inliers", 12) or 12))
    max_drop_fraction = float(reg_params.get("guided_residual_clip_max_drop_fraction", 0.2) or 0.0)
    min_gain_px = float(reg_params.get("guided_residual_clip_min_median_gain_px", 0.1) or 0.0)
    if mad_factor <= 0 or robust.inlier_count < min_inliers:
        return robust, diagnostics

    inlier_residuals = robust.residuals[robust.inliers]
    if inlier_residuals.size < min_inliers:
        return robust, diagnostics

    median_residual = float(np.median(inlier_residuals))
    mad = float(np.median(np.abs(inlier_residuals - median_residual)))
    scale = max(1.4826 * mad, 1e-6)
    clip_limit = median_residual + mad_factor * scale
    diagnostics["guided_residual_clip_limit"] = float(clip_limit)

    clipped_seed_mask = np.asarray(robust.inliers, dtype=bool) & (robust.residuals <= clip_limit)
    kept_inliers = int(np.count_nonzero(clipped_seed_mask))
    diagnostics["guided_residual_clip_kept_inliers"] = kept_inliers
    if kept_inliers < min_inliers:
        return robust, diagnostics

    max_drop = max(1, int(np.floor(float(robust.inlier_count) * max_drop_fraction)))
    if kept_inliers < int(robust.inlier_count) - max_drop:
        return robust, diagnostics

    clipped_transform = _refit_transform_from_mask(
        pts_fixed_xy,
        pts_moving_xy,
        clipped_seed_mask,
        robust.method,
    )
    if clipped_transform is None:
        return robust, diagnostics

    clipped_residuals = _point_residuals(clipped_transform, pts_moving_xy, pts_fixed_xy)
    residual_threshold = (
        float(reg_params["similarity_residual_threshold"])
        if robust.method == "similarity"
        else float(reg_params["ransac_residual_threshold"])
    )
    clipped_inliers = clipped_residuals <= residual_threshold
    clipped = RobustTransformResult(
        transform=clipped_transform,
        inliers=clipped_inliers,
        residuals=clipped_residuals,
        method=robust.method,
    )
    diagnostics["guided_residual_clip_median_after"] = float(clipped.median_inlier_residual)

    if clipped.inlier_count < min_inliers:
        return robust, diagnostics

    inlier_drop = int(robust.inlier_count) - int(clipped.inlier_count)
    median_gain = float(robust.median_inlier_residual) - float(clipped.median_inlier_residual)
    if inlier_drop > max_drop or median_gain < min_gain_px:
        return robust, diagnostics

    diagnostics["guided_residual_clip_used"] = True
    return clipped, diagnostics


def _match_cells_with_strategy(
    df_fixed: pd.DataFrame,
    df_moving: pd.DataFrame,
    config: MatchingConfig,
    *,
    mode: str,
    image_shape: tuple[int, int],
    patch_rows: int,
    patch_cols: int,
    patch_top_k_per_patch: int | None,
) -> pd.DataFrame:
    strategy = str(mode).strip().lower()
    if strategy != "patch":
        return greedy_match_cells(df_fixed, df_moving, config)

    patch_rows = max(1, int(patch_rows))
    patch_cols = max(1, int(patch_cols))
    fixed_patched = _add_patch_coordinates(df_fixed, image_shape, patch_rows, patch_cols)
    moving_patched = _add_patch_coordinates(df_moving, image_shape, patch_rows, patch_cols)
    local_top_k = _resolve_patch_top_k(config.top_k, patch_rows, patch_cols, patch_top_k_per_patch)
    matches = match_cells_per_patch(
        fixed_patched,
        moving_patched,
        config,
        top_k_per_patch=local_top_k,
    )
    if len(matches) > config.top_k:
        matches = matches.sort_values("distance", ascending=True).head(config.top_k).reset_index(drop=True)
    return matches.reset_index(drop=True)


def _estimate_image_coarse_alignment(
    source_img: np.ndarray,
    reference_img: np.ndarray,
    reg_params: dict[str, Any],
) -> ImageCoarseAlignmentResult:
    if not bool(reg_params.get("coarse_image_enabled", False)):
        return ImageCoarseAlignmentResult(
            transform=AffineTransform(),
            used=False,
            rotation_deg=0.0,
            translation_xy=np.zeros(2, dtype=float),
            score=float("nan"),
            error=float("nan"),
            evaluated_angles=0,
            elapsed_seconds=0.0,
        )

    start = time.perf_counter()
    fixed_gray = _normalize_for_image_coarse_alignment(reference_img)
    moving_gray = _normalize_for_image_coarse_alignment(source_img)
    fixed_small, moving_small, scale = _resize_for_image_coarse_alignment(
        fixed_gray,
        moving_gray,
        int(reg_params["coarse_image_target_max_dim"]),
    )

    crop_ratio = float(reg_params["coarse_image_crop_ratio"])
    upsample_factor = max(1, int(reg_params["coarse_image_upsample_factor"]))
    min_score = float(reg_params["coarse_image_min_score"])

    fixed_crop, moving_crop = _central_overlap_crops(fixed_small, moving_small, crop_ratio)
    if min(fixed_crop.shape[0], fixed_crop.shape[1], moving_crop.shape[0], moving_crop.shape[1]) < 16:
        elapsed = time.perf_counter() - start
        return ImageCoarseAlignmentResult(
            transform=AffineTransform(),
            used=False,
            rotation_deg=0.0,
            translation_xy=np.zeros(2, dtype=float),
            score=float("nan"),
            error=float("nan"),
            evaluated_angles=0,
            elapsed_seconds=elapsed,
        )

    try:
        shift_yx, error, _phase = phase_cross_correlation(
            fixed_crop,
            moving_crop,
            upsample_factor=upsample_factor,
        )
    except Exception:
        elapsed = time.perf_counter() - start
        return ImageCoarseAlignmentResult(
            transform=AffineTransform(),
            used=False,
            rotation_deg=0.0,
            translation_xy=np.zeros(2, dtype=float),
            score=float("nan"),
            error=float("nan"),
            evaluated_angles=1,
            elapsed_seconds=elapsed,
        )

    shifted_moving = ndimage_shift(
        moving_small,
        shift=np.asarray(shift_yx, dtype=float),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    fixed_eval, moving_eval = _central_overlap_crops(fixed_small, shifted_moving, crop_ratio)
    score = _normalized_cross_correlation(fixed_eval, moving_eval)

    elapsed = time.perf_counter() - start
    if not np.isfinite(score) or score < min_score:
        return ImageCoarseAlignmentResult(
            transform=AffineTransform(),
            used=False,
            rotation_deg=0.0,
            translation_xy=np.zeros(2, dtype=float),
            score=float(score),
            error=float(error),
            evaluated_angles=1,
            elapsed_seconds=elapsed,
        )

    translation_xy = np.asarray([shift_yx[1], shift_yx[0]], dtype=float) / float(max(scale, 1e-8))
    translation_transform = AffineTransform(translation=(float(translation_xy[0]), float(translation_xy[1])))
    return ImageCoarseAlignmentResult(
        transform=translation_transform,
        used=True,
        rotation_deg=0.0,
        translation_xy=translation_xy,
        score=float(score),
        error=float(error),
        evaluated_angles=1,
        elapsed_seconds=elapsed,
    )


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


def _log_message(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is None:
        print(message)
    else:
        logger(message)


def _empty_match_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "idx_fixed",
            "idx_moving",
            "cell_id_fixed",
            "cell_id_moving",
            "distance",
            "confidence",
            "inlier",
            "residual_px",
            "match_stage",
        ]
    )


def _build_final_match_table(
    matches: pd.DataFrame,
    feats_fixed: pd.DataFrame,
    feats_moving: pd.DataFrame,
    transform: AffineTransform,
    *,
    inliers: np.ndarray | None,
    match_stage: str,
) -> pd.DataFrame:
    if matches.empty:
        return _empty_match_table()

    table = matches.copy().reset_index(drop=True)
    pts_fixed = feats_fixed.loc[table["idx1"], ["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    pts_moving = feats_moving.loc[table["idx2"], ["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    residuals = np.linalg.norm(transform(pts_moving) - pts_fixed, axis=1)

    if inliers is None or len(inliers) != len(table):
        inlier_flags = np.zeros(len(table), dtype=bool)
    else:
        inlier_flags = np.asarray(inliers, dtype=bool)

    if "confidence" not in table.columns:
        table["confidence"] = np.nan

    table = table.rename(
        columns={
            "idx1": "idx_fixed",
            "idx2": "idx_moving",
            "cell_id_1": "cell_id_fixed",
            "cell_id_2": "cell_id_moving",
        }
    )
    table["inlier"] = inlier_flags
    table["residual_px"] = residuals
    table["match_stage"] = match_stage
    return table


def _transform_summary(transform: AffineTransform) -> dict[str, Any]:
    scale = np.asarray(transform.scale, dtype=float)
    translation = np.asarray(transform.translation, dtype=float)
    return {
        "affine_matrix": np.asarray(transform.params, dtype=float).tolist(),
        "translation_xy": translation.tolist(),
        "rotation_deg": float(np.degrees(float(transform.rotation))),
        "scale_xy": scale.tolist(),
        "shear_deg": float(np.degrees(float(transform.shear))),
    }


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
    match_table: pd.DataFrame | None = None,
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

    diagnostics = dict(diagnostics)
    diagnostics.update(_transform_summary(transform))
    diagnostics["valid_overlap_ratio"] = float(np.mean(valid_mask)) if valid_mask.size else float("nan")

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
        match_table=_empty_match_table() if match_table is None else match_table,
    )


def run_registration(
    source_artifacts: SegmentationArtifacts,
    reference_artifacts: SegmentationArtifacts,
    reg_params: dict[str, Any],
    *,
    logger: Callable[[str], None] | None = None,
) -> RegistrationArtifacts:
    """
    Run the registration pipeline using precomputed source/reference masks and features.
    """
    ensure_registration_dependencies()
    source_img = source_artifacts.image
    reference_img = reference_artifacts.image
    mask_src = source_artifacts.mask
    mask_ref = reference_artifacts.mask
    feats_src = source_artifacts.features
    feats_ref = reference_artifacts.features

    t_start = time.perf_counter()
    n_src = len(feats_src)
    n_ref = len(feats_ref)
    _log_message(logger, f"    Cells found: source={n_src}, reference={n_ref}")

    reference_shape = tuple(int(v) for v in to_2d_gray(reference_img).shape[:2])
    stage_timings = {
        "image_coarse_seconds": 0.0,
        "matching_seconds": 0.0,
        "transform_estimation_seconds": 0.0,
        "knn_refine_seconds": 0.0,
        "fft_refine_seconds": 0.0,
    }

    image_coarse = _estimate_image_coarse_alignment(source_img, reference_img, reg_params)
    stage_timings["image_coarse_seconds"] = image_coarse.elapsed_seconds
    if image_coarse.used:
        _log_message(
            logger,
            "    Image coarse init: "
            f"rotation={image_coarse.rotation_deg:.2f} deg, "
            f"dx={image_coarse.translation_xy[0]:.2f}, dy={image_coarse.translation_xy[1]:.2f}, "
            f"score={image_coarse.score:.3f}",
        )
        feats_src_for_matching = apply_transform_to_features(feats_src, image_coarse.transform, reference_shape)
    else:
        if int(image_coarse.evaluated_angles) > 0:
            _log_message(
                logger,
                "    Image coarse init skipped: "
                f"best score={image_coarse.score:.3f}, evaluated angles={image_coarse.evaluated_angles}",
            )
        feats_src_for_matching = feats_src

    if n_src < 3 or n_ref < 3:
        _log_message(logger, "    Too few cells for registration, falling back to identity transform")
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
            match_table=_empty_match_table(),
            diagnostics={
                "transform_method": "insufficient_cells",
                "source_cell_count": int(n_src),
                "reference_cell_count": int(n_ref),
                "coarse_match_count": 0,
                "fine_match_count": 0,
                "validated_match_count": 0,
                "guided_match_count": 0,
                "final_match_count": 0,
                "match_stage": "identity",
                "inlier_count": 0,
                "image_coarse_used": bool(image_coarse.used),
                "image_coarse_rotation_deg": float(image_coarse.rotation_deg),
                "image_coarse_translation_xy": image_coarse.translation_xy.tolist(),
                "image_coarse_score": float(image_coarse.score),
                "image_coarse_error": float(image_coarse.error),
                "image_coarse_evaluated_angles": int(image_coarse.evaluated_angles),
                "coarse_offset_xy": [0.0, 0.0],
                "median_inlier_residual": float("inf"),
                "mean_inlier_residual": float("inf"),
                "fft_shift_yx": [0.0, 0.0],
                "guided_residual_clip_used": False,
                "guided_residual_clip_limit": float("nan"),
                "guided_residual_clip_kept_inliers": 0,
                "guided_residual_clip_median_before": float("nan"),
                "guided_residual_clip_median_after": float("nan"),
                **stage_timings,
            },
        )

    matching_start = time.perf_counter()
    match_result = two_stage_match_cells(
        feats_ref,
        feats_src_for_matching,
        mask_ref.shape,
        feature_weight=reg_params["feature_weight"],
        topology_weight=reg_params["topology_weight"],
        position_weight=reg_params["position_weight"],
        top_k=reg_params["top_k"],
        distance_threshold=reg_params["distance_threshold"],
        spatial_window_size=reg_params["spatial_window_size"],
        min_cells_for_two_stage=reg_params["min_cells_for_two_stage"],
        coarse_top_k=reg_params["coarse_top_k"],
        coarse_distance_threshold=reg_params["coarse_distance_threshold"],
        coarse_matching_mode=reg_params["coarse_matching_mode"],
        coarse_patch_rows=reg_params["coarse_patch_rows"],
        coarse_patch_cols=reg_params["coarse_patch_cols"],
        coarse_patch_top_k_per_patch=reg_params["coarse_patch_top_k_per_patch"],
    )
    coarse_match_count = int(len(match_result.coarse_matches))
    coarse_offset_xy = match_result.coarse_offset_xy
    aligned_feats_src = match_result.aligned_df2
    if coarse_match_count > 0:
        _log_message(logger, f"    Coarse matches: {coarse_match_count}")
        _log_message(
            logger,
            "    Coarse offset: "
            f"dx={coarse_offset_xy[0]:.2f}, dy={coarse_offset_xy[1]:.2f}",
        )
    matches = match_result.matches
    fine_match_count = int(len(matches))
    _log_message(logger, f"    Fine matches: {fine_match_count}")

    config_val = MatchingConfig(
        feature_weight=reg_params["feature_weight"],
        topology_weight=reg_params["topology_weight"],
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
    matches = validate_matches(matches, feats_ref, aligned_feats_src, config_val)
    validated_match_count = int(len(matches))
    _log_message(logger, f"    Validated matches: {validated_match_count}")
    stage_timings["matching_seconds"] = time.perf_counter() - matching_start

    if len(matches) < 3:
        _log_message(logger, "    Too few matches for transform estimation, falling back to identity transform")
        elapsed = time.perf_counter() - t_start
        match_table = _build_final_match_table(
            matches,
            feats_ref,
            feats_src,
            AffineTransform(),
            inliers=None,
            match_stage="validated",
        )
        return _finalize_registration(
            source_img=source_img,
            reference_img=reference_img,
            source_mask=mask_src,
            reference_mask=mask_ref,
            source_features=feats_src,
            reference_features=feats_ref,
            transform=AffineTransform(),
            elapsed_seconds=elapsed,
            match_table=match_table,
            diagnostics={
                "transform_method": "insufficient_matches",
                "source_cell_count": int(n_src),
                "reference_cell_count": int(n_ref),
                "coarse_match_count": coarse_match_count,
                "fine_match_count": fine_match_count,
                "validated_match_count": validated_match_count,
                "guided_match_count": 0,
                "final_match_count": int(len(matches)),
                "match_stage": "validated",
                "inlier_count": 0,
                "image_coarse_used": bool(image_coarse.used),
                "image_coarse_rotation_deg": float(image_coarse.rotation_deg),
                "image_coarse_translation_xy": image_coarse.translation_xy.tolist(),
                "image_coarse_score": float(image_coarse.score),
                "image_coarse_error": float(image_coarse.error),
                "image_coarse_evaluated_angles": int(image_coarse.evaluated_angles),
                "coarse_offset_xy": coarse_offset_xy.tolist(),
                "median_inlier_residual": float("inf"),
                "mean_inlier_residual": float("inf"),
                "fft_shift_yx": [0.0, 0.0],
                "guided_residual_clip_used": False,
                "guided_residual_clip_limit": float("nan"),
                "guided_residual_clip_kept_inliers": 0,
                "guided_residual_clip_median_before": float("nan"),
                "guided_residual_clip_median_after": float("nan"),
                **stage_timings,
            },
        )

    transform_start = time.perf_counter()
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
    _log_message(
        logger,
        f"    RANSAC model: {robust.method} | "
        f"inliers: {robust.inlier_count}/{len(matches)} | "
        f"median residual: {robust.median_inlier_residual:.2f}px",
    )
    stage_timings["transform_estimation_seconds"] = time.perf_counter() - transform_start

    guided_match_count = 0
    final_match_stage = "validated"
    residual_clip_diag = {
        "guided_residual_clip_used": False,
        "guided_residual_clip_limit": float("nan"),
        "guided_residual_clip_kept_inliers": 0,
        "guided_residual_clip_median_before": float("nan"),
        "guided_residual_clip_median_after": float("nan"),
    }
    guided_matching_start = time.perf_counter()
    guided_feats_src = apply_transform_to_features(feats_src, affine, reference_shape)
    guided_spatial_cap = reg_params["guided_spatial_window_cap"]
    spatial_window = reg_params["spatial_window_size"]
    if guided_spatial_cap is None or guided_spatial_cap <= 0:
        guided_window = spatial_window
    else:
        guided_window = spatial_window if spatial_window is None else min(spatial_window, guided_spatial_cap)
    guided_top_k = reg_params["guided_top_k"]
    if guided_top_k is None or guided_top_k <= 0:
        guided_top_k = reg_params["top_k"]
    guided_distance_threshold = reg_params["distance_threshold"]
    if guided_distance_threshold is not None:
        guided_distance_threshold += reg_params["guided_distance_relaxation"]
    guided_config = MatchingConfig(
        feature_weight=reg_params["feature_weight"],
        topology_weight=reg_params["topology_weight"],
        position_weight=max(reg_params["position_weight"], reg_params["guided_min_position_weight"]),
        top_k=guided_top_k,
        distance_threshold=guided_distance_threshold,
        spatial_window_size=guided_window,
    )
    guided_matching_mode = str(reg_params.get("guided_matching_mode", "global")).strip().lower()
    guided_matches = _match_cells_with_strategy(
        feats_ref,
        guided_feats_src,
        guided_config,
        mode=guided_matching_mode,
        image_shape=reference_shape,
        patch_rows=int(reg_params.get("guided_patch_rows", 4)),
        patch_cols=int(reg_params.get("guided_patch_cols", 4)),
        patch_top_k_per_patch=reg_params.get("guided_patch_top_k_per_patch"),
    )
    guided_matches = validate_matches(guided_matches, feats_ref, guided_feats_src, config_val)
    guided_match_count = int(len(guided_matches))
    if guided_matching_mode == "patch":
        _log_message(
            logger,
            "    Guided matcher: "
            f"patch mode ({reg_params.get('guided_patch_rows', 4)}x{reg_params.get('guided_patch_cols', 4)}) "
            f"-> {guided_match_count} validated matches",
        )
    stage_timings["matching_seconds"] += time.perf_counter() - guided_matching_start
    if guided_match_count >= 3:
        guided_transform_start = time.perf_counter()
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
            final_match_stage = "guided_rematch"
            _log_message(
                logger,
                f"    Guided rematch improved support: {robust.method}, "
                f"{robust.inlier_count}/{len(matches)} inliers",
            )
        stage_timings["transform_estimation_seconds"] += time.perf_counter() - guided_transform_start

    if final_match_stage == "guided_rematch":
        clipped_robust, residual_clip_diag = _maybe_apply_guided_residual_clip(
            robust,
            pts_ref_xy,
            pts_src_xy,
            reg_params,
        )
        if clipped_robust is not robust:
            robust = clipped_robust
            affine = robust.transform
            _log_message(
                logger,
                "    Guided residual clip accepted: "
                f"{robust.inlier_count}/{len(matches)} inliers | "
                f"median residual {robust.median_inlier_residual:.2f}px",
            )

    if robust.inlier_count >= 3:
        pts_ref_xy_inliers = pts_ref_xy[robust.inliers]
        pts_src_xy_inliers = pts_src_xy[robust.inliers]
    else:
        pts_ref_xy_inliers = np.empty((0, 2), dtype=float)
        pts_src_xy_inliers = np.empty((0, 2), dtype=float)

    if len(pts_ref_xy_inliers) >= 3:
        knn_start = time.perf_counter()
        _log_message(logger, f"    Refining with k={reg_params['neighbor_k']} neighbors (KNN)...")
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
        stage_timings["knn_refine_seconds"] = time.perf_counter() - knn_start
        _log_message(logger, "    KNN refinement complete")

    # FFT-based phase-correlation refinement was removed from the runtime.
    fft_shift_yx = np.zeros(2, dtype=float)
    stage_timings["fft_refine_seconds"] = 0.0

    elapsed = time.perf_counter() - t_start
    match_table = _build_final_match_table(
        matches,
        feats_ref,
        feats_src,
        affine,
        inliers=robust.inliers,
        match_stage=final_match_stage,
    )
    return _finalize_registration(
        source_img=source_img,
        reference_img=reference_img,
        source_mask=mask_src,
        reference_mask=mask_ref,
        source_features=feats_src,
        reference_features=feats_ref,
        transform=affine,
        elapsed_seconds=elapsed,
        match_table=match_table,
        diagnostics={
            "transform_method": robust.method,
            "source_cell_count": int(n_src),
            "reference_cell_count": int(n_ref),
            "coarse_match_count": coarse_match_count,
            "fine_match_count": fine_match_count,
            "validated_match_count": validated_match_count,
            "guided_match_count": guided_match_count,
            "final_match_count": int(len(matches)),
            "match_stage": final_match_stage,
            "inlier_count": int(robust.inlier_count),
            "guided_matching_mode": guided_matching_mode,
            "image_coarse_used": bool(image_coarse.used),
            "image_coarse_rotation_deg": float(image_coarse.rotation_deg),
            "image_coarse_translation_xy": image_coarse.translation_xy.tolist(),
            "image_coarse_score": float(image_coarse.score),
            "image_coarse_error": float(image_coarse.error),
            "image_coarse_evaluated_angles": int(image_coarse.evaluated_angles),
            "coarse_offset_xy": coarse_offset_xy.tolist(),
            "median_inlier_residual": float(robust.median_inlier_residual),
            "mean_inlier_residual": float(robust.mean_inlier_residual),
            "fft_shift_yx": fft_shift_yx.tolist(),
            **residual_clip_diag,
            **stage_timings,
        },
    )


def load_segmentation_artifacts(
    image_path: Path,
    segmenter: CellposeSegmenter,
    cache: dict[Path, SegmentationArtifacts],
) -> tuple[SegmentationArtifacts, bool]:
    if image_path in cache:
        return cache[image_path], True

    ensure_registration_dependencies()
    start = time.perf_counter()
    image = imread(str(image_path))
    mask, _, _ = segmenter.segment_array(image)
    features = compute_cell_features(mask, CellFeaturesConfig())
    artifacts = SegmentationArtifacts(
        image_path=image_path,
        image=image,
        mask=mask,
        features=features,
        segmentation_seconds=time.perf_counter() - start,
    )
    cache[image_path] = artifacts
    return artifacts, False


def get_or_create_segmentation(
    image_path: Path,
    segmenter: CellposeSegmenter,
    cache: dict[Path, SegmentationArtifacts],
) -> SegmentationArtifacts:
    artifacts, _cache_hit = load_segmentation_artifacts(image_path, segmenter, cache)
    return artifacts
