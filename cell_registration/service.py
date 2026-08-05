"""UI-independent registration services used by the TopoAlign CLI.

The service layer deliberately imports only the root ``cell_registration``
implementation.  The napari plugin and benchmark packages are not imported
here and remain unchanged.
"""

from __future__ import annotations

import json
import time
from contextlib import redirect_stdout
from io import StringIO
from dataclasses import replace
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pandas as pd
import tifffile
from skimage.transform import AffineTransform, estimate_transform

from .cli_config import TopoAlignConfig
from .config import DEFAULT_CELLPOSE_CONFIG, DEFAULT_FEATURE_CONFIG
from .features import compute_cell_features
from .io_utils import infer_image_mode, load_image, project_intensity_max, save_mask
from .matching import MatchingConfig, match_cells_per_cluster, two_stage_match_cells
from .results import RegistrationResult, StageTimer
from .warp import valid_overlap_mask, warp_array


def _load_table(path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    table = table.reset_index(drop=True)
    required = {"centroid_x", "centroid_y"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Feature table {path} is missing columns: {', '.join(missing)}")
    if "cell_id" not in table.columns:
        table.insert(0, "cell_id", np.arange(len(table), dtype=int))
    return table


def _load_mask(path: str | Path) -> np.ndarray:
    mask = np.asarray(iio.imread(path))
    if mask.ndim == 3:
        mask = np.max(mask, axis=0)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D or a Z-stack, got {mask.shape} from {path}.")
    if np.any(mask < 0):
        raise ValueError(f"Mask contains negative labels: {path}")
    return mask.astype(np.int32, copy=False)


def _registration_channel(img: np.ndarray, axis: str, channel: int) -> np.ndarray:
    if axis == "none":
        if img.ndim != 2:
            raise ValueError("channel_axis='none' requires a 2D image.")
        return img
    if axis == "first":
        if img.ndim != 3:
            raise ValueError("channel_axis='first' requires a C,Y,X image.")
        return img[channel % img.shape[0]]
    if axis == "last":
        if img.ndim != 3:
            raise ValueError("channel_axis='last' requires a Y,X,C image.")
        return img[..., channel % img.shape[-1]]
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[-1] <= 4:
        return img[..., channel % img.shape[-1]]
    raise ValueError(f"Cannot infer a 2D registration channel from {img.shape}; set --channel-axis.")


def _spatial_shape(img: np.ndarray, axis: str) -> tuple[int, int]:
    if img.ndim == 2:
        return (int(img.shape[0]), int(img.shape[1]))
    if axis == "first":
        return (int(img.shape[1]), int(img.shape[2]))
    if axis in ("last", "auto") and img.shape[-1] <= 4:
        return (int(img.shape[0]), int(img.shape[1]))
    if axis == "auto":
        return (int(img.shape[1]), int(img.shape[2]))
    raise ValueError("Cannot infer image spatial shape; set channel_axis to first or last.")


def _json_write(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target


def _matrix_payload(transform: AffineTransform, *, method: str) -> dict[str, Any]:
    return {
        "product": "TopoAlign",
        "method": method,
        "direction": "moving_to_fixed",
        "coordinate_space": "pixel_xy",
        "matrix": np.asarray(transform.params, dtype=float).tolist(),
    }


def _feature_config(config: TopoAlignConfig):
    return replace(
        DEFAULT_FEATURE_CONFIG,
        min_area=config.segmentation.min_area,
        max_area=config.segmentation.max_area,
    )


def _feature_shape(*tables: pd.DataFrame) -> tuple[int, int]:
    points = pd.concat([table[["centroid_x", "centroid_y"]] for table in tables], ignore_index=True)
    if points.empty:
        raise ValueError("Cannot infer image shape from empty feature tables.")
    return (
        max(1, int(np.ceil(points["centroid_y"].max())) + 1),
        max(1, int(np.ceil(points["centroid_x"].max())) + 1),
    )


def segment_image(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    channel_axis: str = "auto",
    registration_channel: int = -1,
    gpu: bool = False,
) -> tuple[np.ndarray, Path]:
    """Segment an image and write ``mask.tif`` plus metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    image = load_image(image_path)
    from .segmentation import CellposeSegmenter

    segmenter = CellposeSegmenter(replace(DEFAULT_CELLPOSE_CONFIG, gpu=gpu))
    if channel_axis == "auto" and infer_image_mode(image) == "3d_zstack":
        mask_3d, mask, _, _ = segmenter.segment_zstack(image)
        mask_shape = list(mask_3d.shape)
        mode = "3d_zstack"
    else:
        channel = _registration_channel(image, channel_axis, registration_channel)
        mask, _, _ = segmenter.segment_array(channel)
        mask_shape = list(mask.shape)
        mode = "2d"
    mask_path = out / "mask.tif"
    save_mask(mask_path, mask)
    metadata_path = _json_write(
        out / "segmentation.json",
        {
            "product": "TopoAlign",
            "source": str(Path(image_path).resolve()),
            "mode": mode,
            "mask_shape": mask_shape,
            "projected_mask_shape": list(mask.shape),
            "channel_axis": channel_axis,
            "registration_channel": registration_channel,
            "gpu": gpu,
        },
    )
    return np.asarray(mask), metadata_path


def extract_features(
    mask_path: str | Path,
    output_dir: str | Path,
    *,
    config: TopoAlignConfig | None = None,
) -> tuple[pd.DataFrame, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = config or TopoAlignConfig()
    mask = _load_mask(mask_path)
    features = compute_cell_features(mask, _feature_config(cfg))
    csv_path = out / "features.csv"
    features.to_csv(csv_path, index=False)
    metadata_path = _json_write(
        out / "features.json",
        {
            "product": "TopoAlign",
            "source_mask": str(Path(mask_path).resolve()),
            "shape": list(mask.shape),
            "cell_count": int(len(features)),
            "columns": list(features.columns),
        },
    )
    return features, metadata_path


def match_features(
    fixed_features: pd.DataFrame,
    moving_features: pd.DataFrame,
    image_shape: tuple[int, int],
    config: TopoAlignConfig,
) -> pd.DataFrame:
    """Run the existing root two-stage landmark matcher."""
    m = config.matching
    with redirect_stdout(StringIO()):
        result = two_stage_match_cells(
            fixed_features,
            moving_features,
            image_shape,
            feature_weight=m.feature_weight,
            topology_weight=m.topology_weight,
            position_weight=m.position_weight,
            top_k=m.top_k,
            distance_threshold=m.distance_threshold,
            spatial_window_size=m.spatial_window_size,
            coarse_top_k=max(24, m.top_k),
            coarse_distance_threshold=2.0,
            coarse_matching_mode="morphology_guided",
            coarse_allow_scale=False,
            coarse_prefer_affine=False,
            coarse_residual_threshold=max(5.0, config.transform.ransac_residual_threshold * 2.0),
            coarse_max_trials=min(max(config.transform.ransac_max_trials, 200), 2000),
            initial_coarse_transform=_load_initial_matrix(config.transform.initial_coarse_transform),
            initial_coarse_transform_method="json_coarse_affine",
        )
    matches = result.matches.copy()
    if m.use_spatial_clusters:
        from .main import assign_clusters_from_round1

        f_work, s_work = assign_clusters_from_round1(fixed_features, result.aligned_df2, m.n_clusters)
        with redirect_stdout(StringIO()):
            matches = match_cells_per_cluster(
                f_work,
                s_work,
                MatchingConfig(
                    feature_weight=m.feature_weight,
                    topology_weight=m.topology_weight,
                    position_weight=m.position_weight,
                    top_k=m.top_k,
                    distance_threshold=m.distance_threshold,
                    spatial_window_size=m.spatial_window_size,
                ),
            )
    return matches.reset_index(drop=True)


def _load_initial_matrix(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    matrix = payload
    if isinstance(payload, dict):
        for key in ("matrix", "moving_to_fixed", "affine", "transform"):
            if key in payload:
                matrix = payload[key]
                break
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"Initial transform must be 3x3, got {matrix.shape}.")
    return matrix


def estimate_transform_from_matches(
    fixed_features: pd.DataFrame,
    moving_features: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    method: str = "rigid",
    use_ransac: bool = False,
    ransac_max_trials: int = 1000,
    ransac_residual_threshold: float = 2.0,
    residual_prune_quantile: float | None = None,
) -> tuple[AffineTransform, pd.DataFrame]:
    if matches.empty:
        raise ValueError("No matches are available for transform estimation.")
    method_key = {"rigid": "euclidean", "similarity": "similarity", "affine": "affine"}.get(method)
    if method_key is None:
        raise ValueError(f"Unsupported transform method: {method}")
    minimum = 3 if method == "affine" else 2
    if len(matches) < minimum:
        raise ValueError(f"At least {minimum} matches are required for {method} transform estimation.")

    def points(frame: pd.DataFrame) -> np.ndarray:
        return frame[["centroid_x", "centroid_y"]].to_numpy(dtype=float)

    def fit(current: pd.DataFrame) -> AffineTransform:
        moving = points(moving_features.iloc[current["idx2"].to_numpy(dtype=int)])
        fixed = points(fixed_features.iloc[current["idx1"].to_numpy(dtype=int)])
        if method == "rigid" and use_ransac:
            from .main import estimate_rigid_transform_from_matches_ransac

            rigid, _ = estimate_rigid_transform_from_matches_ransac(
                fixed_features,
                moving_features,
                current,
                max_trials=ransac_max_trials,
                residual_threshold=ransac_residual_threshold,
            )
            matrix = np.eye(3, dtype=float)
            matrix[:2, :2] = rigid.rotation
            matrix[:2, 2] = rigid.translation
            return AffineTransform(matrix=matrix)
        return AffineTransform(matrix=estimate_transform(method_key, moving, fixed).params)

    transform = fit(matches)
    moving_all = points(moving_features.iloc[matches["idx2"].to_numpy(dtype=int)])
    fixed_all = points(fixed_features.iloc[matches["idx1"].to_numpy(dtype=int)])
    residuals = np.linalg.norm(transform(moving_all) - fixed_all, axis=1)
    out = matches.copy()
    out["residual_px"] = residuals
    if residual_prune_quantile is not None and 0 < residual_prune_quantile < 1 and len(out) >= 3:
        threshold = float(np.quantile(residuals, residual_prune_quantile))
        keep = residuals <= threshold
        if int(keep.sum()) >= 3 and int(keep.sum()) < len(out):
            out = out.loc[keep].reset_index(drop=True)
            transform = fit(out)
            moving_all = points(moving_features.iloc[out["idx2"].to_numpy(dtype=int)])
            fixed_all = points(fixed_features.iloc[out["idx1"].to_numpy(dtype=int)])
            out["residual_px"] = np.linalg.norm(transform(moving_all) - fixed_all, axis=1)
    return transform, out


def _normalise_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    lo, hi = np.nanpercentile(arr, [1, 99]) if arr.size else (0.0, 1.0)
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def _save_overlay(fixed: np.ndarray, registered: np.ndarray, path: Path) -> None:
    fixed_2d = project_intensity_max(fixed) if fixed.ndim != 2 else fixed
    moving_2d = project_intensity_max(registered) if registered.ndim != 2 else registered
    if fixed_2d.shape != moving_2d.shape:
        return
    overlay = np.stack([_normalise_image(fixed_2d), _normalise_image(moving_2d), _normalise_image(fixed_2d)], axis=-1)
    tifffile.imwrite(path, overlay, metadata={"axes": "YXC"})


def register(config: TopoAlignConfig) -> RegistrationResult:
    """Run a complete CLI registration and write a result manifest."""
    out = Path(config.output.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = RegistrationResult(run_id=out.name)
    config.to_json(out / "config.resolved.json")
    started = time.perf_counter()

    fixed_image = load_image(config.fixed) if config.fixed else None
    moving_image = load_image(config.moving) if config.moving else None

    with StageTimer(result, "segmentation"):
        fixed_mask = _load_mask(config.fixed_mask) if config.fixed_mask else None
        moving_mask = _load_mask(config.moving_mask) if config.moving_mask else None
        if fixed_mask is None and not config.fixed_features:
            if fixed_image is None:
                raise ValueError("Either --fixed or --fixed-mask is required.")
            fixed_mask, _ = segment_image(
                config.fixed,
                out / "segmentation_fixed",
                channel_axis=config.segmentation.channel_axis,
                registration_channel=config.segmentation.registration_channel,
                gpu=config.segmentation.gpu,
            )
        if moving_mask is None and not config.moving_features:
            if moving_image is None:
                raise ValueError("Either --moving or --moving-mask is required.")
            moving_mask, _ = segment_image(
                config.moving,
                out / "segmentation_moving",
                channel_axis=config.segmentation.channel_axis,
                registration_channel=config.segmentation.registration_channel,
                gpu=config.segmentation.gpu,
            )

    with StageTimer(result, "features"):
        fixed_features = _load_table(config.fixed_features) if config.fixed_features else compute_cell_features(fixed_mask, _feature_config(config))
        moving_features = _load_table(config.moving_features) if config.moving_features else compute_cell_features(moving_mask, _feature_config(config))
        registration_shape = tuple(fixed_mask.shape) if fixed_mask is not None else config.fixed_shape or _feature_shape(fixed_features, moving_features)
        if config.output.save_features:
            fixed_features.to_csv(out / "fixed_features.csv", index=False)
            moving_features.to_csv(out / "moving_features.csv", index=False)
            result.artifacts.update({"fixed_features": str(out / "fixed_features.csv"), "moving_features": str(out / "moving_features.csv")})

    with StageTimer(result, "matching"):
        if config.matches:
            matches = pd.read_csv(config.matches).reset_index(drop=True)
        else:
            matches = match_features(fixed_features, moving_features, registration_shape, config)
        if config.output.save_matches:
            matches.to_csv(out / "matches.csv", index=False)
            result.artifacts["matches"] = str(out / "matches.csv")
        result.diagnostics["match_count"] = int(len(matches))

    with StageTimer(result, "transform"):
        transform, matches = estimate_transform_from_matches(
            fixed_features,
            moving_features,
            matches,
            method=config.transform.method,
            use_ransac=config.transform.use_ransac,
            ransac_max_trials=config.transform.ransac_max_trials,
            ransac_residual_threshold=config.transform.ransac_residual_threshold,
            residual_prune_quantile=config.transform.residual_prune_quantile,
        )
        transform_path = out / "transform.moving_to_fixed.json"
        _json_write(transform_path, _matrix_payload(transform, method=config.transform.method))
        result.artifacts["transform"] = str(transform_path)
        if not matches.empty:
            matches.to_csv(out / "matches.csv", index=False)
        result.diagnostics.update(
            {
                "transform_direction": "moving_to_fixed",
                "transform_matrix": np.asarray(transform.params).tolist(),
                "residual_mean_px": float(matches["residual_px"].mean()) if "residual_px" in matches and not matches.empty else None,
                "residual_max_px": float(matches["residual_px"].max()) if "residual_px" in matches and not matches.empty else None,
            }
        )

    warp_source = moving_image if moving_image is not None else moving_mask
    if warp_source is not None:
        with StageTimer(result, "warp"):
            fixed_output_shape = tuple(fixed_mask.shape) if fixed_mask is not None else config.fixed_shape or _feature_shape(fixed_features, moving_features)
            registered = warp_array(
                warp_source,
                transform,
                fixed_output_shape,
                channel_axis=config.segmentation.channel_axis if moving_image is not None else "none",
                order=1 if moving_image is not None else 0,
            )
            if config.output.save_registered_moving:
                registered_path = out / "registered_moving.tif"
                tifffile.imwrite(registered_path, registered)
                result.artifacts["registered_moving"] = str(registered_path)
            valid_path = out / "valid_overlap_mask.tif"
            source_axis = config.segmentation.channel_axis if moving_image is not None else "none"
            tifffile.imwrite(valid_path, valid_overlap_mask(_spatial_shape(warp_source, source_axis), transform, fixed_output_shape).astype(np.uint8))
            result.artifacts["valid_overlap_mask"] = str(valid_path)
            if config.output.save_overlay:
                overlay_path = out / "overlay.tif"
                _save_overlay(fixed_image if fixed_image is not None else fixed_mask, registered, overlay_path)
                if overlay_path.exists():
                    result.artifacts["overlay"] = str(overlay_path)
    result.diagnostics["elapsed_seconds"] = time.perf_counter() - started
    result.artifacts["config"] = str(out / "config.resolved.json")
    diagnostics_path = out / "diagnostics.json"
    _json_write(diagnostics_path, {"product": "TopoAlign", **result.diagnostics, "warnings": result.warnings, "errors": result.errors})
    result.artifacts["diagnostics"] = str(diagnostics_path)
    result.artifacts["result"] = str(out / "result.json")
    result.write(out / "result.json")
    return result
