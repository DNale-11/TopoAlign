"""
Batch registration runner for benchmark/unregistration case folders.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from registration_runtime import (
    DEFAULT_OBJECT_EVAL_PARAMS,
    DEFAULT_REG_PARAMS,
    build_segmenter,
    ensure_registration_dependencies,
    get_environment_snapshot,
    load_segmentation_artifacts,
    prepare_torch_runtime,
    run_registration,
    warp_image_with_transform,
)
from cell_registration.evaluation import compute_object_registration_metrics
from napari_cell_registration.core import compute_cell_features as compute_registered_cell_features
from napari_cell_registration.core.config import CellFeaturesConfig as CoreCellFeaturesConfig
from napari_cell_registration.core.config import MatchingConfig as CoreMatchingConfig
import numpy as np
import pandas as pd
from tifffile import imwrite

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = SCRIPT_DIR / "unregistration"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "unregistration_full"
DEFAULT_RESCUE_MIN_INLIERS = 10
DEFAULT_PROFILE_SELECTION_MODE = "single"
DEFAULT_ALT_FEATURE_WEIGHT = 0.85
DEFAULT_SELECTION_MIN_F1_GAIN = 0.0
DEFAULT_SELECTION_MAX_MASK_DICE_LOSS = 0.0005
DEFAULT_SELECTION_MAX_MATCHED_IOU_LOSS = 0.002
DEFAULT_SELECTION_MAX_CENTROID_MEDIAN_INCREASE = 0.05
DEFAULT_SELECTION_MAX_CENTROID_P95_INCREASE = 0.1
RESCUE_MATCHING_KEYS = (
    "feature_weight",
    "topology_weight",
    "position_weight",
    "distance_threshold",
    "spatial_window_size",
    "top_k",
    "min_cells_for_two_stage",
    "coarse_top_k",
    "coarse_distance_threshold",
    "coarse_matching_mode",
    "coarse_patch_rows",
    "coarse_patch_cols",
    "coarse_patch_top_k_per_patch",
    "guided_min_position_weight",
    "guided_spatial_window_cap",
    "guided_top_k",
    "guided_distance_relaxation",
    "guided_matching_mode",
    "guided_patch_rows",
    "guided_patch_cols",
    "guided_patch_top_k_per_patch",
    "validation_min_confidence",
    "validation_max_feature_diff",
    "validation_neighbor_k",
    "validation_max_neighbor_profile_diff",
    "validation_ambiguity_ratio",
    "validation_ambiguity_min_gap",
)


@dataclass(frozen=True)
class UnregistrationCase:
    case_id: str
    fixed_path: Path
    moving_paths: tuple[Path, ...]


@dataclass(frozen=True)
class UnregistrationPair:
    case_id: str
    fixed_path: Path
    moving_path: Path


def _parse_case_id_list(value: str) -> list[str]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of case ids.")
    return parts


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)


def _warp_mask_for_export(
    mask: np.ndarray,
    transform: Any,
    output_shape: tuple[int, int],
) -> np.ndarray:
    warped = warp_image_with_transform(mask.astype(np.int32), transform, output_shape, order=0)
    return np.rint(warped).astype(np.int32)


def _build_feature_export_payload(
    reg_params: dict[str, Any],
    object_eval_params: dict[str, Any],
    profile_selection_params: dict[str, Any],
) -> dict[str, Any]:
    matching_config = CoreMatchingConfig()
    feature_config = CoreCellFeaturesConfig()
    return {
        "matching_feature_columns": list(matching_config.feature_columns),
        "matching_topology_feature_columns": list(matching_config.topology_feature_columns),
        "cell_feature_config": {
            "min_area": feature_config.min_area,
            "max_area": feature_config.max_area,
            "extra_properties": list(feature_config.extra_properties),
            "topology_neighbor_k": feature_config.topology_neighbor_k,
        },
        "cellpose_config": {
            "pretrained_model": "cpsam",
            "gpu_enabled": bool(reg_params.get("gpu_enabled", True)),
            "diameter": reg_params["cellpose_diameter"],
            "flow_threshold": reg_params["cellpose_flow_threshold"],
            "cellprob_threshold": reg_params["cellpose_cellprob_threshold"],
            "min_size": reg_params["cellpose_min_size"],
        },
        "reg_params": reg_params,
        "object_eval_params": object_eval_params,
        "profile_selection_params": profile_selection_params,
    }


def _list_tiff_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        [
            path.resolve()
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        ],
        key=lambda path: path.name,
    )


def discover_unregistration_cases(
    input_root: Path,
    *,
    selected_case_ids: set[str] | None = None,
) -> tuple[list[UnregistrationCase], list[dict[str, Any]]]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    cases: list[UnregistrationCase] = []
    failures: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()

    for case_dir in sorted([path for path in input_root.iterdir() if path.is_dir()], key=lambda path: path.name):
        case_id = case_dir.name
        seen_case_ids.add(case_id)
        if selected_case_ids is not None and case_id not in selected_case_ids:
            continue

        fixed_files = _list_tiff_files(case_dir / "fixed")
        moving_files = _list_tiff_files(case_dir / "moving")

        if len(fixed_files) != 1:
            failures.append(
                {
                    "case_id": case_id,
                    "status": "invalid_case_layout",
                    "error_type": "fixed_count_mismatch",
                    "error_message": f"Expected exactly 1 fixed TIFF, found {len(fixed_files)}.",
                    "fixed_candidates": [str(path) for path in fixed_files],
                    "moving_candidates": [str(path) for path in moving_files],
                }
            )
            continue

        if not moving_files:
            failures.append(
                {
                    "case_id": case_id,
                    "status": "invalid_case_layout",
                    "error_type": "missing_moving",
                    "error_message": "No moving TIFF files found.",
                    "fixed_candidates": [str(path) for path in fixed_files],
                }
            )
            continue

        cases.append(
            UnregistrationCase(
                case_id=case_id,
                fixed_path=fixed_files[0],
                moving_paths=tuple(moving_files),
            )
        )

    if selected_case_ids is not None:
        missing_case_ids = sorted(selected_case_ids - seen_case_ids)
        for case_id in missing_case_ids:
            failures.append(
                {
                    "case_id": case_id,
                    "status": "missing_case",
                    "error_type": "case_not_found",
                    "error_message": f"Requested case id '{case_id}' was not found under {input_root}.",
                }
            )

    return cases, failures


def _features_for_export(features: pd.DataFrame) -> pd.DataFrame:
    exported = features.copy()
    if "centroid_x" in exported.columns and "x" not in exported.columns:
        exported["x"] = exported["centroid_x"]
    if "centroid_y" in exported.columns and "y" not in exported.columns:
        exported["y"] = exported["centroid_y"]
    return exported


def _pair_output_dir(output_dir: Path, pair: UnregistrationPair) -> Path:
    pair_name = f"{pair.moving_path.stem}_to_{pair.fixed_path.stem}"
    return output_dir / pair.case_id / pair_name


def _build_reg_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
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


def _registration_score(registration) -> tuple[int, float, float]:
    diagnostics = registration.diagnostics
    return (
        int(diagnostics.get("inlier_count", 0)),
        -float(diagnostics.get("median_inlier_residual", float("inf"))),
        -float(diagnostics.get("mean_inlier_residual", float("inf"))),
    )


def _build_rescue_reg_params(reg_params: dict[str, Any]) -> dict[str, Any]:
    rescue = dict(reg_params)
    for key in RESCUE_MATCHING_KEYS:
        rescue[key] = DEFAULT_REG_PARAMS[key]
    return rescue


def _run_registration_with_rescue(
    moving_artifacts,
    fixed_artifacts,
    reg_params: dict[str, Any],
    *,
    profile_name: str,
):
    registration = run_registration(
        moving_artifacts,
        fixed_artifacts,
        reg_params,
    )
    registration.diagnostics["selected_profile_name"] = profile_name
    registration.diagnostics["profile_family_name"] = profile_name
    registration.diagnostics["rescue_used"] = False
    registration.diagnostics["effective_reg_params"] = dict(reg_params)

    primary_inliers = int(registration.diagnostics.get("inlier_count", 0))
    if primary_inliers >= DEFAULT_RESCUE_MIN_INLIERS:
        return registration

    print(
        f"    Low inlier support ({primary_inliers}) under {profile_name}; "
        "retrying with baseline matching profile..."
    )
    rescue_params = _build_rescue_reg_params(reg_params)
    rescue_registration = run_registration(
        moving_artifacts,
        fixed_artifacts,
        rescue_params,
    )

    if _registration_score(rescue_registration) > _registration_score(registration):
        print(
            "    Baseline matching profile selected "
            f"({rescue_registration.diagnostics.get('inlier_count', 0)} inliers)"
        )
        rescue_name = "baseline_matching_rescue"
        if profile_name != "tuned_strict":
            rescue_name = f"{profile_name}__baseline_matching_rescue"
        rescue_registration.diagnostics["selected_profile_name"] = rescue_name
        rescue_registration.diagnostics["profile_family_name"] = profile_name
        rescue_registration.diagnostics["rescue_used"] = True
        rescue_registration.diagnostics["effective_reg_params"] = rescue_params
        rescue_registration.diagnostics["primary_profile_name"] = profile_name
        rescue_registration.diagnostics["primary_inlier_count"] = primary_inliers
        rescue_registration.diagnostics["primary_median_inlier_residual"] = float(
            registration.diagnostics.get("median_inlier_residual", float("inf"))
        )
        return rescue_registration

    print(f"    {profile_name} retained after rescue comparison")
    registration.diagnostics["rescue_used"] = True
    registration.diagnostics["effective_reg_params"] = dict(reg_params)
    registration.diagnostics["primary_profile_name"] = profile_name
    registration.diagnostics["primary_inlier_count"] = primary_inliers
    registration.diagnostics["primary_median_inlier_residual"] = float(
        registration.diagnostics.get("median_inlier_residual", float("inf"))
    )
    return registration


def _build_object_eval_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "instance_iou_threshold": args.instance_iou_threshold,
        "min_valid_instance_area_px": args.min_valid_instance_area_px,
        "min_valid_fraction": args.min_valid_fraction,
    }


def _build_profile_selection_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": args.profile_selection_mode,
        "alt_feature_weight": args.alt_feature_weight,
        "min_f1_gain": args.selection_min_f1_gain,
        "max_mask_dice_loss": args.selection_max_mask_dice_loss,
        "max_matched_mean_iou_loss": args.selection_max_matched_mean_iou_loss,
        "max_centroid_median_increase": args.selection_max_centroid_median_increase,
        "max_centroid_p95_increase": args.selection_max_centroid_p95_increase,
    }


def _metric_with_fallback(metrics: dict[str, Any], key: str, fallback: float) -> float:
    value = metrics.get(key, fallback)
    if pd.isna(value):
        return float(fallback)
    return float(value)


def _build_profile_variant_reg_params(
    reg_params: dict[str, Any],
    *,
    feature_weight: float,
) -> dict[str, Any]:
    variant = dict(reg_params)
    variant["feature_weight"] = float(feature_weight)
    return variant


def _profile_candidate_summary(
    *,
    registration,
    object_metrics: dict[str, float],
) -> dict[str, Any]:
    diagnostics = registration.diagnostics
    return {
        "selected_profile_name": diagnostics.get("selected_profile_name"),
        "profile_family_name": diagnostics.get("profile_family_name"),
        "rescue_used": bool(diagnostics.get("rescue_used", False)),
        "inlier_count": int(diagnostics.get("inlier_count", 0)),
        "median_inlier_residual": float(diagnostics.get("median_inlier_residual", float("nan"))),
        "mean_inlier_residual": float(diagnostics.get("mean_inlier_residual", float("nan"))),
        "match_f1": _metric_with_fallback(object_metrics, "match_f1", float("nan")),
        "matched_mean_iou": _metric_with_fallback(object_metrics, "matched_mean_iou", float("nan")),
        "mask_dice": _metric_with_fallback(object_metrics, "mask_dice", float("nan")),
        "centroid_error_median_px": _metric_with_fallback(
            object_metrics,
            "centroid_error_median_px",
            float("nan"),
        ),
        "centroid_error_p95_px": _metric_with_fallback(
            object_metrics,
            "centroid_error_p95_px",
            float("nan"),
        ),
    }


def _should_replace_with_object_metrics(
    current_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    selection_params: dict[str, Any],
) -> tuple[bool, dict[str, float]]:
    delta_f1 = _metric_with_fallback(candidate_metrics, "match_f1", -1.0) - _metric_with_fallback(
        current_metrics,
        "match_f1",
        -1.0,
    )
    delta_mask_dice = _metric_with_fallback(
        candidate_metrics,
        "mask_dice",
        -1.0,
    ) - _metric_with_fallback(current_metrics, "mask_dice", -1.0)
    delta_matched_mean_iou = _metric_with_fallback(
        candidate_metrics,
        "matched_mean_iou",
        -1.0,
    ) - _metric_with_fallback(current_metrics, "matched_mean_iou", -1.0)
    delta_centroid_median = _metric_with_fallback(
        candidate_metrics,
        "centroid_error_median_px",
        float("inf"),
    ) - _metric_with_fallback(current_metrics, "centroid_error_median_px", float("inf"))
    delta_centroid_p95 = _metric_with_fallback(
        candidate_metrics,
        "centroid_error_p95_px",
        float("inf"),
    ) - _metric_with_fallback(current_metrics, "centroid_error_p95_px", float("inf"))

    deltas = {
        "match_f1": float(delta_f1),
        "mask_dice": float(delta_mask_dice),
        "matched_mean_iou": float(delta_matched_mean_iou),
        "centroid_error_median_px": float(delta_centroid_median),
        "centroid_error_p95_px": float(delta_centroid_p95),
    }
    should_replace = (
        delta_f1 > float(selection_params["min_f1_gain"])
        and delta_mask_dice >= -float(selection_params["max_mask_dice_loss"])
        and delta_matched_mean_iou >= -float(selection_params["max_matched_mean_iou_loss"])
        and delta_centroid_median <= float(selection_params["max_centroid_median_increase"])
        and delta_centroid_p95 <= float(selection_params["max_centroid_p95_increase"])
    )
    return should_replace, deltas


def _run_registration_with_profile_selection(
    moving_artifacts,
    fixed_artifacts,
    reg_params: dict[str, Any],
    object_eval_params: dict[str, Any],
    profile_selection_params: dict[str, Any],
):
    registration = _run_registration_with_rescue(
        moving_artifacts,
        fixed_artifacts,
        reg_params,
        profile_name="tuned_strict",
    )
    object_metrics = compute_object_registration_metrics(
        moving_mask=registration.moving_mask,
        fixed_mask=registration.fixed_mask,
        moving_features=registration.moving_features,
        fixed_features=registration.fixed_features,
        transform=registration.transform,
        valid_mask=registration.valid_mask,
        instance_iou_threshold=object_eval_params["instance_iou_threshold"],
        min_valid_instance_area_px=object_eval_params["min_valid_instance_area_px"],
        min_valid_fraction=object_eval_params["min_valid_fraction"],
    )

    selection_mode = str(profile_selection_params.get("mode", DEFAULT_PROFILE_SELECTION_MODE)).strip().lower()
    candidate_summaries = [
        _profile_candidate_summary(
            registration=registration,
            object_metrics=object_metrics,
        )
    ]
    selection_details: dict[str, Any] = {
        "mode": selection_mode,
        "selected_by_object_metrics": False,
        "candidate_profiles": candidate_summaries,
    }

    alt_feature_weight = float(profile_selection_params.get("alt_feature_weight", 0.0))
    alt_enabled = selection_mode == "object_guarded" and alt_feature_weight > 0.0
    if not alt_enabled or abs(float(reg_params["feature_weight"]) - alt_feature_weight) < 1e-9:
        registration.diagnostics["profile_selection_mode"] = selection_mode
        registration.diagnostics["profile_selection_used"] = False
        registration.diagnostics["profile_selection_candidate_count"] = len(candidate_summaries)
        registration.diagnostics["profile_selection"] = selection_details
        return registration, object_metrics

    alt_params = _build_profile_variant_reg_params(
        reg_params,
        feature_weight=alt_feature_weight,
    )
    try:
        alt_registration = _run_registration_with_rescue(
            moving_artifacts,
            fixed_artifacts,
            alt_params,
            profile_name=f"feature_weight_{alt_feature_weight:.2f}".replace(".", ""),
        )
        alt_object_metrics = compute_object_registration_metrics(
            moving_mask=alt_registration.moving_mask,
            fixed_mask=alt_registration.fixed_mask,
            moving_features=alt_registration.moving_features,
            fixed_features=alt_registration.fixed_features,
            transform=alt_registration.transform,
            valid_mask=alt_registration.valid_mask,
            instance_iou_threshold=object_eval_params["instance_iou_threshold"],
            min_valid_instance_area_px=object_eval_params["min_valid_instance_area_px"],
            min_valid_fraction=object_eval_params["min_valid_fraction"],
        )
        candidate_summaries.append(
            _profile_candidate_summary(
                registration=alt_registration,
                object_metrics=alt_object_metrics,
            )
        )
        should_replace, selection_deltas = _should_replace_with_object_metrics(
            object_metrics,
            alt_object_metrics,
            profile_selection_params,
        )
        selection_details["selection_deltas_vs_primary"] = selection_deltas
        if should_replace:
            selection_details["selected_by_object_metrics"] = True
            alt_registration.diagnostics["profile_selection_mode"] = selection_mode
            alt_registration.diagnostics["profile_selection_used"] = True
            alt_registration.diagnostics["profile_selection_candidate_count"] = len(candidate_summaries)
            alt_registration.diagnostics["profile_selection"] = selection_details
            return alt_registration, alt_object_metrics
    except Exception as exc:
        selection_details["alternate_profile_error"] = {
            "profile_name": f"feature_weight_{alt_feature_weight:.2f}",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    registration.diagnostics["profile_selection_mode"] = selection_mode
    registration.diagnostics["profile_selection_used"] = False
    registration.diagnostics["profile_selection_candidate_count"] = len(candidate_summaries)
    registration.diagnostics["profile_selection"] = selection_details
    return registration, object_metrics


def _build_success_summary_row(
    *,
    output_dir: Path,
    pair: UnregistrationPair,
    registration,
    object_metrics: dict[str, float],
    fixed_seg_seconds: float,
    moving_seg_seconds: float,
    fixed_cache_hit: bool,
    moving_cache_hit: bool,
    total_pair_seconds: float,
) -> dict[str, Any]:
    diagnostics = registration.diagnostics
    translation_xy = diagnostics.get("translation_xy", [float("nan"), float("nan")])
    scale_xy = diagnostics.get("scale_xy", [float("nan"), float("nan")])
    artifact_dir = _pair_output_dir(output_dir, pair)

    row = {
        "case_id": pair.case_id,
        "fixed_name": pair.fixed_path.name,
        "moving_name": pair.moving_path.name,
        "status": "success",
        "artifact_dir": str(artifact_dir.relative_to(output_dir)),
        "fixed_cache_hit": fixed_cache_hit,
        "moving_cache_hit": moving_cache_hit,
        "fixed_cell_count": int(len(registration.fixed_features)),
        "moving_cell_count": int(len(registration.moving_features)),
        "fixed_segmentation_seconds": float(fixed_seg_seconds),
        "moving_segmentation_seconds": float(moving_seg_seconds),
        "image_coarse_seconds": float(diagnostics.get("image_coarse_seconds", 0.0)),
        "matching_seconds": float(diagnostics.get("matching_seconds", 0.0)),
        "transform_estimation_seconds": float(diagnostics.get("transform_estimation_seconds", 0.0)),
        "knn_refine_seconds": float(diagnostics.get("knn_refine_seconds", 0.0)),
        "registration_seconds": float(registration.elapsed_seconds),
        "total_pair_seconds": float(total_pair_seconds),
        "coarse_match_count": int(diagnostics.get("coarse_match_count", 0)),
        "fine_match_count": int(diagnostics.get("fine_match_count", 0)),
        "validated_match_count": int(diagnostics.get("validated_match_count", 0)),
        "guided_match_count": int(diagnostics.get("guided_match_count", 0)),
        "final_match_count": int(diagnostics.get("final_match_count", 0)),
        "inlier_count": int(diagnostics.get("inlier_count", 0)),
        "transform_method": diagnostics.get("transform_method"),
        "match_stage": diagnostics.get("match_stage"),
        "guided_matching_mode": diagnostics.get("guided_matching_mode", "global"),
        "selected_profile_name": diagnostics.get("selected_profile_name"),
        "rescue_used": bool(diagnostics.get("rescue_used", False)),
        "profile_selection_mode": diagnostics.get("profile_selection_mode", DEFAULT_PROFILE_SELECTION_MODE),
        "profile_selection_used": bool(diagnostics.get("profile_selection_used", False)),
        "profile_selection_candidate_count": int(diagnostics.get("profile_selection_candidate_count", 1)),
        "image_coarse_used": bool(diagnostics.get("image_coarse_used", False)),
        "image_coarse_rotation_deg": float(diagnostics.get("image_coarse_rotation_deg", 0.0)),
        "image_coarse_translation_x": float(diagnostics.get("image_coarse_translation_xy", [0.0, 0.0])[0]),
        "image_coarse_translation_y": float(diagnostics.get("image_coarse_translation_xy", [0.0, 0.0])[1]),
        "image_coarse_score": float(diagnostics.get("image_coarse_score", float("nan"))),
        "image_coarse_error": float(diagnostics.get("image_coarse_error", float("nan"))),
        "image_coarse_evaluated_angles": int(diagnostics.get("image_coarse_evaluated_angles", 0)),
        "translation_x": float(translation_xy[0]),
        "translation_y": float(translation_xy[1]),
        "rotation_deg": float(diagnostics.get("rotation_deg", float("nan"))),
        "scale_x": float(scale_xy[0]),
        "scale_y": float(scale_xy[1]),
        "shear_deg": float(diagnostics.get("shear_deg", float("nan"))),
        "median_inlier_residual": float(diagnostics.get("median_inlier_residual", float("nan"))),
        "mean_inlier_residual": float(diagnostics.get("mean_inlier_residual", float("nan"))),
    }
    row.update(object_metrics)
    return row


def _build_failure_row(
    *,
    pair: UnregistrationPair | None,
    case_id: str,
    status: str,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "fixed_name": pair.fixed_path.name if pair is not None else None,
        "moving_name": pair.moving_path.name if pair is not None else None,
        "status": status,
        "error_type": error_type,
        "error_message": error_message,
    }


def _build_pair_diagnostics(
    *,
    pair: UnregistrationPair,
    registration,
    fixed_artifacts,
    moving_artifacts,
    environment: dict[str, Any],
    reg_params: dict[str, Any],
    object_eval_params: dict[str, Any],
    object_metrics: dict[str, float],
    fixed_seg_seconds: float,
    moving_seg_seconds: float,
    fixed_cache_hit: bool,
    moving_cache_hit: bool,
    total_pair_seconds: float,
) -> dict[str, Any]:
    diagnostics = registration.diagnostics
    return {
        "case_id": pair.case_id,
        "fixed_path": str(pair.fixed_path),
        "moving_path": str(pair.moving_path),
        "input": {
            "fixed_name": pair.fixed_path.name,
            "moving_name": pair.moving_path.name,
            "fixed_shape": list(fixed_artifacts.image.shape),
            "moving_shape": list(moving_artifacts.image.shape),
            "fixed_dtype": str(fixed_artifacts.image.dtype),
            "moving_dtype": str(moving_artifacts.image.dtype),
        },
        "environment": environment,
        "gpu_enabled": bool(reg_params.get("gpu_enabled", True)),
        "reg_params": reg_params,
        "effective_reg_params": diagnostics.get("effective_reg_params", reg_params),
        "object_eval_params": object_eval_params,
        "segmentation": {
            "fixed_cells": int(len(fixed_artifacts.features)),
            "moving_cells": int(len(moving_artifacts.features)),
            "fixed_cache_hit": fixed_cache_hit,
            "moving_cache_hit": moving_cache_hit,
            "fixed_segmentation_seconds": float(fixed_seg_seconds),
            "moving_segmentation_seconds": float(moving_seg_seconds),
        },
        "matching": {
            "coarse_match_count": int(diagnostics.get("coarse_match_count", 0)),
            "fine_match_count": int(diagnostics.get("fine_match_count", 0)),
            "validated_match_count": int(diagnostics.get("validated_match_count", 0)),
            "guided_match_count": int(diagnostics.get("guided_match_count", 0)),
            "final_match_count": int(diagnostics.get("final_match_count", 0)),
            "inlier_count": int(diagnostics.get("inlier_count", 0)),
            "match_stage": diagnostics.get("match_stage"),
            "selected_profile_name": diagnostics.get("selected_profile_name"),
            "rescue_used": bool(diagnostics.get("rescue_used", False)),
            "profile_selection_mode": diagnostics.get("profile_selection_mode", DEFAULT_PROFILE_SELECTION_MODE),
            "profile_selection_used": bool(diagnostics.get("profile_selection_used", False)),
            "profile_selection_candidate_count": int(diagnostics.get("profile_selection_candidate_count", 1)),
        },
        "profile_selection": diagnostics.get("profile_selection"),
        "transform": {
            "transform_method": diagnostics.get("transform_method"),
            "affine_matrix": diagnostics.get("affine_matrix"),
            "translation_xy": diagnostics.get("translation_xy"),
            "rotation_deg": diagnostics.get("rotation_deg"),
            "scale_xy": diagnostics.get("scale_xy"),
            "shear_deg": diagnostics.get("shear_deg"),
        },
        "quality": {
            "median_inlier_residual": diagnostics.get("median_inlier_residual"),
            "mean_inlier_residual": diagnostics.get("mean_inlier_residual"),
            "valid_overlap_ratio": diagnostics.get("valid_overlap_ratio"),
            **object_metrics,
        },
        "timings": {
            "fixed_segmentation_seconds": float(fixed_seg_seconds),
            "moving_segmentation_seconds": float(moving_seg_seconds),
            "image_coarse_seconds": float(diagnostics.get("image_coarse_seconds", 0.0)),
            "matching_seconds": float(diagnostics.get("matching_seconds", 0.0)),
            "transform_estimation_seconds": float(diagnostics.get("transform_estimation_seconds", 0.0)),
            "knn_refine_seconds": float(diagnostics.get("knn_refine_seconds", 0.0)),
            "registration_seconds": float(registration.elapsed_seconds),
            "total_pair_seconds": float(total_pair_seconds),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-archive registration on benchmark/unregistration cases.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--case-ids",
        type=_parse_case_id_list,
        default=None,
        help="Optional comma-separated subset of case folders, e.g. A2-1,A4-1.",
    )
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument(
        "--profile-selection-mode",
        choices=("single", "object_guarded"),
        default=DEFAULT_PROFILE_SELECTION_MODE,
        help="Evaluate an alternate feature-weight profile and replace the primary result only when pair-level object metrics pass conservative gates.",
    )
    parser.add_argument(
        "--alt-feature-weight",
        type=float,
        default=DEFAULT_ALT_FEATURE_WEIGHT,
        help="Feature weight used by the alternate profile for object-guarded selection; <=0 disables the alternate profile.",
    )
    parser.add_argument(
        "--selection-min-f1-gain",
        type=float,
        default=DEFAULT_SELECTION_MIN_F1_GAIN,
    )
    parser.add_argument(
        "--selection-max-mask-dice-loss",
        type=float,
        default=DEFAULT_SELECTION_MAX_MASK_DICE_LOSS,
    )
    parser.add_argument(
        "--selection-max-matched-mean-iou-loss",
        type=float,
        default=DEFAULT_SELECTION_MAX_MATCHED_IOU_LOSS,
    )
    parser.add_argument(
        "--selection-max-centroid-median-increase",
        type=float,
        default=DEFAULT_SELECTION_MAX_CENTROID_MEDIAN_INCREASE,
    )
    parser.add_argument(
        "--selection-max-centroid-p95-increase",
        type=float,
        default=DEFAULT_SELECTION_MAX_CENTROID_P95_INCREASE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_torch_runtime()
    ensure_registration_dependencies()

    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reg_params = _build_reg_params(args)
    reg_params["gpu_enabled"] = bool(args.gpu)
    object_eval_params = _build_object_eval_params(args)
    profile_selection_params = _build_profile_selection_params(args)
    environment = get_environment_snapshot()
    selected_case_ids = set(args.case_ids) if args.case_ids is not None else None

    start_time = datetime.now().astimezone().isoformat()
    manifest = {
        "started_at": start_time,
        "ended_at": None,
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "command": [sys.executable, *sys.argv],
        "environment": environment,
        "reg_params": reg_params,
        "object_eval_params": object_eval_params,
        "profile_selection_params": profile_selection_params,
        "selected_case_ids": sorted(selected_case_ids) if selected_case_ids is not None else None,
        "discovered_case_count": 0,
        "discovered_pair_count": 0,
        "success_count": 0,
        "failure_count": 0,
    }
    manifest_path = output_dir / "run_manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(
        output_dir / "feature_params.json",
        _build_feature_export_payload(
            reg_params=reg_params,
            object_eval_params=object_eval_params,
            profile_selection_params=profile_selection_params,
        ),
    )

    print("=" * 78)
    print("  Unregistration Batch Registration")
    print("=" * 78)
    print(f"  Input root:  {input_root}")
    print(f"  Output dir:  {output_dir}")
    print(f"  GPU enabled: {bool(args.gpu)}")
    print(f"  Python env:  {environment.get('conda_default_env') or environment.get('sys_prefix')}")

    cases, discovery_failures = discover_unregistration_cases(
        input_root,
        selected_case_ids=selected_case_ids,
    )
    pair_total = sum(len(case.moving_paths) for case in cases)
    manifest["discovered_case_count"] = len(cases)
    manifest["discovered_pair_count"] = pair_total
    _write_json(manifest_path, manifest)

    print(f"  Discovered {len(cases)} valid cases and {pair_total} registration pairs")
    if discovery_failures:
        print(f"  Discovery failures: {len(discovery_failures)}")

    segmenter = build_segmenter(reg_params, gpu=bool(args.gpu))
    segmentation_cache = {}
    summary_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = list(discovery_failures)
    completed_pairs = 0

    for case_index, case in enumerate(cases, start=1):
        print(
            f"\n[{case_index}/{len(cases)}] Case {case.case_id} | "
            f"fixed={case.fixed_path.name} | moving={len(case.moving_paths)}"
        )

        try:
            fixed_artifacts, fixed_cache_hit = load_segmentation_artifacts(
                case.fixed_path,
                segmenter,
                segmentation_cache,
            )
        except Exception as exc:
            error_message = str(exc)
            print(f"  Fixed segmentation failed: {error_message}")
            for moving_path in case.moving_paths:
                pair = UnregistrationPair(case.case_id, case.fixed_path, moving_path)
                failure = {
                    **_build_failure_row(
                        pair=pair,
                        case_id=case.case_id,
                        status="failed",
                        error_type=type(exc).__name__,
                        error_message=error_message,
                    ),
                    "fixed_path": str(case.fixed_path),
                    "moving_path": str(moving_path),
                }
                failures.append(failure)
                summary_rows.append(failure)
            continue

        fixed_seg_seconds = 0.0 if fixed_cache_hit else float(fixed_artifacts.segmentation_seconds)
        print(
            f"  Fixed cells: {len(fixed_artifacts.features)} | "
            f"segmentation_seconds={fixed_seg_seconds:.2f} | cache_hit={fixed_cache_hit}"
        )

        for moving_index, moving_path in enumerate(case.moving_paths, start=1):
            completed_pairs += 1
            pair = UnregistrationPair(case.case_id, case.fixed_path, moving_path)
            pair_output_dir = _pair_output_dir(output_dir, pair)
            pair_start = time.perf_counter()

            print(
                f"  [{completed_pairs}/{pair_total}] "
                f"Moving {moving_index}/{len(case.moving_paths)}: {moving_path.name}"
            )

            try:
                moving_artifacts, moving_cache_hit = load_segmentation_artifacts(
                    moving_path,
                    segmenter,
                    segmentation_cache,
                )
                moving_seg_seconds = 0.0 if moving_cache_hit else float(moving_artifacts.segmentation_seconds)
                pair_fixed_seg_seconds = fixed_seg_seconds if moving_index == 1 else 0.0
                pair_fixed_cache_hit = fixed_cache_hit if moving_index == 1 else True
                print(
                    f"    Moving cells: {len(moving_artifacts.features)} | "
                    f"segmentation_seconds={moving_seg_seconds:.2f} | cache_hit={moving_cache_hit}"
                )

                registration, object_metrics = _run_registration_with_profile_selection(
                    moving_artifacts,
                    fixed_artifacts,
                    reg_params,
                    object_eval_params,
                    profile_selection_params,
                )

                pair_output_dir.mkdir(parents=True, exist_ok=True)
                registered_mask = _warp_mask_for_export(
                    registration.moving_mask,
                    registration.transform,
                    tuple(int(v) for v in registration.fixed_mask.shape),
                )
                registered_features = _features_for_export(
                    compute_registered_cell_features(registered_mask, CoreCellFeaturesConfig())
                )
                imwrite(str(pair_output_dir / "registered_to_fixed.tif"), registration.registered_image)
                imwrite(str(pair_output_dir / "registered_mask.tif"), registered_mask)
                imwrite(str(pair_output_dir / "valid_overlap_mask.tif"), registration.valid_mask.astype("uint8"))
                imwrite(str(pair_output_dir / "fixed_mask.tif"), registration.fixed_mask.astype("int32"))
                imwrite(str(pair_output_dir / "moving_mask.tif"), registration.moving_mask.astype("int32"))
                _features_for_export(registration.fixed_features).to_csv(pair_output_dir / "fixed_features.csv", index=False)
                _features_for_export(registration.moving_features).to_csv(pair_output_dir / "moving_features.csv", index=False)
                registered_features.to_csv(pair_output_dir / "registered_features.csv", index=False)
                registration.match_table.to_csv(pair_output_dir / "registration_matches.csv", index=False)
                _write_json(
                    pair_output_dir / "feature_params.json",
                    _build_feature_export_payload(
                        reg_params=reg_params,
                        object_eval_params=object_eval_params,
                        profile_selection_params=profile_selection_params,
                    ),
                )

                total_pair_seconds = time.perf_counter() - pair_start
                diagnostics_payload = _build_pair_diagnostics(
                    pair=pair,
                    registration=registration,
                    fixed_artifacts=fixed_artifacts,
                    moving_artifacts=moving_artifacts,
                    environment=environment,
                    reg_params=reg_params,
                    object_eval_params=object_eval_params,
                    object_metrics=object_metrics,
                    fixed_seg_seconds=pair_fixed_seg_seconds,
                    moving_seg_seconds=moving_seg_seconds,
                    fixed_cache_hit=pair_fixed_cache_hit,
                    moving_cache_hit=moving_cache_hit,
                    total_pair_seconds=total_pair_seconds,
                )
                _write_json(pair_output_dir / "diagnostics.json", diagnostics_payload)

                summary_rows.append(
                    _build_success_summary_row(
                        output_dir=output_dir,
                        pair=pair,
                        registration=registration,
                        object_metrics=object_metrics,
                        fixed_seg_seconds=pair_fixed_seg_seconds,
                        moving_seg_seconds=moving_seg_seconds,
                        fixed_cache_hit=pair_fixed_cache_hit,
                        moving_cache_hit=moving_cache_hit,
                        total_pair_seconds=total_pair_seconds,
                    )
                )
            except Exception as exc:
                error_message = str(exc)
                print(f"    Pair failed: {error_message}")
                failure = {
                    **_build_failure_row(
                        pair=pair,
                        case_id=case.case_id,
                        status="failed",
                        error_type=type(exc).__name__,
                        error_message=error_message,
                    ),
                    "fixed_path": str(case.fixed_path),
                    "moving_path": str(moving_path),
                }
                failures.append(failure)
                summary_rows.append(failure)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    failures_path = output_dir / "failures.json"
    _write_json(failures_path, failures)

    success_count = 0
    if "status" in summary_df.columns:
        success_count = int((summary_df["status"] == "success").sum())
    failure_count = len(failures)
    manifest["ended_at"] = datetime.now().astimezone().isoformat()
    manifest["success_count"] = success_count
    manifest["failure_count"] = failure_count
    _write_json(manifest_path, manifest)

    print("\nSummary")
    print("-" * 78)
    print(f"  Success pairs: {success_count}")
    print(f"  Failure records: {failure_count}")
    print(f"  Summary CSV: {summary_path}")
    print(f"  Failures JSON: {failures_path}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
